import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    def true_score(points):
        """Compute (d_min/d_max)^2 using squared distances."""
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 0.0
        return d_min_sq / d_max_sq

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def neg_score(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        return -true_score(pts)

    # =================================================================
    # NEW: Direct QP-style objective with barrier for max constraint
    # Maximize min pairwise sq-dist subject to max pairwise sq-dist <= 1
    # via log-barrier. We work in normalized space but treat the
    # constraint on the actual distances.
    # =================================================================
    def qp_barrier_obj(x_flat, mu=0.01):
        """
        Objective to minimize:
          -min_dist_sq + mu * sum(log(max(0, dist_sq_i - 1 + eps)))
        Actually, we want to maximize min_dist_sq subject to all dist_sq <= 1.
        Barrier term: -mu * sum(log(1 - dist_sq_i)) for dist_sq_i < 1.
        If any dist_sq >= 1, we add a large penalty.
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        
        # To make the barrier well-defined, we need to scale points such that
        # the constraint is meaningful. However, since we normalize RMS, the scale
        # is fixed. The "1" in the constraint is arbitrary relative to RMS=1.
        # Let's define the constraint relative to the current max distance.
        # Better approach: 
        # Objective: max (min_d / max_d). 
        # Let's just use the log-sum-exp smooth min/max which worked well, 
        # but combine it with a specific "QP-like" structure?
        
        # The prompt suggests: "maximizing the minimum pairwise squared distance 
        # subject to the maximum pairwise squared distance being fixed to 1, 
        # using a barrier function".
        
        # Since we have rotational/translational/scaling invariance, we can fix
        # the scaling by setting max_dist_sq = 1.
        # So:
        # 1. Compute dists_sq.
        # 2. d_max_sq = max(dists_sq).
        # 3. Scale points so that d_max_sq becomes 1. (This is a projection).
        # 4. Now the objective is simply max(min_d_sq).
        # 5. But we need to optimize the *shape* that maximizes min_d_sq under this scaling.
        
        # Let's implement this "shape optimization" directly.
        # f(x) = -min( dists_sq(x) / max(dists_sq(x)) )
        # This is equivalent to the original score, but the landscape might be
        # friendlier if we view it as:
        # Minimize -min(dists_sq) subject to max(dists_sq) = 1.
        
        # Let's try a smooth approximation of:
        # min(dists_sq) - mu * (max(dists_sq) - 1)^2  (if max > 1)
        # Or better:
        # We want to maximize r = min_d / max_d.
        # log(r) = log(min_d) - log(max_d).
        # Smooth log(min_d) ~ -T log(sum(exp(-d^2/T)))
        # Smooth log(max_d) ~ T log(sum(exp(d^2/T)))
        
        # This is essentially the LSE objective from the previous program, 
        # but the "inspiration" mentions QP. 
        # Let's look at the "inspiration 1" code snippet provided. It uses:
        # smooth_obj = sum(dists_sq ** (-power/2.0))
        # This is a p-norm approximation of the minimum distance (inverse p-norm).
        # Minimizing sum(1/d^p) maximizes the minimum distance.
        # This is a very smooth, convex-like function (in terms of distances) 
        # that drives all distances up, but since the space is bounded (RMS norm),
        # it will push points apart until they hit the "boundary" of the configuration space.
        
        # Let's combine:
        # 1. Use the inverse-power sum to maximize min distance (smooth, good gradients).
        # 2. Use the true score for final evaluation.
        # 3. Use Simulated Annealing (from knowledge) to escape local minima.
        
        dists_sq = np.maximum(dists_sq, 1e-10)
        # Minimize sum of inverse distances to push min distance up
        # Higher power = sharper focus on min distance
        return np.sum(dists_sq ** (-2.0))

    def qp_barrier_grad(x_flat):
        eps = 1e-7
        grad = np.zeros_like(x_flat)
        f0 = qp_barrier_obj(x_flat)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_plus[i] += eps
            f_plus = qp_barrier_obj(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    # =================================================================
    # Simulated Annealing Objective
    # =================================================================
    def sa_obj(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        return -true_score(pts)

    # Best tracking
    best_score = 0.0
    best_points = None

    def try_update(points):
        nonlocal best_score, best_points
        if points is None:
            return
        points = normalize_rms(points)
        if not np.all(np.isfinite(points)):
            return
        s = true_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # =================================================================
    # PHASE 1: Generate diverse initial candidates
    # =================================================================
    candidates = []

    # --- Model A: Two antipodal rings ---
    def make_anticipodal_rings(h, r1, r2, theta1, twist):
        angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta1
        ring1 = np.column_stack([
            r1 * np.cos(angles1),
            r1 * np.sin(angles1),
            np.full(7, h)
        ])
        angles2 = angles1 + twist
        ring2 = np.column_stack([
            r2 * np.cos(angles2),
            r2 * np.sin(angles2),
            np.full(7, -h)
        ])
        pts = np.vstack([ring1, ring2])
        return normalize_rms(pts)

    h_vals = np.linspace(0.0, 1.0, 15)
    r_ratio_vals = np.linspace(0.2, 1.0, 12)
    theta1_vals = np.linspace(0, 2*np.pi/7, 7, endpoint=False)
    twist_vals = np.linspace(0, 2*np.pi/7, 7, endpoint=False)

    for h in h_vals:
        if time.time() > deadline - 50:
            break
        for rr in r_ratio_vals:
            for th1 in theta1_vals:
                for tw in twist_vals:
                    pts = make_anticipodal_rings(h, 1.0, rr, th1, tw)
                    candidates.append(pts)

    # --- Model B: icosahedron + 2 poles ---
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(normalize_rms(np.vstack([ico, poles])))

    # --- Model C: Cube + octahedron ---
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    candidates.append(normalize_rms(np.vstack([cube, octa])))

    # --- Model D: Random antipodal ---
    for seed in range(20):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        pts = normalize_rms(np.vstack([pts_upper, pts_lower]))
        candidates.append(pts)

    # --- Model E: Fibonacci sphere points ---
    def fibonacci_sphere(n_pts):
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_pts):
            z = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(1.0 - z**2)
            theta = golden_angle * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points, dtype=float)

    fib_pts = fibonacci_sphere(14)
    fib_pts = fib_pts / np.linalg.norm(fib_pts, axis=1, keepdims=True)
    candidates.append(normalize_rms(fib_pts))

    for angle in [np.pi/6, np.pi/4, np.pi/3]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated = fib_pts @ rot.T
        candidates.append(normalize_rms(rotated))

    # --- Model F: Iterative repulsion ---
    for seed in range(10):
        rng = np.random.RandomState(1000 + seed)
        pts = rng.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        for it in range(200):
            dists = pdist(pts, metric='sqeuclidean')
            forces = np.zeros((n, 3))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-10:
                        continue
                    force = diff / (dist_sq * np.sqrt(dist_sq))
                    forces[i] += force
            pts = pts + 0.01 * forces
            pts = normalize_rms(pts)
        candidates.append(pts)

    # Score all candidates
    for cand in candidates:
        try_update(cand)

    # =================================================================
    # PHASE 2: Local optimization from best candidates
    # =================================================================
    if best_points is not None:
        polish_list = [best_points]

        cand_scores = [(true_score(c), c) for c in candidates]
        cand_scores.sort(key=lambda x: -x[0])
        for s, c in cand_scores[1:8]:
            if s > best_score * 0.75:
                polish_list.append(c)

        for init_pts in polish_list:
            if time.time() > deadline - 60:
                break

            init_pts = normalize_rms(init_pts)
            x0 = init_pts.flatten()

            # Step 2a: L-BFGS-B on smooth inverse-power objective
            try:
                res1 = minimize(
                    qp_barrier_obj,
                    x0,
                    jac=qp_barrier_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts1 = normalize_rms(res1.x.reshape(n, d))
            except Exception:
                pts1 = init_pts

            # Step 2b: Nelder-Mead on true objective
            try:
                res2 = minimize(
                    neg_score,
                    pts1.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                             'adaptive': True, 'maxfev': 10000}
                )
                pts2 = normalize_rms(res2.x.reshape(n, d))
                try_update(pts2)
            except Exception:
                pts2 = pts1

            # Step 2c: More NM iterations if time allows
            if time.time() < deadline - 70:
                try:
                    res3 = minimize(
                        neg_score,
                        pts2.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17,
                                 'adaptive': True, 'maxfev': 20000}
                    )
                    pts3 = normalize_rms(res3.x.reshape(n, d))
                    try_update(pts3)
                except Exception:
                    pass

    # =================================================================
    # PHASE 3: Simulated Annealing (Structural Change)
    # Knowledge: SA explores non-convex landscape more robustly.
    # =================================================================
    if time.time() < deadline - 150:
        rng_sa = np.random.RandomState(42)
        # Start from best point
        current_pts = best_points.copy() if best_points is not None else candidates[0]
        current_pts = normalize_rms(current_pts)
        current_score = true_score(current_pts)
        
        T = 0.05  # Initial temperature
        T_min = 1e-6
        alpha = 0.999  # Cooling rate
        max_iters = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 150:
                break
            if T < T_min:
                break
                
            # Perturb
            noise_scale = 0.1 * T
            perturbed = current_pts + rng_sa.randn(n, d) * noise_scale
            perturbed = normalize_rms(perturbed)
            
            new_score = true_score(perturbed)
            
            # Acceptance criterion
            if new_score > current_score:
                current_pts = perturbed
                current_score = new_score
                try_update(current_pts)
            else:
                delta = current_score - new_score
                if delta < 0:
                    accept = True
                else:
                    accept_prob = np.exp(-delta / T)
                    accept = rng_sa.rand() < accept_prob
                
                if accept:
                    current_pts = perturbed
                    current_score = new_score
            
            T *= alpha

    # =================================================================
    # PHASE 4: Differential evolution for global search
    # =================================================================
    if time.time() < deadline - 100:
        try:
            bounds = [(-2, 2)] * (n * d)
            res_de = differential_evolution(
                neg_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=30,
                tol=1e-12,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='sobol',
                update_interval=1
            )
            pts_de = normalize_rms(res_de.x.reshape(n, d))
            try_update(pts_de)
        except Exception:
            pass

    # =================================================================
    # PHASE 5: Fine-tuning with random restarts + local polish
    # =================================================================
    if best_points is not None and time.time() < deadline - 40:
        rng = np.random.RandomState(123)
        for restart in range(30):
            if time.time() > deadline - 20:
                break
            noise_scale = 0.1 * (1.0 - restart / 40.0)
            perturbed = best_points + rng.randn(n, d) * noise_scale
            perturbed = normalize_rms(perturbed)

            try:
                res = minimize(
                    neg_score,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True, 'maxfev': 5000}
                )
                pts_r = normalize_rms(res.x.reshape(n, d))
                try_update(pts_r)
            except Exception:
                continue

    # =================================================================
    # PHASE 6: Final ultra-fine polish
    # =================================================================
    if best_points is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True, 'maxfev': 50000}
            )
            pts_final = normalize_rms(res_final.x.reshape(n, d))
            try_update(pts_final)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    if not np.all(np.isfinite(best_points)):
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    return best_points
