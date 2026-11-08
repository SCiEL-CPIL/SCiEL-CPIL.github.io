import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def compute_active_triangles(points, threshold=1e-6):
        """Return indices of triangles with area within threshold of min."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return np.array([], dtype=np.int32)
        if area_hull < 1e-9:
            return np.array([], dtype=np.int32)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        active = np.where(areas <= min_area + threshold * min_area)[0]
        return active

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _epigraph_objective(points_flat, t, beta=100.0):
        """
        Epigraph formulation: minimize -t subject to areas >= t for all triangles.
        Uses soft constraints: sum(exp(-beta*(areas - t))) as penalty.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft penalty for violations: areas < t
        violations = np.maximum(0, t - areas)
        # Smooth penalty using log-sum-exp
        if np.max(violations) > 1e-15:
            neg_beta_viol = -beta * violations
            max_val = np.max(neg_beta_viol)
            if max_val < -1e6:
                penalty = 1e6
            else:
                log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_viol - max_val)))
                penalty = log_sum_exp / beta
        else:
            penalty = 0.0
        
        return -t + penalty

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min: -log(sum(exp(-beta * areas))) / beta
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _active_set_objective(points_flat, active_idx, beta=50.0):
        """
        Objective focusing only on active (binding) triangles.
        This reduces the effective dimensionality of the optimization.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j[active_idx]] - x[tri_i[active_idx]]
        v1y = y[tri_j[active_idx]] - y[tri_i[active_idx]]
        v2x = x[tri_k[active_idx]] - x[tri_i[active_idx]]
        v2y = y[tri_k[active_idx]] - y[tri_i[active_idx]]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        if len(areas) == 0:
            return _soft_objective(points_flat, beta)
        
        # Soft min over active triangles only
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _exact_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_to_points(coeffs, max_harmonic=6):
        """Convert Fourier coefficients to 13 point positions.
        coeffs = [a1, b1, a2, b2, ..., a_m, b_m] (2*max_harmonic params)
        r_k = 1 + sum_{m=1}^{max_harmonic} (a_m * cos(m*theta_k) + b_m * sin(m*theta_k))
        """
        n_harm = min(len(coeffs) // 2, max_harmonic)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for m in range(1, n_harm + 1):
            a_m = coeffs[2 * (m - 1)]
            b_m = coeffs[2 * (m - 1) + 1]
            radii += a_m * np.cos(m * angles) + b_m * np.sin(m * angles)
        # Ensure positive radii (convexity)
        radii = np.maximum(radii, 0.1)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_to_points_full(coeffs):
        """Backward compatible wrapper."""
        return _fourier_to_points(coeffs, max_harmonic=len(coeffs)//2)

    def _fourier_soft_objective(coeffs, beta=100.0):
        """Soft objective in Fourier space."""
        pts = _fourier_to_points_full(coeffs)
        return _soft_objective(pts.flatten(), beta)

    def _fourier_exact_objective(coeffs):
        """Exact objective in Fourier space."""
        pts = _fourier_to_points_full(coeffs)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _fourier_active_objective(coeffs, active_idx, beta=50.0):
        """Active-set soft objective in Fourier space."""
        pts = _fourier_to_points_full(coeffs)
        return _active_set_objective(pts.flatten(), active_idx, beta)

    def _newton_refine_fourier(coeffs_init, max_iter=50, tol=1e-10):
        """
        Newton-Raphson refinement in Fourier space.
        Uses numerical Hessian for the soft objective.
        Returns refined coefficients and score.
        """
        coeffs = coeffs_init.copy()
        beta = 200.0
        
        # Compute initial objective and gradient
        def _obj(c):
            return _fourier_soft_objective(c, beta)
        
        def _grad(c, eps=1e-7):
            g = np.zeros_like(c)
            for i in range(len(c)):
                c_plus = c.copy()
                c_minus = c.copy()
                c_plus[i] += eps
                c_minus[i] -= eps
                g[i] = (_obj(c_plus) - _obj(c_minus)) / (2 * eps)
            return g
        
        def _hessian(c, eps=1e-6):
            """Numerical Hessian."""
            n = len(c)
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    c_pp = c.copy()
                    c_pm = c.copy()
                    c_mp = c.copy()
                    c_mm = c.copy()
                    c_pp[i] += eps
                    c_pp[j] += eps
                    c_pm[i] += eps
                    c_pm[j] -= eps
                    c_mp[i] -= eps
                    c_mp[j] += eps
                    c_mm[i] -= eps
                    c_mm[j] -= eps
                    h = (_obj(c_pp) - _obj(c_pm) - _obj(c_mp) + _obj(c_mm)) / (4 * eps * eps)
                    H[i, j] = h
                    H[j, i] = h
            return H
        
        # Newton iterations
        for it in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break
            
            obj_val = _obj(coeffs)
            grad = _grad(coeffs)
            
            # Check convergence
            if np.linalg.norm(grad) < tol:
                break
            
            # Compute Hessian
            H = _hessian(coeffs)
            
            # Solve H * delta = -grad
            try:
                # Add small regularization for stability
                H_reg = H + 1e-8 * np.eye(len(coeffs))
                delta = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                delta = -grad * 0.01
            
            # Line search to ensure decrease
            alpha = 1.0
            new_obj = obj_val
            for ls in range(20):
                candidate = coeffs + alpha * delta
                new_obj = _obj(candidate)
                if new_obj < obj_val - 1e-12 * np.abs(obj_val):
                    break
                alpha *= 0.5
            
            if new_obj >= obj_val:
                # No improvement, try smaller step
                candidate = coeffs - 0.01 * grad
                new_obj = _obj(candidate)
                if new_obj < obj_val:
                    coeffs = candidate
                else:
                    break
            else:
                coeffs = candidate
            
            obj_val = new_obj
        
        # Compute final score
        pts = _fourier_to_points_full(coeffs)
        score = compute_score(pts)
        return coeffs, score

    def _active_set_polish(points, max_rounds=10, tol=1e-12):
        """
        Active-set polish: identify binding triangles, then optimize
        only to maximize the minimum area among those triangles.
        This is a local refinement that exploits the fact that at the optimum,
        typically only a few triangles are binding.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        
        for round_num in range(max_rounds):
            if time.time() > start_time + time_limit - 3:
                break
            
            # Identify active triangles
            active = compute_active_triangles(pts, threshold=1e-4)
            if len(active) == 0:
                break
            
            # Create a focused objective that only considers active triangles
            def _focused_obj(x_flat):
                return _active_set_objective(x_flat, active, beta=200.0)
            
            # Optimize with L-BFGS-B on the focused objective
            try:
                res = opt.minimize(
                    _focused_obj,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                candidate = res.x.reshape(-1, 2)
                cand_score = compute_score(candidate)
                
                if cand_score > best_score + tol:
                    pts = candidate
                    best_score = cand_score
                else:
                    # No improvement; tighten the active set threshold
                    active = compute_active_triangles(pts, threshold=1e-5)
                    if len(active) <= 3:
                        break
            except Exception:
                break
        
        return pts, best_score

    def _coordinate_descent(points, max_iter=500, tol=1e-10):
        """
        Coordinate descent to maximize min triangle area / hull area.
        Uses adaptive step sizes and more thorough local search.
        Optimized to reduce ConvexHull calls.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        
        # Adaptive step sizes - start smaller for refinement
        base_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
                
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score
                
                # Try perturbations with adaptive step sizes
                for step in base_steps:
                    # 8 directions
                    directions = [
                        (step, 0), (-step, 0), (0, step), (0, -step),
                        (step*0.707, step*0.707), (-step*0.707, step*0.707),
                        (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                    ]
                    for dx, dy in directions:
                        new_pt = pts.copy()
                        new_pt[i] = pts[i] + np.array([dx, dy])
                        score = compute_score(new_pt)
                        if score > best_local_score + tol:
                            best_local_score = score
                            best_local = new_pt[i].copy()
                            improved = True
                
                if improved:
                    pts[i] = best_local
                    best_score = best_local_score
            
            if not improved:
                break
        
        return pts, best_score

    def _gradient_ascent_step(points, step_size=0.01, n_iter=20):
        """
        Gradient ascent step using numerical gradient of the soft-min objective.
        Moves points in the direction that increases the minimum triangle area.
        """
        pts = points.copy()
        beta = 100.0
        
        def _obj(x_flat):
            return _soft_objective(x_flat, beta)
        
        def _grad(x_flat, eps=1e-6):
            g = np.zeros_like(x_flat)
            for i in range(len(x_flat)):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                g[i] = (_obj(x_plus) - _obj(x_minus)) / (2 * eps)
            return g
        
        for _ in range(n_iter):
            if time.time() > start_time + time_limit - 3:
                break
            
            x_flat = pts.flatten()
            grad = _grad(x_flat)
            
            # Gradient ascent (we want to maximize, so move in +grad direction)
            # But _obj returns -soft_min, so we want to minimize _obj, i.e., move in -grad direction
            pts = (x_flat - step_size * grad).reshape(-1, 2)
        
        return pts


    # Generate initial configurations
    initializations = []
    
    # 1. Regular 13-gon (all on boundary)
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 on boundary + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. Various split configurations: k on outer ring, (13-k) on inner ring
    for k_outer in range(5, 13):
        k_inner = N - k_outer
        outer = _regular_polygon(k_outer, 1.0)
        for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if k_inner == 1:
                inner = np.array([[0.0, 0.0]])
            else:
                inner = _regular_polygon(k_inner, r_inner)
            config = np.vstack([outer, inner])
            initializations.append(config)
    
    # 4. Concentric polygons with different symmetries
    for r1 in [0.3, 0.4, 0.5]:
        for r2 in [0.6, 0.7, 0.8]:
            center = np.array([[0.0, 0.0]])
            inner_hex = _regular_polygon(6, r1)
            outer_hex = _regular_polygon(6, r2)
            config = np.vstack([center, inner_hex, outer_hex])
            initializations.append(config)
    
    # 5. 13 = 1 + 3 + 9 (center + triangle + nonagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        triangle = _regular_polygon(3, r1)
        nonagon = _regular_polygon(9, 1.0)
        config = np.vstack([center, triangle, nonagon])
        initializations.append(config)
    
    # 6. 13 = 1 + 4 + 8 (center + square + octagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        square = _regular_polygon(4, r1)
        octagon = _regular_polygon(8, 1.0)
        config = np.vstack([center, square, octagon])
        initializations.append(config)
    
    # 7. 13 = 1 + 5 + 7 (center + pentagon + heptagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        pentagon = _regular_polygon(5, r1)
        heptagon = _regular_polygon(7, 1.0)
        config = np.vstack([center, pentagon, heptagon])
        initializations.append(config)
    
    # 8. 13 = 1 + 2 + 10 (center + dipole + decagon)
    for r1 in [0.3, 0.4, 0.5]:
        center = np.array([[0.0, 0.0]])
        dipole = np.array([[-r1, 0.0], [r1, 0.0]])
        decagon = _regular_polygon(10, 1.0)
        config = np.vstack([center, dipole, decagon])
        initializations.append(config)
    
    # Add some random perturbations to break symmetry
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 5):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        initializations.append(perturbed)
    
    # Fourier-based initializations: start from regular polygon and add small deformations
    # Use up to 6 harmonics (12 parameters) for richer representation
    max_harm = 6
    fourier_bounds = [(-0.12, 0.12)] * (2 * max_harm)
    rng_fourier = np.random.RandomState(123)
    for _ in range(15):
        coeffs = rng_fourier.uniform(-0.08, 0.08, size=2 * max_harm)
        pts = _fourier_to_points(coeffs, max_harm)
        initializations.append(pts)
    
    # Also try pure harmonic deformations
    for m in range(1, max_harm + 1):
        for amp in [0.03, 0.06, 0.1, 0.12]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                coeffs = np.zeros(2 * max_harm)
                coeffs[2*(m-1)] = amp * np.cos(phase)
                coeffs[2*(m-1)+1] = amp * np.sin(phase)
                pts = _fourier_to_points(coeffs, max_harm)
                initializations.append(pts)
    
    # Deduplicate and limit initializations
    best_score = -float('inf')
    best_points = None
    seen_scores = set()
    
    # Differential evolution on Fourier parameterization (reduced search space)
    # This helps escape local minima that full 26-D search might get stuck in
    if time.time() < start_time + time_limit - 70:
        try:
            # Use more harmonics but with tighter bounds for higher-order terms
            de_bounds = []
            for m in range(1, max_harm + 1):
                bound = 0.12 / np.sqrt(m)  # Decay with harmonic order
                de_bounds.append((-bound, bound))
                de_bounds.append((-bound, bound))
            
            de_result = opt.differential_evolution(
                lambda c: _fourier_soft_objective(c, beta=100.0),
                bounds=de_bounds,
                seed=42,
                maxiter=100,
                popsize=25,
                tol=1e-9,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,  # We'll do our own polish
                init='latinhypercube'
            )
            if de_result.fun < 1e5:
                best_coeffs = de_result.x
                de_points = _fourier_to_points(best_coeffs, max_harm)
                de_score = compute_score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                # Add DE result to initializations for further refinement
                initializations.append(de_points)
                
                # Newton refinement on DE result
                if time.time() < start_time + time_limit - 35:
                    try:
                        new_coeffs, new_score = _newton_refine_fourier(best_coeffs)
                        if new_score > best_score:
                            best_score = new_score
                            best_points = _fourier_to_points_full(new_coeffs).copy()
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Optimize each initialization
    for idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 20:
            break
            
        current = init.copy()
        current_score = compute_score(current)
        
        # Quick check: skip if this is a known bad config
        score_key = round(current_score, 6)
        if score_key in seen_scores:
            continue
        seen_scores.add(score_key)
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 10:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 1b: Fourier-space optimization with Newton refinement
        if time.time() < start_time + time_limit - 15:
            try:
                # Estimate Fourier coefficients from current points
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                radii = np.sqrt(current[:, 0]**2 + current[:, 1]**2)
                # Fit Fourier series to radii using up to 6 harmonics
                A = np.zeros((N, 2 * max_harm))
                for m in range(1, max_harm + 1):
                    A[:, 2*(m-1)] = np.cos(m * angles)
                    A[:, 2*(m-1)+1] = np.sin(m * angles)
                # Least squares fit
                coeffs_fit, _, _, _ = np.linalg.lstsq(A, radii - 1.0, rcond=None)
                
                # Optimize in Fourier space with L-BFGS-B
                for beta_f in [50.0, 100.0, 200.0, 400.0]:
                    if time.time() > start_time + time_limit - 12:
                        break
                    try:
                        res_f = opt.minimize(
                            lambda c, b=beta_f: _fourier_soft_objective(c, b),
                            coeffs_fit,
                            method='L-BFGS-B',
                            bounds=fourier_bounds,
                            options={'maxiter': 60, 'ftol': 1e-13, 'gtol': 1e-11}
                        )
                        if res_f.fun < 1e5:
                            f_pts = _fourier_to_points(res_f.x, max_harm)
                            f_score = compute_score(f_pts)
                            if f_score > compute_score(current):
                                current = f_pts
                                coeffs_fit = res_f.x
                    except Exception:
                        pass
                
                # Newton refinement in Fourier space
                if time.time() < start_time + time_limit - 8:
                    try:
                        new_coeffs, new_score = _newton_refine_fourier(coeffs_fit)
                        if new_score > compute_score(current):
                            current = _fourier_to_points_full(new_coeffs)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Step 2: Coordinate descent refinement
        if time.time() > start_time + time_limit - 8:
            break
        try:
            current, score_cd = _coordinate_descent(current, max_iter=150)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass
        
        # Step 3: Active-set polish (new)
        if time.time() < start_time + time_limit - 6:
            try:
                current, score_as = _active_set_polish(current, max_rounds=8)
                if score_as > best_score:
                    best_score = score_as
                    best_points = current.copy()
            except Exception:
                pass
        
        # Step 4: Nelder-Mead with exact objective for final polish
        if time.time() > start_time + time_limit - 4:
            break
        try:
            res = opt.minimize(
                _exact_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 5: Another active-set polish after Nelder-Mead
        if time.time() < start_time + time_limit - 2:
            try:
                candidate, score_as2 = _active_set_polish(candidate, max_rounds=5)
                if score_as2 > best_score:
                    best_score = score_as2
                    best_points = candidate.copy()
            except Exception:
                pass
        
        # Step 6: High-precision final polish with very small steps
        if time.time() < start_time + time_limit - 0.5:
            try:
                # Ultra-fine coordinate descent
                fine_steps = [0.0001, 0.00005, 0.00002, 0.00001]
                fine_pts = best_points.copy()
                fine_score = compute_score(fine_pts)
                for step in fine_steps:
                    for i in range(N):
                        directions = [
                            (step, 0), (-step, 0), (0, step), (0, -step),
                            (step*0.707, step*0.707), (-step*0.707, step*0.707),
                            (step*0.707, -step*0.707), (-step*0.707, -step*0.707)
                        ]
                        for dx, dy in directions:
                            new_pt = fine_pts.copy()
                            new_pt[i] = fine_pts[i] + np.array([dx, dy])
                            score = compute_score(new_pt)
                            if score > fine_score + 1e-12:
                                fine_score = score
                                fine_pts[i] = new_pt[i]
                if fine_score > best_score:
                    best_score = fine_score
                    best_points = fine_pts.copy()
                    
                    # Final Newton polish in Fourier space
                    if time.time() < start_time + time_limit - 0.1:
                        try:
                            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                            radii = np.sqrt(best_points[:, 0]**2 + best_points[:, 1]**2)
                            A = np.zeros((N, 2 * max_harm))
                            for m in range(1, max_harm + 1):
                                A[:, 2*(m-1)] = np.cos(m * angles)
                                A[:, 2*(m-1)+1] = np.sin(m * angles)
                            coeffs_final, _, _, _ = np.linalg.lstsq(A, radii - 1.0, rcond=None)
                            new_coeffs, new_score = _newton_refine_fourier(coeffs_final, max_iter=20)
                            if new_score > best_score:
                                best_score = new_score
                                best_points = _fourier_to_points_full(new_coeffs).copy()
                        except Exception:
                            pass
            except Exception:
                pass
    
    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]
    
    return best_points
