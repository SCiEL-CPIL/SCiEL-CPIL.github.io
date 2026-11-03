"""Packing 50 equal circles in a unit circle using differentiable soft-min with ring init."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _score_vec(points):
    """Vectorized score computation."""
    points = points.reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _ring_initialization(n, n_rings=5):
    """Generate ring-structured initial configuration with optimized ring radii."""
    best_points = None
    best_r = 0.0
    
    # Try multiple ring configurations
    for trial in range(20):
        # Random ring structure: assign circles to rings
        # First ring at center, subsequent rings have increasing radius
        ring_counts = [1]  # center
        remaining = n - 1
        for ring in range(1, n_rings + 1):
            if remaining <= 0:
                break
            # Distribute remaining circles among remaining rings
            rings_left = n_rings - ring
            if rings_left <= 0:
                ring_counts.append(remaining)
                remaining = 0
            else:
                count = max(1, remaining // (rings_left + 1))
                ring_counts.append(count)
                remaining -= count
        
        # Adjust to ensure total is n
        while sum(ring_counts) < n:
            ring_counts[-1] += 1
        while sum(ring_counts) > n:
            if len(ring_counts) > 1:
                ring_counts[-1] -= 1
            else:
                break
        
        # Generate points on rings
        points = []
        for ring_idx, count in enumerate(ring_counts):
            if ring_idx == 0:
                points.append([0.0, 0.0])
            else:
                # Radius of this ring
                r_ring = 0.15 + 0.85 * (ring_idx / max(n_rings, 1))
                if trial % 3 == 0:
                    r_ring *= 0.9
                elif trial % 3 == 1:
                    r_ring *= 1.0
                else:
                    r_ring *= 1.1
                
                angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
                for angle in angles:
                    x = r_ring * np.cos(angle)
                    y = r_ring * np.sin(angle)
                    points.append([x, y])
        
        points = np.array(points[:n])
        if len(points) < n:
            continue
        
        # Optimize ring radii and angular offsets using gradient descent
        def soft_min_obj(params, beta=100.0):
            """Differentiable soft-minimum objective."""
            # params: [ring_radii..., angular_offsets...]
            n_rings_actual = len(ring_counts)
            ring_radii = params[:n_rings_actual]
            ang_offsets = params[n_rings_actual:n_rings_actual + n_rings_actual]
            
            pts = []
            for ring_idx, count in enumerate(ring_counts):
                r_ring = ring_radii[ring_idx]
                if ring_idx == 0:
                    pts.append([0.0, 0.0])
                else:
                    offset = ang_offsets[ring_idx]
                    angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
                    for angle in angles:
                        x = r_ring * np.cos(angle)
                        y = r_ring * np.sin(angle)
                        pts.append([x, y])
            
            pts = np.array(pts[:n])
            if len(pts) < n:
                return 1e10
            
            # Compute soft-min of all constraints
            boundary = 1.0 - np.linalg.norm(pts, axis=1)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            iu = np.triu_indices(n, 1)
            pair_dists = dists[iu] / 2.0
            
            all_vals = np.concatenate([boundary, pair_dists])
            # Soft-min via log-sum-exp
            m = np.min(all_vals)
            soft_min = m + np.log(np.sum(np.exp(-beta * (all_vals - m)))) / beta
            return -soft_min
        
        # Initial parameters
        n_rings_actual = len(ring_counts)
        init_params = []
        for ring_idx, count in enumerate(ring_counts):
            if ring_idx == 0:
                init_params.append(0.0)
            else:
                init_params.append(0.15 + 0.85 * (ring_idx / max(n_rings_actual, 1)))
        init_params.extend([0.0] * n_rings_actual)
        
        try:
            res = minimize(
                soft_min_obj,
                np.array(init_params),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            if res.success and np.all(np.isfinite(res.x)):
                # Reconstruct points from optimized parameters
                ring_radii = res.x[:n_rings_actual]
                ang_offsets = res.x[n_rings_actual:n_rings_actual + n_rings_actual]
                
                pts = []
                for ring_idx, count in enumerate(ring_counts):
                    r_ring = ring_radii[ring_idx]
                    if ring_idx == 0:
                        pts.append([0.0, 0.0])
                    else:
                        offset = ang_offsets[ring_idx]
                        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
                        for angle in angles:
                            x = r_ring * np.cos(angle)
                            y = r_ring * np.sin(angle)
                            pts.append([x, y])
                
                pts = np.array(pts[:n])
                if len(pts) == n:
                    r = _score_vec(pts)
                    if r > best_r:
                        best_r = r
                        best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: simple hexagonal
        best_points = np.zeros((n, 2))
        best_r = 0.0
        for spacing in np.linspace(0.05, 0.25, 30):
            s = spacing
            pts = []
            max_coord = int(1.0 / s) + 2
            for j in range(-max_coord, max_coord + 1):
                for i in range(-max_coord, max_coord + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3) / 2
                    if np.hypot(x, y) <= 1.0 - s / 2:
                        pts.append([x, y])
            pts = np.array(pts)
            if len(pts) >= n:
                dists = np.linalg.norm(pts, axis=1)
                idx = np.argsort(dists)[:n]
                cand = pts[idx]
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_points = cand.copy()
    
    return best_points


def _softmin_objective(x, pair_idx, beta):
    """Differentiable soft-minimum objective for circle packing."""
    c = x.reshape(N, 2)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    all_vals = np.concatenate([d, b])
    # Stable log-sum-exp
    m = np.min(all_vals)
    soft_min = m + np.log(np.sum(np.exp(-beta * (all_vals - m)))) / beta
    return -soft_min


def _softmin_gradient(x, pair_idx, beta):
    """Gradient of the soft-minimum objective."""
    c = x.reshape(N, 2)
    ci = c[pair_idx[:, 0]]
    cj = c[pair_idx[:, 1]]
    diff = ci - cj
    d = np.sqrt(np.sum(diff ** 2, axis=1))
    d_safe = np.maximum(d, 1e-10)
    b = 1.0 - np.hypot(c[:, 0], c[:, 1])
    b_safe = np.maximum(b, 1e-10)
    
    all_vals = np.concatenate([d / 2, b])
    m = np.min(all_vals)
    weights = np.exp(-beta * (all_vals - m))
    weights /= np.sum(weights)
    
    # Gradient with respect to positions
    grad = np.zeros((N, 2))
    # Pairwise distances
    for k in range(len(pair_idx)):
        i, j = pair_idx[k]
        w = weights[k] * beta
        diff_k = diff[k]
        grad[i] += w * diff_k / (2 * d_safe[k])
        grad[j] -= w * diff_k / (2 * d_safe[k])
    # Boundary
    for i in range(N):
        w = weights[N + i] * beta
        r_i = np.hypot(c[i, 0], c[i, 1])
        r_safe = max(r_i, 1e-10)
        grad[i] -= w * c[i] / r_safe
    
    return grad.flatten()


def _local_optimize(c, time_limit=10.0, beta_schedule=None):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    
    if beta_schedule is None:
        beta_schedule = [100.0, 200.0, 400.0, 800.0]
    
    best_c = c.copy()
    best_r = _score_vec(best_c)
    
    t_start = time.time()
    
    for beta in beta_schedule:
        if time.time() - t_start >= time_limit - 2:
            break
        
        def neg_soft_min(x):
            return _softmin_objective(x, pair_idx, beta)
        
        def callback(x):
            if time.time() - t_start >= time_limit - 1:
                raise Exception("timeout")
        
        try:
            res = minimize(
                neg_soft_min,
                best_c.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15},
                callback=callback
            )
            if res.success and np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r = _score_vec(cand)
                if r > best_r:
                    best_r = r
                    best_c = cand.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _gradient_descent_polish(c, time_limit=10.0, beta=500.0):
    """Direct gradient descent polish for fine-tuning."""
    pair_idx = _pairwise_indices(N)
    best_c = c.copy()
    best_r = _score_vec(best_c)
    
    t_start = time.time()
    lr = 0.01
    momentum = 0.9
    v = np.zeros(2 * N)
    
    while time.time() - t_start < time_limit - 1:
        x = best_c.flatten()
        obj = _softmin_objective(x, pair_idx, beta)
        grad = _softmin_gradient(x, pair_idx, beta)
        
        # Update momentum
        v = momentum * v - lr * grad
        x_new = x + v
        
        # Clamp to reasonable bounds
        x_new = np.clip(x_new, -1.5, 1.5)
        
        cand = x_new.reshape(N, 2)
        r = _score_vec(cand)
        
        if r > best_r:
            best_r = r
            best_c = cand.copy()
            lr = min(lr * 1.1, 0.05)
        else:
            lr *= 0.9
        
        if lr < 1e-6:
            break
    
    return best_c, best_r


def _anneal(points, t_start, deadline, seed=None):
    """Simulated annealing to escape local minima."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    def cost(p):
        p = p.reshape(-1, 2)
        n = len(p)
        r_target = best_score
        overlap_energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(p[i] - p[j])
                if d < 2 * r_target:
                    overlap_energy += (2 * r_target - d) ** 2
        for i in range(n):
            d = 1.0 - np.linalg.norm(p[i])
            if d < r_target:
                overlap_energy += (r_target - d) ** 2
        return overlap_energy
    
    T = 0.05
    T_min = 0.0001
    cooling = 0.9995
    step_size = 0.02
    
    current_cost = cost(pts)
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        n_moves = np.random.randint(1, 4)
        new_pts = pts.copy()
        for _ in range(n_moves):
            idx = np.random.randint(n)
            delta = np.random.randn(2) * step_size * (1 + T)
            new_pts[idx] += delta
        
        new_cost = cost(new_pts)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            pts = new_pts
            current_cost = new_cost
            s = _score_vec(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_score


def _compute_r(c):
    """Compute the max common radius for given centers."""
    return _score_vec(c)


def construct():
    t_start = time.time()
    
    # Start with ring-structured initialization
    points = _ring_initialization(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # Local optimization from ring init
    if time.time() < DEADLINE - 25:
        c_opt, r_opt = _local_optimize(points, time_limit=min(15.0, DEADLINE - (time.time() - t_start)))
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # Gradient descent polish
    if time.time() < DEADLINE - 20:
        c_grad, r_grad = _gradient_descent_polish(best_points, time_limit=min(10.0, DEADLINE - (time.time() - t_start)))
        if r_grad > best_score:
            best_score = r_grad
            best_points = c_grad.copy()
    
    # Simulated annealing to escape local minima
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 30:
        sa_points, sa_score = _anneal(best_points, time.time(), time.time() + min(remaining - 15, 40), seed=42)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Local optimize after SA
        if time.time() < DEADLINE - 15:
            c_opt, r_opt = _local_optimize(best_points, time_limit=min(12.0, DEADLINE - (time.time() - t_start)))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        
        # Another gradient polish
        if time.time() < DEADLINE - 10:
            c_grad, r_grad = _gradient_descent_polish(best_points, time_limit=min(8.0, DEADLINE - (time.time() - t_start)))
            if r_grad > best_score:
                best_score = r_grad
                best_points = c_grad.copy()
    
    # Multi-start: perturb best and try again
    n_restarts = 0
    while time.time() < DEADLINE - 12:
        perturbation_scale = 0.015 / (1.0 + 0.1 * n_restarts)
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        try:
            c_opt, r_opt = _local_optimize(x0.reshape(N, 2), time_limit=min(8.0, DEADLINE - (time.time() - t_start)))
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        except Exception:
            pass
        
        if n_restarts >= 4:
            break
    
    # Final polish with high beta
    if time.time() < DEADLINE - 5:
        c_final, r_final = _local_optimize(best_points, time_limit=min(6.0, DEADLINE - (time.time() - t_start) - 2), beta_schedule=[400.0, 800.0, 1200.0])
        if r_final > best_score:
            best_points = c_final
            best_score = r_final
        
        # Final gradient polish
        if time.time() < DEADLINE - 2:
            c_grad, r_grad = _gradient_descent_polish(best_points, time_limit=min(4.0, DEADLINE - (time.time() - t_start) - 1), beta=1000.0)
            if r_grad > best_score:
                best_points = c_grad
                best_score = r_grad
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    return c_final
