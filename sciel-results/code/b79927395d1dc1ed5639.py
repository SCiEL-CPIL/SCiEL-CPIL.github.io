import numpy as np
from scipy.optimize import minimize
import time

N = 300
_DEADLINE = time.time() + 145.0  # Safety margin before 150s kill

def _hex_lattice_init(n_points, rotation=0.0, seed=42):
    """Generate a hexagonal lattice of points inside the unit circle."""
    rng = np.random.RandomState(seed)
    # Estimate spacing for ~n points in unit disk
    # Area of hex lattice per point ~ s^2 * sqrt(3)/2
    # Area of unit disk = pi
    # n * s^2 * sqrt(3)/2 ~ pi  =>  s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    # Add margin to ensure we have enough points
    s *= 0.95
    k = int(1.0 / s) + 3
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # Apply rotation
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    # Add small random noise to break symmetry
    points += rng.normal(0, s * 0.01, size=points.shape)
    
    return points

def _radial_ring_init(n_points, seed=42):
    """Generate points on concentric rings."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    n_rings = int(1.0 / s) + 1
    
    points = []
    for r_idx in range(n_rings):
        r = (r_idx + 0.5) * s
        if r >= 1.0:
            break
        n_on_ring = max(1, int(2 * np.pi * r / s))
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        remaining = n_points - len(points)
        r = np.sqrt(rng.random(remaining))
        a = rng.random(remaining) * 2 * np.pi
        extra = np.column_stack([r * np.cos(a), r * np.sin(a)])
        points = np.vstack([points, extra])
    elif len(points) > n_points:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:n_points]]
    
    points += rng.normal(0, s * 0.01, size=points.shape)
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    if len(points) < 2:
        return 0.0
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient efficiently.
    Uses vectorized O(N) gradient accumulation.
    """
    C = x.reshape(N, 2)
    n = N
    
    # Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists_sq = np.sum(D**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    x_pairs_flat = dists[i_idx, j_idx] / 2.0
    
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    # Clip to avoid log(0) or exp overflow issues
    x_all = np.maximum(x_all, 1e-12)
    
    # Soft min: min(x) - (1/beta) * log(sum(exp(-beta * (x - min(x)))))
    # Using log-sum-exp trick for stability
    # val = min_x - (1/beta) * log_sum_exp(-beta * (x_all - min_x))
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    # Pairwise gradient contribution
    # grad_i += sum_j P_ij * 0.5 * (C_i - C_j) / dist_ij
    P_pairs = P[:len(i_idx)]
    
    L_flat = dists[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    # Vectorized accumulation
    # For each pair (i, j), weight w = P_ij * 0.5
    # grad[i] += w * (C[i] - C[j]) / dist
    # grad[j] -= w * (C[i] - C[j]) / dist
    
    w = P_pairs * 0.5
    w_over_d = w / L_flat_safe
    
    # Accumulate into grad[i] and grad[j]
    # grad[i] += w_over_d * (C[i] - C[j])
    # This can be written as:
    # grad[i] += w_over_d * C[i] - w_over_d * C[j]
    
    # Use np.add.at for accumulation
    # Term 1: + w_over_d * C[i]
    np.add.at(grad, i_idx, w_over_d[:, np.newaxis] * C[i_idx])
    # Term 2: - w_over_d * C[j]
    np.add.at(grad, i_idx, -w_over_d[:, np.newaxis] * C[j_idx])
    
    # Term 3: - w_over_d * C[i] (for grad[j])
    np.add.at(grad, j_idx, -w_over_d[:, np.newaxis] * C[i_idx])
    # Term 4: + w_over_d * C[j] (for grad[j])
    np.add.at(grad, j_idx, w_over_d[:, np.newaxis] * C[j_idx])
    
    # Wall gradient contribution
    # grad_i += P_wall_i * (-1) * (C_i / |C_i|) ? 
    # Objective is min(1 - |C|). Grad of (1-|C|) is -C/|C|.
    # We are minimizing -softmin, so grad is -grad(softmin).
    # softmin grad wrt x_wall is P_wall * 1.
    # So -softmin grad is -P_wall * 1 for the scalar x_wall.
    # x_wall = 1 - |C|. d(x_wall)/d(C) = -C/|C|.
    # Chain rule: d(-softmin)/d(C) = -P_wall * (-C/|C|) = P_wall * C/|C|.
    
    P_walls = P[len(i_idx):]
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _optimize_lbfgs(points, beta, max_iter=300, time_limit=10.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit or time.time() > _DEADLINE:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def _subgradient_polish(points, max_iters=200, tol=1e-10, time_limit=15.0):
    """
    Local optimization using exact min with subgradient descent and adaptive line search.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit or time.time() > _DEADLINE:
            break
        
        # Compute all constraint values
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        # Pair constraints: dists/2
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        bound_vals = boundary_dists
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        min_val = np.min(all_vals)
        
        if min_val < tol:
            break
        
        # Find active constraints (within tolerance of minimum)
        active_mask = np.abs(all_vals - min_val) < 1e-6
        active_pairs = np.where(active_mask[:len(ii)])[0]
        active_bounds = np.where(active_mask[len(ii):])[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build gradient for the minimum constraint
        # Use subgradient: average of gradients of active constraints
        grad = np.zeros_like(pts)
        
        # Pairwise active constraints
        if len(active_pairs) > 0:
            for idx in active_pairs:
                i, j = ii[idx], jj[idx]
                d = dists[i, j]
                if d < 1e-10:
                    continue
                direction = (pts[i] - pts[j]) / d
                # Grad of dist/2 is 0.5 * direction for i, -0.5 * direction for j
                grad[i] += direction / (2.0 * len(active_pairs))
                grad[j] -= direction / (2.0 * len(active_pairs))
        
        # Boundary active constraints
        if len(active_bounds) > 0:
            for idx in active_bounds:
                i = active_bounds[idx]
                r = radii[i]
                if r < 1e-10:
                    continue
                # Grad of (1 - r) is -pts[i]/r
                direction = -pts[i] / r
                grad[i] += direction / (len(active_bounds))
        
        # Normalize gradient
        grad_norm = np.sqrt(np.sum(grad**2))
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Adaptive line search: start with larger steps, shrink if no improvement
        best_step = 0.0
        best_obj = min_val
        
        # Try a range of step sizes, starting larger
        base_step = 0.01
        for mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            step = base_step * mult
            new_pts = pts + step * grad
            new_score = _compute_true_score(new_pts)
            if new_score > best_obj + 1e-12:
                best_obj = new_score
                best_step = step
                break
        
        if best_step == 0.0:
            break
        
        pts = pts + best_step * grad
        
        # Project back inside unit circle if needed
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.9999
        if np.any(mask):
            pts[mask] = pts[mask] * (0.9999 / norms[mask])[:, np.newaxis]
    
    return pts

def _epigraph_polish(points, max_iters=100, time_limit=15.0):
    """
    Epigraph formulation: maximize t subject to all pairwise and wall constraints >= t.
    Uses scipy minimize with SLSQP on the epigraph problem.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    ii, jj = np.triu_indices(n, k=1)
    
    def epigraph_obj(x):
        if time.time() - start_time > time_limit or time.time() > _DEADLINE:
            raise TimeoutError()
        C = x[:n*2].reshape(n, 2)
        t = x[n*2]
        # Return negative t to maximize
        return -t
    
    def epigraph_grad(x):
        C = x[:n*2].reshape(n, 2)
        t = x[n*2]
        grad = np.zeros_like(x)
        # Gradient of -t w.r.t. t is -1
        grad[n*2] = -1.0
        return grad
    
    def make_constraint(i, j):
        def constraint(x):
            C = x[:n*2].reshape(n, 2)
            t = x[n*2]
            d = np.linalg.norm(C[i] - C[j]) / 2.0 - t
            return d
        return constraint
    
    def make_wall_constraint(i):
        def constraint(x):
            C = x[:n*2].reshape(n, 2)
            t = x[n*2]
            r = np.linalg.norm(C[i])
            return 1.0 - r - t
        return constraint
    
    # Build constraints
    constraints = []
    for idx in range(len(ii)):
        i, j = ii[idx], jj[idx]
        constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(i)})
    
    # Bounds: points in unit disk, t >= 0
    bounds = []
    for _ in range(n):
        bounds.append((-1.0, 1.0))
        bounds.append((-1.0, 1.0))
    bounds.append((0.0, 1.0))
    
    x0 = np.concatenate([pts.flatten(), [0.05]])
    
    try:
        res = minimize(epigraph_obj, x0, method='SLSQP', jac=epigraph_grad,
                      constraints=constraints, bounds=bounds,
                      options={'maxiter': max_iters, 'ftol': 1e-12})
        if res.success or (hasattr(res, 'fun') and res.fun < -0.05):
            new_pts = res.x[:n*2].reshape(n, 2)
            # Verify validity
            new_score = _compute_true_score(new_pts)
            old_score = _compute_true_score(pts)
            if new_score > old_score:
                return new_pts
    except Exception:
        pass
    
    return pts

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin before 150s kill
    
    # Multi-start with different initializations
    inits = []
    
    # Hex lattice with different rotations
    for rot in [0.0, np.pi / 6, np.pi / 3, np.pi / 4]:
        inits.append(_hex_lattice_init(N, rotation=rot))
    
    # Radial ring init
    inits.append(_radial_ring_init(N))
    
    best_points = None
    best_score = -np.inf
    
    # First pass: optimize each init with progressive beta
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        current = init.copy()
        beta = 50.0
        
        # Progressive beta increase
        for _ in range(3):
            if time.time() - start_time > time_limit:
                break
            current = _optimize_lbfgs(current, beta, max_iter=200, time_limit=8.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
            beta *= 3.0
        
        # Final polish with high beta
        if time.time() - start_time > time_limit:
            break
        current = _optimize_lbfgs(current, 500.0, max_iter=300, time_limit=10.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Second pass: refine the best solution with subgradient polish
    if best_points is not None and time.time() - start_time < time_limit:
        current = best_points.copy()
        for beta in [1000.0, 2000.0]:
            if time.time() - start_time > time_limit:
                break
            current = _optimize_lbfgs(current, beta, max_iter=500, time_limit=12.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        
        # Subgradient polish on the best
        if time.time() - start_time < time_limit - 5:
            current = _subgradient_polish(best_points, max_iters=300, tol=1e-10, time_limit=15.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Final high-precision polish with L-BFGS at very high beta
    if best_points is not None and time.time() - start_time < time_limit - 10:
        current = best_points.copy()
        current = _optimize_lbfgs(current, 5000.0, max_iter=500, time_limit=15.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
        
        # Subgradient polish again
        if time.time() - start_time < time_limit - 5:
            current = _subgradient_polish(best_points, max_iters=200, tol=1e-11, time_limit=10.0)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Epigraph polish for final precision
    if best_points is not None and time.time() - start_time < time_limit - 8:
        current = _epigraph_polish(best_points, max_iters=50, time_limit=12.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Final subgradient polish with very tight tolerance
    if best_points is not None and time.time() - start_time < time_limit - 1:
        current = _subgradient_polish(best_points, max_iters=100, tol=1e-12, time_limit=5.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Ensure all points are inside the unit circle
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] * (0.9999 / norms[mask])[:, np.newaxis]
    
    return best_points
