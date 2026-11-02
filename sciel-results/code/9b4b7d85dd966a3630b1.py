import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0, seed=42):
    """Generate a hexagonal lattice of points inside the unit circle."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
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
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    """
    C = x.reshape(N, 2)
    
    # Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    L_safe = np.where(L > 1e-12, L, 1.0)
    x_pairs = L_safe / 2.0
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = x_pairs[i_idx, j_idx]
    
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _optimize_lbfgs(points, beta, max_iter=300, time_limit=10.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-12})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def _subgradient_polish(points, max_iters=200, tol=1e-10, time_limit=15.0):
    """
    Local optimization using exact min with subgradient descent and adaptive line search.
    This handles the nonsmooth minimum more directly than soft-min gradient descent.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
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
                grad[i] += direction / (2.0 * len(active_pairs))
                grad[j] -= direction / (2.0 * len(active_pairs))
        
        # Boundary active constraints
        if len(active_bounds) > 0:
            for idx in active_bounds:
                i = active_bounds[idx]
                r = radii[i]
                if r < 1e-10:
                    continue
                direction = -pts[i] / r
                grad[i] += direction / (2.0 * len(active_bounds))
        
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

def _per_point_refine(points, max_points=50, time_limit=10.0):
    """
    Refine the most constrained points using Nelder-Mead on a 2D subproblem.
    Only applies to points that are likely on the minimum constraint.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    # Compute current score and identify constrained points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    radii = np.sqrt(np.sum(pts**2, axis=1))
    boundary_dists = 1.0 - radii
    
    ii, jj = np.triu_indices(n, k=1)
    pair_vals = dists[ii, jj] / 2.0
    bound_vals = boundary_dists
    all_vals = np.concatenate([pair_vals, bound_vals])
    min_val = np.min(all_vals)
    
    # Identify points involved in active constraints
    active_mask = np.abs(all_vals - min_val) < 1e-5
    active_pairs = np.where(active_mask[:len(ii)])[0]
    active_bounds = np.where(active_mask[len(ii):])[0]
    
    constrained_points = set()
    for idx in active_pairs:
        constrained_points.add(ii[idx])
        constrained_points.add(jj[idx])
    for idx in active_bounds:
        constrained_points.add(idx)
    
    # Limit to max_points for time management
    if len(constrained_points) > max_points:
        # Prioritize points with smallest boundary distance or pair distance
        point_scores = np.zeros(n)
        for i in range(n):
            min_pair_dist = np.min(dists[i])
            point_scores[i] = min(min_pair_dist / 2.0, boundary_dists[i])
        constrained_list = list(constrained_points)
        constrained_list.sort(key=lambda i: point_scores[i])
        constrained_points = set(constrained_list[:max_points])
    
    for i in sorted(constrained_points):
        if time.time() - start_time > time_limit:
            break
        
        p_i = pts[i].copy()
        
        def obj(x):
            diffs = pts - x
            d = np.linalg.norm(diffs, axis=1)
            d[i] = np.inf
            min_pair = np.min(d) / 2.0
            min_bound = 1.0 - np.linalg.norm(x)
            return -min(min_pair, min_bound)
        
        try:
            res = minimize(obj, p_i, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
            # Only accept if it improves
            old_pts = pts.copy()
            old_score = _compute_true_score(old_pts)
            pts[i] = res.x
            new_score = _compute_true_score(pts)
            if new_score <= old_score:
                pts[i] = old_pts[i]
        except Exception:
            pass
    
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
        if time.time() - start_time > time_limit:
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
    time_limit = 140.0  # Leave 10s margin before 150s kill
    
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
    
    # Per-point refinement on most constrained points
    if best_points is not None and time.time() - start_time < time_limit - 3:
        current = _per_point_refine(best_points, max_points=30, time_limit=8.0)
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
