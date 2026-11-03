import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_r_and_grad(points):
    """
    Compute r and its gradient w.r.t. all point coordinates.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
    Returns (r, grad) where grad has shape (N, 2).
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    
    r = min(np.min(boundary), np.min(pair_dists))
    
    # Gradient
    grad = np.zeros((n, 2))
    
    # Boundary contribution
    min_boundary = np.min(boundary)
    if abs(min_boundary - r) < 1e-12:
        for i in range(n):
            if abs(boundary[i] - min_boundary) < 1e-12:
                if norms[i] > 1e-12:
                    grad[i] -= points[i] / norms[i]
    
    # Pair contribution
    min_pair = np.min(pair_dists)
    if abs(min_pair - r) < 1e-12:
        for idx in range(len(pair_dists)):
            if abs(pair_dists[idx] - min_pair) < 1e-12:
                i, j = iu[0][idx], iu[1][idx]
                d_vec = points[i] - points[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-12:
                    grad[i] += d_vec / (2 * d_norm)
                    grad[j] -= d_vec / (2 * d_norm)
    
    return r, grad

def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))

def _compute_min_dist(points):
    """
    Compute the minimum distance metric r for a given configuration.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
    Returns (r, bottleneck_type, bottleneck_indices)
    bottleneck_type: 0 for boundary, 1 for pair
    bottleneck_indices: indices of the bottleneck constraint(s)
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Boundary distances
    radii = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(radii)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    min_pair = np.min(pair_dists)
    
    if min_boundary < min_pair:
        # Bottleneck is boundary
        # Find all indices where boundary distance is close to min
        close_indices = np.where(np.abs(radii - min_boundary) < 1e-10)[0]
        return min_boundary, 0, close_indices
    else:
        # Bottleneck is pair
        # Find all pairs where distance is close to min
        close_pairs = np.where(np.abs(pair_dists - min_pair) < 1e-10)[0]
        # Map back to (i, j)
        iu_arr = np.array(iu)
        close_idx_pairs = iu_arr[close_pairs]
        return min_pair, 1, close_idx_pairs

def _score_vec(points):
    """Vectorized score computation for quick evaluation."""
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

def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points

def _local_optimize_lbfgs(c, time_limit=10.0):
    """Local optimization using L-BFGS-B with smooth-min objective."""
    pair_idx = _pairwise_indices(N)
    
    def neg_soft_min(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 500.0
        # Numerically stable soft-min
        m = np.min(all_vals)
        return m + np.log(np.sum(np.exp(-beta * (all_vals - m)))) / beta
    
    def neg_soft_min_grad(x):
        c = x.reshape(N, 2)
        ci = c[pair_idx[:, 0]]
        cj = c[pair_idx[:, 1]]
        d = np.sqrt(np.sum((ci - cj) ** 2, axis=1)) / 2
        b = 1 - np.hypot(c[:, 0], c[:, 1])
        all_vals = np.concatenate([d, b])
        beta = 500.0
        m = np.min(all_vals)
        weights = np.exp(-beta * (all_vals - m))
        weights /= np.sum(weights)
        
        # Gradient of soft-min w.r.t. all_vals
        grad_vals = -weights
        
        # Gradient w.r.t. c
        grad_c = np.zeros((N, 2))
        
        # Pair distances gradient
        d_grad = np.zeros_like(d)
        for idx in range(len(d)):
            i, j = pair_idx[idx, 0], pair_idx[idx, 1]
            d_vec = c[i] - c[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-12:
                d_grad[idx] = grad_vals[idx]
        
        # Accumulate pair gradients
        for idx in range(len(d)):
            if abs(d_grad[idx]) > 1e-15:
                i, j = pair_idx[idx, 0], pair_idx[idx, 1]
                d_vec = c[i] - c[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-12:
                    unit = d_vec / d_norm
                    grad_c[i] += d_grad[idx] * unit / 2
                    grad_c[j] -= d_grad[idx] * unit / 2
        
        # Boundary gradient
        for idx in range(N):
            if abs(grad_vals[len(d) + idx]) > 1e-15:
                norm = np.linalg.norm(c[idx])
                if norm > 1e-12:
                    grad_c[idx] -= grad_vals[len(d) + idx] * c[idx] / norm
        
        return grad_c.flatten()
    
    t_start = time.time()
    try:
        res = minimize(
            neg_soft_min,
            c.flatten(),
            jac=neg_soft_min_grad,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success or res.fun < neg_soft_min(c.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return c

def _polish_gradient_descent(points, time_limit=5.0):
    """
    High-precision gradient descent polish using exact subgradient of the min function.
    """
    pts = points.copy().flatten()
    t_start = time.time()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    lr = 0.005  # Learning rate
    momentum = 0.8
    v = np.zeros_like(pts)
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - t_start < time_limit:
        c = pts.reshape(N, 2)
        r, grad = _compute_r_and_grad(c)
        
        if r > best_score + 1e-15:
            best_score = r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reduce learning rate and reset momentum
                lr *= 0.5
                v *= 0.5
                no_improve_count = 0
                if lr < 1e-8:
                    break
        
        # Normalize gradient for stability
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        else:
            break
        
        # Momentum update
        v = momentum * v + lr * grad
        pts = pts + v
        
        # Project back to feasible region (simple clamp)
        c_new = pts.reshape(N, 2)
        norms = np.linalg.norm(c_new, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                scale = (1.0 - 1e-10) / norms[i]
                c_new[i] *= scale
                pts[i*2:(i+1)*2] = c_new[i]
    
    return best_pts.reshape(N, 2), best_score

def construct():
    t_start = time.time()
    
    # 1. Hexagonal Initialization
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # 2. Coarse Optimization with L-BFGS (with gradient)
    if time.time() < DEADLINE - 40:
        c_opt = _local_optimize_lbfgs(points, time_limit=min(25.0, DEADLINE - (time.time() - t_start) - 40))
        r_opt = _score_vec(c_opt)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
    
    # 3. Fine Optimization with Gradient Descent Polish
    if time.time() < DEADLINE - 20:
        polished, r_polish = _polish_gradient_descent(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 20))
        if r_polish > best_score:
            best_score = r_polish
            best_points = polished.copy()
    
    # 4. Multi-Start Perturb and Polish
    # Try to escape local minima by perturbing the best solution
    n_restarts = 0
    while time.time() < DEADLINE - 10 and n_restarts < 8:
        # Perturb all points slightly
        perturbation_scale = 0.03 / (1.0 + 0.3 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * perturbation_scale
        
        # Coarse optimize the perturbed solution
        c_opt = _local_optimize_lbfgs(x0, time_limit=min(12.0, DEADLINE - (time.time() - t_start) - 10))
        r_opt = _score_vec(c_opt)
        
        # Fine polish
        if time.time() < DEADLINE - 3:
            polished, r_polish = _polish_gradient_descent(c_opt, time_limit=min(8.0, DEADLINE - (time.time() - t_start) - 3))
            r_opt = max(r_opt, r_polish)
            if r_polish > r_opt:
                c_opt = polished
        
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()
        
        n_restarts += 1
    
    # 5. Final Polish with tighter tolerance
    if time.time() < DEADLINE - 2:
        polished, r_final = _polish_gradient_descent(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 2))
        if r_final > best_score:
            best_points = polished
            best_score = r_final
    
    # Ensure validity: clamp points that might be slightly outside
    c_final = best_points.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            scale = (1 - best_score) / r2[i]
            c_final[i] *= scale
    
    # Final verification and potential micro-adjustment
    r_check = _score_vec(c_final)
    if r_check < best_score - 1e-12:
        # Slight shrink to ensure validity
        c_final *= (1.0 - 1e-10)
    
    return c_final
