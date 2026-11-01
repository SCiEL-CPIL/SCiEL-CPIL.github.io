"""Packing of 50 equal circles in a unit square via vectorized optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)

def _compute_radius(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r_wall = float(np.min(d_wall))
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = float(np.min(dist[IU])) / 2.0
    
    return min(r_wall, r_pair)

def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations. Vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Pairwise violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
    
    # Wall violations
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d_wall)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))

def _violation_grad(pts, r_target):
    """Analytical gradient of violation for target radius. Vectorized."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    grad = np.zeros((N, 2))
    
    # Pairwise gradient
    # We need to sum over all j != i for each i: 
    # grad_i += (pts_i - pts_j) * (2*r - dist_ij) / dist_ij^2  (if dist_ij < 2*r)
    # Actually, derivative of max(0, 2r - d) wrt p_i is -(p_i - p_j)/d * (1 if d < 2r else 0)
    # Let's use the vectorized form:
    # W_ij = (2*r - dist_ij) / dist_ij^2 if dist_ij < 2*r else 0
    # This is not quite right. 
    # f(x) = max(0, 2r - ||x_i - x_j||)
    # grad_xi f = -(x_i - x_j) / ||x_i - x_j|| if ||x_i - x_j|| < 2r else 0
    
    mask = dist < 2.0 * r_target
    # Avoid division by zero
    safe_dist = np.where(mask, dist, 1.0)
    weights = np.where(mask, (2.0 * r_target - safe_dist) / safe_dist, 0.0)
    # Wait, the gradient of max(0, C - d) is -1 * grad(d) if C-d > 0.
    # grad(d) wrt x_i is (x_i - x_j) / d.
    # So grad contribution is -(x_i - x_j) / d.
    
    # We want to sum over j for each i.
    # Let D be the matrix of (x_i - x_j).
    # We want sum_j mask_ij * (-(x_i - x_j) / dist_ij)
    
    # Normalize directions
    dir_vecs = diff / safe_dist[:, :, None] # Shape (N, N, 2)
    # Multiply by weights (N, N)
    # weight_matrix = mask * (1.0 / safe_dist) ? No.
    # The term is just -1 * (x_i - x_j) / d_ij.
    # So we sum -dir_vecs[i,j] over all j where mask[i,j] is True.
    
    # Construct gradient from pairwise interactions
    # grad[i] = sum_j (mask[i,j] * -(pts[i] - pts[j]) / dist[i,j])
    
    # Vectorized sum:
    # G[i] = sum_j (M[i,j] * -D[i,j] / D_norm[i,j])
    
    # D is pts[:, None, :] - pts[None, :, :]
    # D_norm is dist
    
    # Let's compute the sum directly
    # For each i, sum over j != i
    
    # This is equivalent to:
    # grad[i] = - sum_j (mask[i,j] * (pts[i] - pts[j]) / dist[i,j])
    
    # We can do this by creating a matrix of unit vectors and summing.
    unit_vecs = diff / safe_dist[:, :, None]
    # Apply mask
    unit_vecs = unit_vecs * mask[:, :, None]
    # Sum over j (axis=1)
    grad_pair = -np.sum(unit_vecs, axis=1)
    
    grad += grad_pair
    
    # Wall gradient
    # If pts[i,0] < -HALF + r, push right (grad[0] += 1)
    # If pts[i,0] > HALF - r, push left (grad[0] -= 1)
    # etc.
    
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    # Wall violations are max(0, r - d_wall)
    # Grad is 1 if violated
    
    # Left wall: d_left < r => pts[i,0] < -HALF + r
    mask_left = d_left < r_target
    mask_right = d_right < r_target
    mask_bottom = d_bottom < r_target
    mask_top = d_top < r_target
    
    grad[:, 0] += mask_left.astype(float)
    grad[:, 0] -= mask_right.astype(float)
    grad[:, 1] += mask_bottom.astype(float)
    grad[:, 1] -= mask_top.astype(float)
    
    return grad

def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=5000):
    """Fast simulated annealing to minimize violation."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.01
    t_end = 0.00001
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        temp = t_start * (t_end / t_start) ** progress
        
        # Move 1 or 2 random circles
        n_move = rng.integers(1, 3)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol

def _lbfgs_polish(pts0, r_target, deadline):
    """Polish using L-BFGS-B to minimize violation for target radius."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation(p, r_target)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_grad(p, r_target).ravel()
    
    try:
        res = minimize(obj, pts0.ravel(), method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 20})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts, float(res.fun)
    except Exception:
        return pts0, _violation(pts0, r_target)

def _soft_min_obj(x, beta):
    p = x.reshape(N, 2)
    # Wall distances
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    # Pairwise distances / 2
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    
    # Soft min
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    
    return -r_soft

def _soft_min_grad(x, beta):
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    # d_wall[i] = min(...) 
    # This is tricky because of the min. 
    # Let's just use the gradient of the sum of walls and pairs separately? 
    # No, the soft min combines them.
    # We need gradients of each d_i.
    
    # For walls, the gradient is non-zero only for the active wall.
    # d_left = p[:,0] + 0.5 -> grad (1, 0)
    # d_right = 0.5 - p[:,0] -> grad (-1, 0)
    # d_bottom = p[:,1] + 0.5 -> grad (0, 1)
    # d_top = 0.5 - p[:,1] -> grad (0, -1)
    
    # We need to know which wall is active for each point.
    d_left = p[:, 0] + HALF
    d_right = HALF - p[:, 0]
    d_bottom = p[:, 1] + HALF
    d_top = HALF - p[:, 1]
    
    # Active wall is the one with min distance
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_idx = np.argmin(walls, axis=1)
    
    # Assign gradients
    for i in range(N):
        w = weights[i] # Weight for this point's wall constraint
        if w > 1e-10:
            if min_idx[i] == 0: grad[i, 0] += w
            elif min_idx[i] == 1: grad[i, 0] -= w
            elif min_idx[i] == 2: grad[i, 1] += w
            else: grad[i, 1] -= w
            
    # Pair gradients
    # d_pair[k] = dist[i,j] / 2
    # grad p_i = (p_i - p_j) / (2 * dist)
    # grad p_j = (p_j - p_i) / (2 * dist)
    
    for k in range(len(IU[0])):
        i, j = IU[0][k], IU[1][k]
        w = weights[N + k]
        if w > 1e-10:
            dij = p[i] - p[j]
            dist_ij = np.linalg.norm(dij)
            if dist_ij > 1e-10:
                g = w * dij / (2.0 * dist_ij)
                grad[i] += g
                grad[j] -= g
                
    return -grad.ravel() # We are maximizing r, so minimizing -r

def _hex_init(rng, jitter=0.0):
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def construct():
    start = time.time()
    deadline = start + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps, deadline_offset=5.0):
        nonlocal best_pts, best_r
        if time.time() > deadline - deadline_offset:
            return
        
        # SA
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - deadline_offset, rng, n_steps=n_sa_steps)
        
        # L-BFGS
        pts_lbfgs, viol_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - deadline_offset)
        
        r = _compute_radius(pts_lbfgs)
        if r > best_r:
            best_r = r
            best_pts = pts_lbfgs.copy()
    
    # Initializations
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    
    # Phase 1: Get a good starting point
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        try_improve(init, 0.07, 2000)
        
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search / Push radius
    # Start from current best_r and push higher
    r_lo = best_r
    r_hi = 0.07138
    
    # We can do a few iterations of pushing
    n_push_iters = 8
    for i in range(n_push_iters):
        if time.time() > deadline - 15.0:
            break
        r_mid = (r_lo + r_hi) / 2.0
        # Try to achieve r_mid
        # Use SA + L-BFGS
        try_improve(best_pts, r_mid, 3000, deadline_offset=10.0)
        
        if _compute_radius(best_pts) >= r_mid - 1e-9:
            r_lo = r_mid
        else:
            r_hi = r_mid
            
    # Phase 3: High precision polish using soft-min
    if time.time() < deadline - 10.0:
        def obj(x):
            if time.time() > deadline - 2.0:
                raise RuntimeError("timeout")
            return _soft_min_obj(x, 100.0)
        
        def grad(x):
            if time.time() > deadline - 2.0:
                raise RuntimeError("timeout")
            return _soft_min_grad(x, 100.0)
        
        try:
            res = minimize(obj, best_pts.ravel(), method='L-BFGS-B', jac=grad,
                           bounds=[(-HALF, HALF)] * (2 * N),
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10})
            pts_soft = res.x.reshape(N, 2)
            pts_soft = np.clip(pts_soft, -HALF, HALF)
            r_soft = _compute_radius(pts_soft)
            if r_soft > best_r:
                best_r = r_soft
                best_pts = pts_soft
        except Exception:
            pass
            
    # Final check
    return best_pts
