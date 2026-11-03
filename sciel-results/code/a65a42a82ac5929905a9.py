"""Optimized packing of 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the maximum valid radius for given points in [-1/2, 1/2]^3."""
    if len(pts) < 2:
        return 0.5
    d = pdist(pts)
    min_pair_dist = d.min() / 2.0
    max_coord = np.abs(pts).max(axis=1)
    wall_dist = (0.5 - max_coord).min()
    return min(min_pair_dist, wall_dist)


def _repulsion_energy(pts, t=0.01):
    """
    Smooth surrogate: minimize -log-sum-exp of all constraints.
    Constraints: pairwise distances >= 2r, wall distances >= r.
    We maximize r, so we minimize -r + penalty for violations.
    
    Actually, let's use a different approach: minimize the maximum violation.
    Use soft-min via log-sum-exp for a smooth objective.
    
    Better: minimize f(pts) = -min(min_pair_dist/2, min_wall_dist)
    But this is non-smooth. Use a smooth approximation.
    
    Alternative: minimize the sum of squared violations of r=0.111 constraints,
    but we want to find the optimal r.
    
    Let's use: maximize r where r = min(constraints).
    Equivalent to: minimize -r subject to all constraints >= r.
    
    For scipy, let's minimize a smooth function that approximates the negative radius.
    """
    pts = np.array(pts).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts).max(axis=1)
    
    # We want to maximize min(min_pair_dist/2, min_wall_dist)
    # = maximize min over all constraints
    # Use log-sum-exp trick: -log-sum-exp(-constraints) approximates min(constraints)
    # for large negative temperature
    
    # All constraints (in units of radius, i.e., half-distances)
    # Pair constraints: dist/2 for each pair
    # Wall constraints: wall_dist for each point
    
    # To keep it efficient, let's use a weighted sum approach
    # Minimize: -alpha * min_constraint + beta * sum of violations
    
    # Actually, let's just directly minimize a smooth approximation
    # of the negative radius using a power mean or similar.
    
    # For efficiency, compute the minimum constraint and use a smooth version
    pair_constraints = np.sqrt(dist_sq) / 2.0
    pair_constraints = pair_constraints[np.triu_indices(n, k=1)]
    all_constraints = np.concatenate([pair_constraints, wall_dists])
    
    # Smooth minimum: -log(sum(exp(-c/t)))*t approximates min(c)
    # Use a moderate t for smoothness
    t_smooth = 0.005
    # Clip to avoid overflow
    neg_c = -all_constraints / t_smooth
    neg_c = np.clip(neg_c, -500, 500)
    log_sum = np.log(np.sum(np.exp(neg_c)))
    smooth_min = -t_smooth * log_sum
    
    return -smooth_min


def _repulsion_energy_grad(pts, t=0.01):
    """Gradient of the repulsion energy (negative smooth minimum radius)."""
    pts = np.array(pts).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return np.zeros_like(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    wall_dists = 0.5 - np.abs(pts).max(axis=1)
    
    pair_constraints = dist / 2.0
    pair_constraints_flat = pair_constraints[np.triu_indices(n, k=1)]
    all_constraints = np.concatenate([pair_constraints_flat, wall_dists])
    
    t_smooth = 0.005
    neg_c = -all_constraints / t_smooth
    neg_c = np.clip(neg_c, -500, 500)
    weights = np.exp(neg_c)
    weights /= weights.sum()
    
    # Gradient of -smooth_min w.r.t. each constraint is -weights
    # So gradient of objective = sum over constraints of (-weight) * (gradient of constraint)
    
    grad = np.zeros_like(pts)
    
    # Pair contributions
    pair_idx = np.triu_indices(n, k=1)
    for idx in range(len(pair_idx[0])):
        i, j = pair_idx[0][idx], pair_idx[1][idx]
        w = weights[idx] * t_smooth  # d(-smooth_min)/d(constraint) = -w, but we want grad of objective
        # Actually: objective = -smooth_min, d(objective)/d(c_k) = -weight_k
        # d(c_k)/d(pts) for pair (i,j): c_k = |p_i - p_j|/2
        # d(c_k)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|) = (p_i - p_j) / (2 * 2*c_k) = (p_i-p_j)/(4*c_k)
        # d(c_k)/d(p_j) = (p_j - p_i) / (4*c_k)
        
        d = pts[i] - pts[j]
        dist_ij = np.linalg.norm(d)
        if dist_ij < 1e-10:
            continue
        # d(constraint)/d(pts[i]) = d / (2 * dist_ij) = d / (4 * constraint)
        grad[i] += -w * d / (2 * dist_ij)
        grad[j] += -w * (-d) / (2 * dist_ij)
    
    # Wall contributions
    for idx in range(n):
        w = weights[len(pair_constraints_flat) + idx] * t_smooth
        # constraint = 0.5 - max(|x|, |y|, |z|)
        # d(constraint)/d(pts) = -sign(pts) for the max coordinate
        max_coord_idx = np.argmax(np.abs(pts[idx]))
        g = np.zeros(3)
        g[max_coord_idx] = -np.sign(pts[idx, max_coord_idx])
        if pts[idx, max_coord_idx] == 0:
            g[max_coord_idx] = 0
        grad[idx] += -w * g
    
    return grad


def _optimize_from_init(pts, deadline):
    """Run L-BFGS-B optimization from a starting point."""
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    
    # Flatten
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        # Clip to cube for evaluation
        p = np.clip(p, -0.5, 0.5)
        return _repulsion_energy(p)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = np.clip(p, -0.5, 0.5)
        g = _repulsion_energy_grad(p)
        return g.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Try a few different smoothness parameters
    for t_smooth in [0.005, 0.01, 0.02]:
        if time.time() > deadline - 10:
            break
        
        # Monkey-patch the smooth parameter
        global _T_SMOOTH
        _T_SMOOTH = t_smooth
        
        res = minimize(
            objective, x0, method='L-BFGS-B', jac=gradient,
            bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if time.time() > deadline - 5:
            break
        
        new_pts = res.x.reshape(-1, 3)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
    
    return best_pts, best_r


def _grid_init(n):
    """Generate a grid initialization."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    # Adjust to get exactly n points
    # Try different grid shapes
    best_pts = None
    best_r = 0
    
    for m1 in range(max(1, m - 2), m + 3):
        for m2 in range(max(1, m - 2), m + 3):
            m3 = max(1, int(round(n / (m1 * m2))))
            if m1 * m2 * m3 < n:
                continue
            
            ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
            
            grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
            if len(grid) >= n:
                subset = grid[:n]
                r = _compute_radius(subset)
                if r > best_r:
                    best_r = r
                    best_pts = subset
    
    if best_pts is None:
        m = int(np.ceil(n ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:n]
        best_r = _compute_radius(best_pts)
    
    return best_pts, best_r


def _bcc_init(n):
    """Generate BCC lattice initialization."""
    best_pts = None
    best_r = 0
    
    for a in np.linspace(0.22, 0.38, 25):
        n_cells = int(1.0 / a)
        pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + i * a
                    y = -0.5 + j * a
                    z = -0.5 + k * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        pts.append([x, y, z])
                    x2 = -0.5 + (i + 0.5) * a
                    y2 = -0.5 + (j + 0.5) * a
                    z2 = -0.5 + (k + 0.5) * a
                    if -0.5 <= x2 <= 0.5 and -0.5 <= y2 <= 0.5 and -0.5 <= z2 <= 0.5:
                        pts.append([x2, y2, z2])
        
        pts = np.array(pts)
        if len(pts) < n:
            continue
        
        # Select n points with best spacing
        # Use greedy: pick points that maximize minimum distance
        np.random.seed(42)
        for trial in range(20):
            if trial == 0:
                subset = pts[:n]
            else:
                idx = np.random.choice(len(pts), n, replace=False)
                subset = pts[idx]
            
            r = _compute_radius(subset)
            if r > best_r:
                best_r = r
                best_pts = subset
    
    if best_pts is None:
        pts, r = _grid_init(n)
        return pts, r
    
    return best_pts, best_r


def _random_init(n, seed=None):
    """Generate random initialization."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(n, 3))
    r = _compute_radius(pts)
    return pts, r


def _local_polish(pts, deadline):
    """Final high-precision polish using steepest descent on the exact objective."""
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    lr = 0.001
    for it in range(200):
        if time.time() > deadline - 2:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_d = dist[i, j]
        
        wall_dists = 0.5 - np.abs(pts).max(axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_d = wall_dists[min_wall_idx]
        
        if min_d / 2.0 < min_wall_d:
            direction = pts[i] - pts[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            pts[i] += lr * direction
            pts[j] -= lr * direction
        else:
            p = pts[min_wall_idx]
            for k in range(3):
                if np.abs(p[k]) == np.abs(p).max():
                    if p[k] > 0:
                        pts[min_wall_idx, k] -= lr
                    else:
                        pts[min_wall_idx, k] += lr
                    break
        
        pts = np.clip(pts, -0.5, 0.5)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        else:
            pts = best_pts.copy()
            lr *= 0.5
    
    return best_pts, best_r


_T_SMOOTH = 0.01  # Global smoothness parameter


def _repulsion_energy(pts, t=None):
    """Smooth surrogate objective."""
    if t is None:
        t = _T_SMOOTH
    pts = np.array(pts).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return 0.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    wall_dists = 0.5 - np.abs(pts).max(axis=1)
    
    pair_constraints = dist / 2.0
    pair_constraints_flat = pair_constraints[np.triu_indices(n, k=1)]
    all_constraints = np.concatenate([pair_constraints_flat, wall_dists])
    
    neg_c = -all_constraints / t
    neg_c = np.clip(neg_c, -500, 500)
    log_sum = np.log(np.sum(np.exp(neg_c)))
    smooth_min = -t * log_sum
    
    return -smooth_min


def _repulsion_energy_grad(pts, t=None):
    """Gradient of the smooth surrogate objective."""
    if t is None:
        t = _T_SMOOTH
    pts = np.array(pts).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return np.zeros_like(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    wall_dists = 0.5 - np.abs(pts).max(axis=1)
    
    pair_constraints = dist / 2.0
    pair_constraints_flat = pair_constraints[np.triu_indices(n, k=1)]
    all_constraints = np.concatenate([pair_constraints_flat, wall_dists])
    
    neg_c = -all_constraints / t
    neg_c = np.clip(neg_c, -500, 500)
    weights = np.exp(neg_c)
    weights /= weights.sum()
    
    grad = np.zeros_like(pts)
    
    pair_idx = np.triu_indices(n, k=1)
    for idx in range(len(pair_idx[0])):
        i, j = pair_idx[0][idx], pair_idx[1][idx]
        w = weights[idx]
        d = pts[i] - pts[j]
        dist_ij = np.linalg.norm(d)
        if dist_ij < 1e-10:
            continue
        grad[i] += -w * d / (2 * dist_ij)
        grad[j] += -w * (-d) / (2 * dist_ij)
    
    for idx in range(n):
        w = weights[len(pair_constraints_flat) + idx]
        max_coord_idx = np.argmax(np.abs(pts[idx]))
        g = np.zeros(3)
        g[max_coord_idx] = -np.sign(pts[idx, max_coord_idx])
        if pts[idx, max_coord_idx] == 0:
            g[max_coord_idx] = 0
        grad[idx] += -w * g
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    global _T_SMOOTH
    
    # Multiple initializations
    inits = []
    
    # Grid init
    pts_grid, r_grid = _grid_init(N)
    inits.append((pts_grid, r_grid))
    
    # BCC init
    pts_bcc, r_bcc = _bcc_init(N)
    inits.append((pts_bcc, r_bcc))
    
    # Random inits
    for seed in [42, 123, 456, 789]:
        pts_rand, r_rand = _random_init(N, seed=seed)
        inits.append((pts_rand, r_rand))
    
    # Sort by initial radius
    inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = inits[0][0]
    best_r = inits[0][1]
    
    # Optimize from each initialization
    for init_idx, (init_pts, init_r) in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Skip if this init is much worse than current best
        if init_r < best_r * 0.8 and init_idx > 0:
            continue
        
        try:
            opt_pts, opt_r = _optimize_from_init(init_pts, deadline)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts
        except Exception:
            continue
        
        if time.time() > deadline - 15:
            break
    
    # Final polish
    if time.time() < deadline - 5:
        polished_pts, polished_r = _local_polish(best_pts, deadline)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
