"""Pack 50 equal circles in the unit square [-1/2, 1/2]^2 to maximize the common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50


def _score(pts):
    """Compute the maximum common radius for a set of centers (N,2)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Wall constraint: min over i,k of (1/2 - |c_ik|)
    wall = 0.5 - np.abs(pts).max()
    if wall < 0:
        return 0.0
    # Pairwise constraint: min over i<j of |c_i - c_j| / 2
    # Use efficient computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    r = min(wall, min_pair / 2.0)
    return r


def _initial_grid():
    """Create an initial 50-point grid configuration."""
    m = int(np.ceil(np.sqrt(N)))  # 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_hexagonal():
    """Create an initial hexagonal-packing-like configuration."""
    # Use a hexagonal lattice clipped to the square
    pts = []
    r_approx = 0.07
    dx = 2 * r_approx
    dy = dx * np.sqrt(3) / 2
    x0, y0 = -0.5 + r_approx, -0.5 + r_approx
    row = 0
    while len(pts) < N:
        y = y0 + row * dy
        if y > 0.5 - r_approx:
            break
        x = x0 if row % 2 == 0 else x0 + dx / 2
        while x <= 0.5 - r_approx:
            if len(pts) < N:
                pts.append([x, y])
            x += dx
        row += 1
    # If we don't have enough, fill remaining with a grid
    while len(pts) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        for i, (x, y) in enumerate([(x, y) for y in ticks for x in ticks]):
            if i >= len(pts) and len(pts) < N:
                pts.append([x, y])
            if len(pts) >= N:
                break
    return np.array(pts[:N])


def _flatten(pts):
    return pts.flatten()


def _unflatten(x):
    return x.reshape(N, 2)


def _objective_and_grad(x):
    """
    Objective: minimize -r where r = min(wall, min_pair/2).
    We use a smooth surrogate: soft-min of the constraints.
    """
    pts = _unflatten(x)
    # Wall distances
    walls = 0.5 - np.abs(pts)  # shape (N, 2)
    wall_vals = walls.flatten()  # shape (2N,)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists[np.triu_indices(N, k=1)] / 2.0  # shape (N*(N-1)/2,)
    
    # Combined constraints
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft-min using log-sum-exp trick for stability
    # softmin(x) = -logsumexp(-x)
    # For numerical stability, shift by min
    m = np.min(all_vals)
    if m < 0:
        m = 0.0
    shifted = all_vals - m
    # softmin ≈ m - logsumexp(-shifted) but for large N this is close to m
    # Use a temperature parameter
    T = 0.005
    neg_shifted = -shifted / T
    # clip for stability
    neg_shifted = np.clip(neg_shifted, -500, 500)
    max_neg = np.max(neg_shifted)
    lse = max_neg + np.log(np.sum(np.exp(neg_shifted - max_neg)))
    soft_min = m - T * lse
    
    # Gradient of soft_min w.r.t. x
    # d(soft_min)/d(all_vals_j) = -exp(-all_vals_j/T) / sum_k exp(-all_vals_k/T)
    weights = np.exp(-shifted / T)
    weights = np.clip(weights, 0, np.inf)
    w_sum = np.sum(weights)
    if w_sum < 1e-300:
        w_sum = 1.0
    grad_all = -weights / w_sum  # d(soft_min)/d(all_vals)
    
    # Now backpropagate to x
    grad_x = np.zeros_like(x)
    
    # Wall part: all_vals[0:2N] = walls.flatten()
    # walls[i,k] = 0.5 - |pts[i,k]|
    # d(walls[i,k])/d(pts[i,k]) = -sign(pts[i,k])
    wall_grad = grad_all[:2 * N].reshape(N, 2)
    sign_pts = np.sign(pts)
    # d(soft_min)/d(pts) from walls: wall_grad * (-sign_pts)
    grad_x += (wall_grad * (-sign_pts)).flatten()
    
    # Pair part: all_vals[2N:] = pair_vals
    # pair_vals[j] = dists[i,j]/2 for i<j
    # d(dists[i,j])/d(pts[i]) = (pts[i]-pts[j])/dists[i,j]
    # d(dists[i,j])/d(pts[j]) = (pts[j]-pts[i])/dists[i,j]
    # So d(pair_vals[j])/d(pts[i]) = (pts[i]-pts[j])/(2*dists[i,j])
    
    pair_grad = grad_all[2 * N:]
    # Reconstruct the pairwise gradient
    idx_i, idx_j = np.triu_indices(N, k=1)
    diff_ij = pts[idx_i] - pts[idx_j]
    dists_ij = np.sqrt((diff_ij ** 2).sum(axis=1))
    # Avoid division by zero
    mask = dists_ij > 1e-12
    grad_ij = np.zeros_like(diff_ij)
    if np.any(mask):
        grad_ij[mask] = diff_ij[mask] / (2.0 * dists_ij[mask, None])
    
    # Accumulate gradients
    for k in range(len(pair_grad)):
        i, j = idx_i[k], idx_j[k]
        grad_x[2 * i:2 * i + 2] += pair_grad[k] * grad_ij[k]
        grad_x[2 * j:2 * j + 2] += pair_grad[k] * (-grad_ij[k])
    
    # Objective is -soft_min, so gradient is -grad_x
    return -soft_min, -grad_x


def _constrained_optimize(pts, deadline):
    """Run SLSQP optimization with box constraints."""
    x0 = _flatten(pts)
    
    # Box constraints: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        val, _ = _objective_and_grad(x)
        return val
    
    def grad(x):
        _, g = _objective_and_grad(x)
        return g
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    start = time.time()
    
    # Use SLSQP with gradient
    try:
        res = minimize(
            obj, x0, jac=grad, method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if time.time() > deadline:
            return best_pts, best_score
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            candidate = _unflatten(res.x)
            # Clip to valid range
            candidate = np.clip(candidate, -0.5, 0.5)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    except Exception:
        pass
    
    # Try L-BFGS-B as well
    start2 = time.time()
    try:
        res2 = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False, 'gtol': 1e-8}
        )
        if time.time() > deadline:
            return best_pts, best_score
        
        if res2.x is not None and np.all(np.isfinite(res2.x)):
            candidate = _unflatten(res2.x)
            candidate = np.clip(candidate, -0.5, 0.5)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    except Exception:
        pass
    
    return best_pts, best_score


def _local_refinement(pts, deadline):
    """Do a local refinement using a higher-temperature soft-min followed by lower."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Try different temperatures for the soft-min
    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for T in temperatures:
        if time.time() > deadline:
            break
        
        def obj_T(x):
            pts_x = _unflatten(x)
            walls = (0.5 - np.abs(pts_x)).flatten()
            diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=-1))
            np.fill_diagonal(dists, np.inf)
            pair_vals = dists[np.triu_indices(N, k=1)] / 2.0
            all_vals = np.concatenate([walls, pair_vals])
            m = np.min(all_vals)
            if m < 0:
                m = 0.0
            shifted = all_vals - m
            neg_shifted = np.clip(-shifted / T, -500, 500)
            max_neg = np.max(neg_shifted)
            lse = max_neg + np.log(np.sum(np.exp(neg_shifted - max_neg)))
            soft_min = m - T * lse
            return -soft_min
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        x0 = _flatten(best_pts)
        
        try:
            res = minimize(
                obj_T, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                candidate = _unflatten(res.x)
                candidate = np.clip(candidate, -0.5, 0.5)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    """Construct the packing of 50 equal circles in the unit square."""
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Start with multiple initial configurations
    inits = [
        _initial_grid(),
        _initial_hexagonal(),
    ]
    
    # Add a few random perturbations of the grid
    rng = np.random.default_rng(42)
    for _ in range(3):
        g = _initial_grid()
        g += rng.normal(0, 0.01, g.shape)
        g = np.clip(g, -0.5, 0.5)
        inits.append(g)
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        pts, score = _constrained_optimize(init, deadline)
        if score > best_score:
            best_pts = pts
            best_score = score
    
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    # Local refinement
    if time.time() < deadline - 10:
        best_pts, best_score = _local_refinement(best_pts, deadline)
    
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    final_score = _score(best_pts)
    
    if final_score <= 0:
        # Fallback to grid
        best_pts = _initial_grid()
    
    return best_pts
