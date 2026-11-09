"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 140.0  # safety margin before 150 s evaluator kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys
from scipy.optimize import minimize

def _grid_score(points, r):
    """Compute the largest radius that fits for the given points."""
    if len(points) < 2:
        return 0.5
    # min pairwise distance
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    return min(min_pair, min_boundary) / 2.0

def _exp_repulsion_objective(x, beta, eps):
    """
    Objective: Sum of exp(-beta * dist) for pairs + exp(-beta * margin) for boundaries.
    Minimizing this pushes points apart and away from walls.
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    
    # Pairwise
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # To avoid overflow in exp, we can just sum exp(-beta * dists)
    # Since dists >= 0, exp(-beta * dists) <= 1.
    pair_term = np.sum(np.exp(-beta * dists) * mask)
    
    # Boundary
    # Use smooth absolute value: sqrt(x^2 + eps^2)
    margins = CUBE_HALF - np.sqrt(pts**2 + eps**2)
    bound_term = np.sum(np.exp(-beta * margins))
    
    return pair_term + bound_term

def _exp_repulsion_grad(x, beta, eps):
    """
    Gradient of the exp repulsion objective.
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Pairwise
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # d/dx_i sum_{j!=i} exp(-beta * d_ij)
    # = sum_{j!=i} exp(-beta * d_ij) * (-beta) * (x_i - x_j) / d_ij
    weights = np.exp(-beta * dists) * mask
    # w[i, j] is weight for pair (i, j)
    # grad[i] += sum_j w[i, j] * (-beta) * d[i, j] / d[i, j]
    # d[i, j] is vector x_i - x_j
    # So grad[i] = -beta * sum_j w[i, j] * (x_i - x_j) / d_ij
    
    # Efficient computation:
    # G[i] = -beta * sum_j w[i, j] * (x_i - x_j) / d_ij
    # Note: w is symmetric, d_ij = -d_ji, w_ij = w_ji
    # So we can compute for i and add to both.
    
    # Vectorized approach:
    # For each i, grad_i = -beta * sum_j (exp(-beta d_ij) / d_ij) * (x_i - x_j)
    # Let A[i, j] = exp(-beta d_ij) / d_ij for i != j, 0 otherwise
    # Then grad = -beta * (A * d) where * is elementwise and sum over j
    
    # To avoid division by zero, we set A to 0 where d is small or diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(mask & (dists > 1e-8), weights / dists, 0.0)
    
    # grad[i] = -beta * sum_j A[i, j] * d[i, j]
    # d[i, j] is (x_i - x_j)
    # sum_j A[i, j] * (x_i - x_j) = x_i * sum_j A[i, j] - sum_j A[i, j] * x_j
    
    sum_A = np.sum(A, axis=1, keepdims=True) # shape (n, 1)
    Ax = np.sum(A[:, :, None] * pts[None, :, :], axis=1) # shape (n, 3)
    
    grad_pair = -beta * (sum_A * pts - Ax)
    
    # Boundary
    # f = sum_i sum_k exp(-beta * (0.5 - sqrt(x_ik^2 + eps^2)))
    # d/dx_ik = exp(...) * (-beta) * d/dx_ik (0.5 - sqrt(...))
    # d/dx_ik (sqrt(x_ik^2 + eps^2)) = x_ik / sqrt(x_ik^2 + eps^2)
    # So d/dx_ik = exp(...) * (-beta) * (-x_ik / sqrt(...)) = beta * exp(...) * x_ik / sqrt(...)
    
    smooth_abs = np.sqrt(pts**2 + eps**2)
    margins = CUBE_HALF - smooth_abs
    exp_m = np.exp(-beta * margins)
    grad_bound = beta * exp_m * (pts / smooth_abs)
    
    return (grad_pair + grad_bound).flatten()

def _refine_exp_repulsion(points, time_limit=60.0):
    """
    Refine using L-BFGS-B with exponential repulsion potential.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points, 0.1)
    
    # Beta controls the "strength" or "range" of the repulsion.
    # Small beta: long-range, smooth.
    # Large beta: short-range, sharp (like hard spheres).
    # We start with small beta to escape local minima, then increase.
    betas = [2.0, 5.0, 10.0, 20.0, 40.0]
    eps = 1e-4
    
    for beta in betas:
        if time.time() - start_t > time_limit:
            break
        
        x0 = best_pts.flatten()
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        
        try:
            res = minimize(
                _exp_repulsion_objective,
                x0,
                args=(beta, eps),
                method='L-BFGS-B',
                jac=_exp_repulsion_grad,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            if res.success or res.fun < _exp_repulsion_objective(x0, beta, eps):
                new_pts = res.x.reshape(-1, 3)
                new_score = _grid_score(new_pts, 0.1)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
        
        if time.time() - start_t > time_limit:
            break
    
    return best_pts, best_score

def _refine_jitter(points, r_init, rng, max_iter=20000, step=0.002, step_decay=0.98, time_limit=60.0):
    """
    Hill-climbing refinement with small random jitter.
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points, r_init)
    step_size = step
    it = 0
    while it < max_iter:
        if time.time() - start_t > time_limit:
            break
        idx = rng.integers(0, len(points))
        delta = rng.uniform(-step_size, step_size, size=3)
        new_pt = best_pts[idx].copy() + delta
        new_pt = np.clip(new_pt, -CUBE_HALF, CUBE_HALF)
        candidate = best_pts.copy()
        candidate[idx] = new_pt
        cand_score = _grid_score(candidate, r_init)
        if cand_score > best_score + 1e-14:
            best_pts = candidate
            best_score = cand_score
            step_size *= step_decay
        else:
            step_size = max(step_size * 0.99, 1e-5)
            if it % 500 == 0:
                step_size = max(step_size * 0.95, 1e-5)
        it += 1
    return best_pts, best_score

def _fcc_lattice():
    """
    Generate FCC lattice points in [-0.5, 0.5]^3 and select 100 with largest min distance to boundary.
    FCC lattice: points at (i/2, j/2, k/2) where i+j+k is even, scaled appropriately.
    """
    # FCC lattice in a cube: generate all points of form (a/2, b/2, c/2) 
    # where a, b, c are integers and a+b+c is even, within range
    points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                if (a + b + c) % 2 == 0:
                    x = a / 2.0
                    y = b / 2.0
                    z = c / 2.0
                    # We want points in [-0.5, 0.5], so scale down
                    # Actually FCC has spacing sqrt(2)/2 * a where a is lattice constant
                    # Let's use a different approach: FCC basis
                    pass
    
    # Better: generate FCC using basis vectors
    # FCC primitive vectors: v1=(0, 0.5, 0.5), v2=(0.5, 0, 0.5), v3=(0.5, 0.5, 0)
    # Lattice point = i*v1 + j*v2 + k*v3
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                x = 0.5 * (j + k)
                y = 0.5 * (i + k)
                z = 0.5 * (i + j)
                # Center: shift so centroid is at origin
                # Mean of i,j,k in [-3,3] is 0, so mean x = 0.5*(0+0)=0
                # But we need to ensure points are in [-0.5, 0.5]
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    points.append((x, y, z))
    
    points = np.array(points)
    
    if len(points) < N:
        # Fallback
        nx, ny, nz = 5, 4, 5
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return points
    
    # Select N points with largest minimum distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_margins = np.min(margins, axis=1)
    idx = np.argsort(-min_margins)[:N]
    return points[idx]

def _cubic_grid():
    """Simple cubic grid 5x4x5 = 100 points."""
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    return np.array([(x, y, z) for z in zs for y in ys for x in xs])

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = []
    
    try:
        fcc = _fcc_lattice()
        if len(fcc) == N:
            configs.append(("fcc", fcc))
    except:
        pass
    
    cubic = _cubic_grid()
    if len(cubic) == N:
        configs.append(("cubic", cubic))
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if time.time() - start_time > DEADLINE_S - 60:
            break
        if len(pts) != N:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts, 0.1)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _cubic_grid()
        best_score = _grid_score(best_pts, 0.1)
    
    # Refinement phase 1: L-BFGS-B with exponential repulsion
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 10 - elapsed
    if remaining > 20:
        lbfgs_time = min(80.0, remaining * 0.6)
        lbfgs_pts, lbfgs_score = _refine_exp_repulsion(best_pts, time_limit=lbfgs_time)
        if lbfgs_score > best_score:
            best_score = lbfgs_score
            best_pts = lbfgs_pts.copy()
    
    # Refinement phase 2: Jitter polish
    elapsed2 = time.time() - start_time
    remaining2 = DEADLINE_S - 10 - elapsed2
    if remaining2 > 5:
        jitter_time = min(40.0, remaining2 * 0.8)
        jitter_pts, jitter_score = _refine_jitter(best_pts, best_score, rng, max_iter=50000, time_limit=jitter_time)
        if jitter_score > best_score:
            best_score = jitter_score
            best_pts = jitter_pts.copy()
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts, best_score)
    
    return best_pts
# EVOLVE-BLOCK-END
