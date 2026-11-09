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

def _soft_min_objective(x, beta):
    """
    Negative soft-min radius objective for L-BFGS-B minimization.
    x is a flat array of shape (300,) representing 100 points in 3D.
    Returns negative of the soft-min radius (we minimize, so negate).
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Soft-min of pairwise distances
    min_d = np.min(dists)
    exp_d = np.exp(-(dists - min_d) / beta)
    exp_d = np.where(mask, exp_d, 0.0)
    soft_min_d = min_d - (1.0 / beta) * np.log(np.sum(exp_d))
    
    # Boundary distances (margins)
    margins = CUBE_HALF - np.abs(pts)
    min_m = np.min(margins)
    exp_m = np.exp(-(margins - min_m) / beta)
    soft_min_m = min_m - (1.0 / beta) * np.log(np.sum(exp_m))
    
    # Soft-min of the two values
    v1, v2 = soft_min_d, soft_min_m
    m = min(v1, v2)
    soft_min_final = m - (1.0 / beta) * np.log(np.exp(-(v1 - m) / beta) + np.exp(-(v2 - m) / beta))
    
    # We want to maximize the radius, so minimize the negative
    return -soft_min_final / 2.0

def _soft_min_grad(x, beta):
    """
    Analytical gradient of the soft-min objective with respect to x.
    """
    pts = x.reshape(-1, 3)
    n = len(pts)
    
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dists_sq = np.sum(d**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dists_sq = np.where(mask, dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Soft-min of pairwise distances
    min_d = np.min(dists)
    exp_d = np.exp(-(dists - min_d) / beta)
    exp_d = np.where(mask, exp_d, 0.0)
    sum_exp_d = np.sum(exp_d)
    soft_min_d = min_d - (1.0 / beta) * np.log(sum_exp_d)
    
    # Gradient of soft_min_d w.r.t. dists
    # d(soft_min_d)/d(dists_ij) = (exp_d_ij / sum_exp_d) for each pair
    # But we need d(soft_min_d)/d(pts)
    # dists_ij = |pts_i - pts_j|
    # d(dists_ij)/d(pts_i) = (pts_i - pts_j) / dists_ij
    # d(dists_ij)/d(pts_j) = (pts_j - pts_i) / dists_ij
    
    # For each pair (i,j), contribution to grad:
    # grad_pts_i += (exp_d_ij / sum_exp_d) * (pts_i - pts_j) / dists_ij
    # grad_pts_j += (exp_d_ij / sum_exp_d) * (pts_j - pts_i) / dists_ij
    
    grad_d = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            w = exp_d[i, j] / sum_exp_d
            if dists[i, j] > 1e-10:
                diff = pts[i] - pts[j]
                grad_d[i] += w * diff / dists[i, j]
                grad_d[j] += w * (pts[j] - pts[i]) / dists[i, j]
    
    # Boundary distances
    margins = CUBE_HALF - np.abs(pts)
    min_m = np.min(margins)
    exp_m = np.exp(-(margins - min_m) / beta)
    sum_exp_m = np.sum(exp_m)
    soft_min_m = min_m - (1.0 / beta) * np.log(sum_exp_m)
    
    # Gradient of soft_min_m w.r.t. margins
    # d(soft_min_m)/d(margin_ik) = exp_m_ik / sum_exp_m
    # margin_ik = CUBE_HALF - |pts_ik|
    # d(margin_ik)/d(pts_ik) = -sign(pts_ik)
    grad_m = np.zeros_like(pts)
    for i in range(n):
        for k in range(3):
            w = exp_m[i, k] / sum_exp_m
            grad_m[i, k] = -w * np.sign(pts[i, k])
    
    # Combine: soft_min_final = soft_min(soft_min_d, soft_min_m)
    v1, v2 = soft_min_d, soft_min_m
    m = min(v1, v2)
    e1 = np.exp(-(v1 - m) / beta)
    e2 = np.exp(-(v2 - m) / beta)
    sum_e = e1 + e2
    
    # d(soft_min_final)/d(v1) = e1/sum_e
    # d(soft_min_final)/d(v2) = e2/sum_e
    w1 = e1 / sum_e
    w2 = e2 / sum_e
    
    # Total gradient: -1/2 * (w1 * grad_d + w2 * grad_m)
    # But grad_d is d(soft_min_d)/d(pts) and grad_m is d(soft_min_m)/d(pts)
    # Actually we need to be careful with signs.
    # soft_min_final = soft_min(soft_min_d, soft_min_m)
    # d(soft_min_final)/d(pts) = w1 * d(soft_min_d)/d(pts) + w2 * d(soft_min_m)/d(pts)
    # = w1 * grad_d + w2 * grad_m
    # Objective = -soft_min_final / 2
    # d(objective)/d(pts) = -1/2 * (w1 * grad_d + w2 * grad_m)
    
    grad_total = -0.5 * (w1 * grad_d + w2 * grad_m)
    return grad_total.flatten()

def _lbfgs_refine(points, time_limit=60.0):
    """
    Refine using L-BFGS-B with progressively increasing beta (sharpening soft-min).
    """
    start_t = time.time()
    best_pts = points.copy()
    best_score = _grid_score(points, 0.1)
    
    # Start with larger beta (smoother), then decrease
    betas = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    
    for beta in betas:
        if time.time() - start_t > time_limit:
            break
        
        x0 = best_pts.flatten()
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=_soft_min_grad,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            if res.success or res.fun < _soft_min_objective(x0, beta):
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
    
    # Refinement phase 1: L-BFGS-B with soft-min
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 10 - elapsed
    if remaining > 20:
        lbfgs_time = min(80.0, remaining * 0.6)
        lbfgs_pts, lbfgs_score = _lbfgs_refine(best_pts, time_limit=lbfgs_time)
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
