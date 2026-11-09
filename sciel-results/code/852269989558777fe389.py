"""Bootstrap direct construction for 100 equal spheres in a unit cube."""
N = 100
CUBE_HALF = 0.5
DEADLINE_S = 115.0  # safety margin before 120 s evaluator kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def _cubic_grid(m):
    """Return (m**3, 3) array of cubic grid points in [-0.5, 0.5]^3."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid

def _grid_score(points, r):
    """Compute the largest radius that fits for the given points."""
    if len(points) < 2:
        return 0.5
    # min pairwise distance
    # Use broadcasting for N=100
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    return min(min_pair, min_boundary) / 2.0

def _refine_jitter(points, r_init, rng, max_iter=20000, step=0.002, step_decay=0.98):
    """
    Hill-climbing refinement with small random jitter.
    Returns best points found and its score.
    """
    best_pts = points.copy()
    best_score = _grid_score(points, r_init)
    step_size = step
    it = 0
    while it < max_iter:
        # Propose a small perturbation to a random point
        idx = rng.integers(0, len(points))
        delta = rng.uniform(-step_size, step_size, size=3)
        new_pt = best_pts[idx].copy() + delta
        # Ensure inside cube
        new_pt = np.clip(new_pt, -CUBE_HALF, CUBE_HALF)
        candidate = best_pts.copy()
        candidate[idx] = new_pt
        cand_score = _grid_score(candidate, r_init)
        if cand_score > best_score + 1e-14:
            best_pts = candidate
            best_score = cand_score
            it += 1
            step_size *= step_decay
        else:
            it += 1
            step_size = max(step_size * 0.99, 1e-5)
            if it % 500 == 0:
                step_size = max(step_size * 0.95, 1e-5)
    return best_pts, best_score

def _body_centered_lattice():
    """
    Construct a BCC-like arrangement for 100 points in a unit cube.
    We place points in a 5x4x5 cubic grid with offsets to approximate BCC.
    """
    # Use a 5x4x5 grid = 100 points
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _fcc_lattice():
    """
    Approximate FCC packing in a cube.
    FCC in a cubic cell has 4 atoms per cell.
    We use a 4x4x4 cubic cell arrangement with FCC offsets.
    """
    # FCC positions in a unit cell (0,0) to (1,1):
    # corners: (0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)
    # face centers: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (0.5,0.5,1), (0.5,1,0.5), (1,0.5,0.5)
    # But that's 14 per cell. We need a simpler approach.
    # Use a 4x4x4 grid of cubic cells, each with a central point and corner sharing.
    # Simpler: use a 5x5x4 grid with alternating offsets.
    
    # Actually, let's use a simple body-centered arrangement:
    # Place points on a cubic grid with half-cell offsets on alternating layers.
    m = 4  # 4x4x4 = 64, not enough
    # Try a different approach: 100 = 4*5*5
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.5, 0.5, nx, endpoint=True)
    ys = np.linspace(-0.5, 0.5, ny, endpoint=True)
    zs = np.linspace(-0.5, 0.5, nz, endpoint=True)
    
    # Shift alternating layers for better packing
    points = []
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                # Shift based on parity to create FCC-like structure
                offset = 0.5 / nx if (i + j + k) % 2 == 0 else 0.0
                px = x + offset
                py = y + offset
                pz = z + offset
                # Keep within bounds
                px = max(-0.5, min(0.5, px))
                py = max(-0.5, min(0.5, py))
                pz = max(-0.5, min(0.5, pz))
                points.append((px, py, pz))
    return np.array(points[:100])

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate for the minimum of pairwise distances and boundary distances.
    Uses log-sum-exp to approximate the minimum.
    """
    pts = points
    n = len(pts)
    
    # Pairwise squared distances
    d = pts[:, None, :] - pts[None, :, :]
    sq_dists = np.sum(d**2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Boundary distances (squared)
    margins = CUBE_HALF - np.abs(pts)
    boundary_sq = np.sum(margins**2, axis=-1)
    
    # We want min of sqrt(sq_dists) and min of boundary margins
    # Use soft min via log-sum-exp on negative values
    # For pairwise: soft_min(sqrt(sq_dists)) ≈ -log(sum(exp(-beta*sqrt(sq_dists))))/beta
    # But sqrt is expensive. Instead work with squared distances and scale beta accordingly.
    # Actually, let's work with the actual distances for correctness.
    
    # Compute actual distances
    dists = np.sqrt(np.maximum(sq_dists, 0))
    np.fill_diagonal(dists, np.inf)
    
    # Soft min of all pairwise distances and boundary margins
    # Combine all constraints into one array
    all_constraints = np.concatenate([dists.flatten(), margins.flatten()])
    # Remove infinities
    all_constraints = all_constraints[np.isfinite(all_constraints)]
    
    if len(all_constraints) == 0:
        return 0.5
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability, subtract max
    max_c = np.max(all_constraints)
    shifted = all_constraints - max_c
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = max_c - log_sum / beta
    
    return soft_min / 2.0

def _soft_min_grad(points, beta=50.0):
    """
    Gradient of the soft-min score with respect to points.
    """
    pts = points
    n = len(pts)
    
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    sq_dists = np.sum(d**2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    dists = np.sqrt(np.maximum(sq_dists, 0))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary margins
    margins = CUBE_HALF - np.abs(pts)
    
    # Combine all constraints
    all_constraints = np.concatenate([dists.flatten(), margins.flatten()])
    all_constraints = all_constraints[np.isfinite(all_constraints)]
    
    if len(all_constraints) == 0:
        return np.zeros_like(pts)
    
    max_c = np.max(all_constraints)
    shifted = all_constraints - max_c
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Weights for soft min: w_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    weights = exp_vals / sum_exp
    
    # Gradient of soft_min = -sum(w_i * grad(x_i))
    # For pairwise distance |c_i - c_j|:
    #   grad wrt c_i = (c_i - c_j) / |c_i - c_j|
    #   grad wrt c_j = (c_j - c_i) / |c_i - c_j|
    # For boundary margin |c_ik|:
    #   grad wrt c_ik = sign(c_ik) if c_ik != 0
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    # dists is n x n, weights for pairwise are the first n*(n-1)/2 * 2 elements? 
    # Actually, dists has n*n elements with diagonal inf. Let's handle it properly.
    # The flat pairwise part has n*n elements, but diagonal is inf so weight is 0.
    pair_weights = weights[:n*n].reshape(n, n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                w = pair_weights[i, j]
                if w > 1e-15:
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        grad[i] += w * diff / dist
                        grad[j] -= w * diff / dist
    
    # Boundary contributions
    # margins is n x 3, flat part is after n*n elements
    boundary_weights = weights[n*n:].reshape(n, 3)
    
    for i in range(n):
        for k in range(3):
            w = boundary_weights[i, k]
            if w > 1e-15:
                # margin = 0.5 - |c_ik|
                # d(margin)/d(c_ik) = -sign(c_ik)
                c_ik = pts[i, k]
                if abs(c_ik) > 1e-12:
                    grad[i, k] += w * (-np.sign(c_ik))
    
    # The soft_min score is soft_min / 2, so gradient is grad / 2
    # But we want to maximize, so we return -grad for minimization
    return -grad / 2.0

def _lbfgs_refine(points, beta=50.0, max_iter=500, deadline_check=None):
    """
    Refine points using L-BFGS on the soft-min surrogate.
    """
    from scipy.optimize import minimize
    
    x0 = points.flatten()
    n = len(points)
    
    def objective(x):
        pts = x.reshape(n, 3)
        pts = np.clip(pts, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        return -_soft_min_score(pts, beta)
    
    def gradient(x):
        pts = x.reshape(n, 3)
        pts = np.clip(pts, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        g = _soft_min_grad(pts, beta)
        return g.flatten()
    
    start_time = time.time()
    
    def callback(xk):
        if deadline_check and deadline_check():
            return True  # Stop optimization
        if time.time() - start_time > 80:
            return True
        return False
    
    res = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (n * 3),
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'callback': callback}
    )
    
    refined_pts = res.x.reshape(n, 3)
    refined_pts = np.clip(refined_pts, -CUBE_HALF, CUBE_HALF)
    refined_score = _grid_score(refined_pts, 0.1)
    
    return refined_pts, refined_score

def construct():
    start_time = time.time()
    
    def deadline_exceeded():
        return time.time() - start_time > DEADLINE_S - 5
    
    # Strategy 1: Body-centered lattice (4x5x5 grid with offsets)
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    configs = [
        ("cubic_5x4x5", _body_centered_lattice()),
    ]
    
    # Also try the original simple cubic grid
    m5 = 5
    ticks5 = (np.arange(m5) + 0.5) / m5 - 0.5
    grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
    configs.append(("cubic_5x5x4", grid5[:100]))
    
    # FCC-like
    try:
        fcc = _fcc_lattice()
        if len(fcc) >= 100:
            configs.append(("fcc_like", fcc[:100]))
    except:
        pass
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if deadline_exceeded():
            break
        if len(pts) != 100:
            continue
        # Ensure valid
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts, 0.1)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _body_centered_lattice()
        best_score = _grid_score(best_pts, 0.1)
    
    # Refine with L-BFGS on soft-min surrogate
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 15 - elapsed
    
    if remaining > 5:
        # Multiple restarts with different beta values
        betas = [30.0, 50.0, 100.0]
        n_restarts = max(1, int(remaining / 20))  # ~20s per refinement
        
        for restart in range(min(n_restarts, 6)):
            if deadline_exceeded():
                break
            
            beta = betas[restart % len(betas)]
            
            if restart == 0:
                pts_to_refine = best_pts.copy()
            else:
                # Perturb the best solution
                perturbation_scale = 0.005 + 0.01 * restart
                pts_to_refine = best_pts + rng.normal(0, perturbation_scale, size=(100, 3))
                pts_to_refine = np.clip(pts_to_refine, -0.5, 0.5)
            
            max_iter = max(100, int(remaining / 0.02))
            refined_pts, refined_score = _lbfgs_refine(
                pts_to_refine, 
                beta=beta,
                max_iter=max_iter,
                deadline_check=deadline_exceeded
            )
            
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
