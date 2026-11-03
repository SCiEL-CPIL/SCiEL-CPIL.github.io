"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Objective and gradient for L-BFGS-B.
    Minimizes the negative of the soft-min of all constraints.
    Constraints: wall clearances and half pairwise distances.
    """
    n = len(x) // 3
    c = x.reshape(n, 3)
    
    # Wall clearances: for each point i and coord k, 0.5 - |c[i,k]|
    # There are n*3 such values
    abs_c = np.abs(c)
    wall_vals = 0.5 - abs_c  # shape (n, 3)
    
    # Pairwise half-distances: for i < j, ||c_i - c_j|| / 2
    # There are n*(n-1)/2 such values
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    # Get upper triangle
    iu = np.triu_indices(n, k=1)
    pair_vals = d[iu] / 2.0  # shape (n*(n-1)/2,)
    
    # Combine all constraint values
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals])
    m = len(all_vals)
    
    # Soft-min: -log(mean(exp(-beta * vals))) / beta
    # = min(vals) - log(mean(exp(-beta*(vals - min)))) / beta
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_shifted = np.exp(-beta * shifted)
    mean_exp = np.mean(exp_shifted)
    soft_min = min_val - np.log(mean_exp) / beta
    
    # Gradient of -soft_min with respect to x
    # d(-soft_min)/dx = -d(soft_min)/dx
    # soft_min = min_val - (1/beta)*log(mean(exp(-beta*(vals-min_val))))
    # Let S = sum(exp(-beta*(vals - min_val)))
    # d(soft_min)/d(vals_j) = -exp(-beta*(vals_j - min_val)) / S
    # (The min_val term contributes 0 to gradient w.r.t. vals since it's piecewise constant)
    
    weights = exp_shifted / (m * mean_exp)  # weights for each constraint value
    
    # Gradient w.r.t. wall values: d(wall_vals[i,k])/dx[i,k] = -sign(c[i,k])
    grad = np.zeros_like(x)
    
    # Wall gradient contribution
    wall_grad = -weights[:n*3].reshape(n, 3) * np.sign(c)
    grad += wall_grad
    
    # Pairwise gradient contribution
    # For each pair (i,j), val = ||c_i - c_j|| / 2
    # d(val)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||) * (1/2) = (c_i - c_j) / (2 * d_ij)
    # Wait: val = d_ij / 2, so d(val)/dc_i = (c_i - c_j) / (2 * d_ij) * 1 ... 
    # Actually: d(||c_i - c_j||/2)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # Let me redo: f = ||c_i - c_j|| / 2
    # df/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # df/dc_j = -(c_i - c_j) / (2 * ||c_i - c_j||)
    
    pair_grad = np.zeros((n, 3))
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        diff = c[i] - c[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-15:
            unit = diff / dist
            w = weights[n*3 + idx]
            pair_grad[i] += w * unit / 2.0
            pair_grad[j] -= w * unit / 2.0
    
    grad += pair_grad
    
    return -soft_min, -grad

def _generate_bcc_lattice(n_points):
    """
    Generate BCC lattice points.
    BCC has 2 atoms per unit cell: corners + body center.
    For a cube of side L, we can place points at:
    - Integer lattice points (corners)
    - Body centers (offset by 0.5 in each dimension)
    
    We scale to fit in [-0.5, 0.5]^3.
    """
    # Determine lattice size
    # BCC has 2 points per unit cell, so for N points we need N/2 cells
    # Each cell is a cube of side a
    # Total volume = (N/2) * a^3 <= 1 (unit cube volume)
    # a <= (2/N)^(1/3)
    
    # We want to pack N points. Use a grid of m x m x m cells, each with 2 points
    # Total points = 2 * m^3
    # Find m such that 2*m^3 >= N, or just use m = ceil((N/2)^(1/3))
    
    m = int(np.ceil((N / 2.0) ** (1.0 / 3.0)))
    
    # Generate all BCC points in a m x m x m grid of cells
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner
                points.append([ix, iy, iz])
                # Body center
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
    
    points = np.array(points, dtype=float)
    
    # Scale and center to fit in [-0.5, 0.5]^3
    # Current range: [0, m-1] in each dimension (corners) or [0.5, m-0.5] (body centers)
    # Max coordinate: m-1 (for corners at the edge)
    # We need to scale so that max coord maps to 0.5
    # Actually, let's compute the bounding box
    
    if len(points) == 0:
        return np.zeros((N, 3))
    
    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)
    span = max_coord - min_coord
    
    if np.any(span < 1e-10):
        # Degenerate case
        return np.zeros((N, 3))
    
    # Scale to fit in [-0.45, 0.45] with some margin
    target_span = 0.9
    scale = target_span / np.max(span)
    points = (points - min_coord) * scale
    points = points - 0.45  # Center
    
    # Select N points (take the first N, or if fewer, pad with random)
    if len(points) >= N:
        # Shuffle and take N
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(points))
        points = points[perm[:N]]
    else:
        # Pad with random points
        rng = np.random.default_rng(42)
        remaining = N - len(points)
        pad = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        points = np.vstack([points, pad])
    
    return points

def _generate_fcc_lattice(n_points):
    """
    Generate FCC lattice points.
    FCC has 4 atoms per unit cell: corners + face centers.
    """
    m = int(np.ceil((n_points / 4.0) ** (1.0 / 3.0)))
    
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner
                points.append([ix, iy, iz])
                # Face centers
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
    
    points = np.array(points, dtype=float)
    
    if len(points) == 0:
        return np.zeros((N, 3))
    
    min_coord = np.min(points, axis=0)
    max_coord = np.max(points, axis=0)
    span = max_coord - min_coord
    
    if np.any(span < 1e-10):
        return np.zeros((N, 3))
    
    target_span = 0.9
    scale = target_span / np.max(span)
    points = (points - min_coord) * scale
    points = points - 0.45
    
    if len(points) >= N:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(points))
        points = points[perm[:N]]
    else:
        rng = np.random.default_rng(42)
        remaining = N - len(points)
        pad = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        points = np.vstack([points, pad])
    
    return points

def _optimize_from_initial(c0, max_iter=2000, tol=1e-10, beta_start=20.0, beta_end=200.0):
    """
    Multi-stage annealing with increasing beta.
    """
    n = len(c0)
    c = np.array(c0, dtype=float).copy()
    best_c = c.copy()
    best_val = _compute_radius(c)
    
    # Annealing schedule
    n_stages = 8
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        # Define objective and gradient
        def obj_grad(x):
            return _soft_min_objective_and_grad(x, beta)
        
        # L-BFGS-B
        try:
            res = minimize(
                obj_grad,
                c.ravel(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
            )
            c = res.x.reshape(n, 3)
            c = np.clip(c, -0.499, 0.499)
            
            val = _compute_radius(c)
            if val > best_val:
                best_val = val
                best_c = c.copy()
        except Exception:
            pass
    
    return best_c, best_val

def _local_polish(c, max_iter=1000):
    """
    Final polish with very high beta.
    """
    n = len(c)
    
    def obj_grad(x):
        return _soft_min_objective_and_grad(x, 500.0)
    
    try:
        res = minimize(
            obj_grad,
            c.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        c_new = res.x.reshape(n, 3)
        c_new = np.clip(c_new, -0.499, 0.499)
        return c_new
    except Exception:
        return c

def construct():
    """
    Construct 50 sphere centres in the unit cube [-0.5, 0.5]^3.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try multiple initializations
    initializations = []
    
    # BCC lattice
    bcc = _generate_bcc_lattice(N)
    initializations.append(bcc)
    
    # FCC lattice
    fcc = _generate_fcc_lattice(N)
    initializations.append(fcc)
    
    # Random with repulsion
    rng = np.random.default_rng(123)
    for seed in range(3):
        c_rand = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(c_rand)
    
    # Grid-based
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        initializations.append(grid[:N])
    
    best_c = None
    best_val = 0.0
    
    for c0 in initializations:
        if time.time() > deadline - 30:
            break
        
        c, val = _optimize_from_initial(c0, max_iter=500, tol=1e-9, beta_start=20.0, beta_end=100.0)
        
        if val > best_val:
            best_val = val
            best_c = c.copy()
    
    # Final polish
    if best_c is not None and time.time() < deadline - 10:
        best_c = _local_polish(best_c, max_iter=500)
        best_val = _compute_radius(best_c)
    
    # Ensure valid output
    if best_c is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    best_c = np.clip(best_c, -0.5, 0.5)
    best_c = np.asarray(best_c, dtype=float)
    
    # Verify shape
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
