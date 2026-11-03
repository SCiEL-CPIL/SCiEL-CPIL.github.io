"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _radius(c):
    """Compute the score for a set of centres: min of half pairwise distances and wall clearances."""
    # Wall clearance: min over all coordinates of (0.5 - |coord|)
    wall = 0.5 - np.max(np.abs(c))
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(wall, min_pair)

def _optimize_local(c0, max_iter=500, tol=1e-10):
    """
    Local optimization using a smooth surrogate.
    We maximize the minimum of (pairwise distances/2, wall clearances).
    Use a log-sum-exp style soft-min to get a smooth objective.
    """
    c0 = np.array(c0, dtype=float)
    n = len(c0)
    
    def objective(x):
        """Negative of soft-min radius (we minimize negative radius)."""
        c = x.reshape(n, 3)
        # Wall clearances
        wall = 0.5 - np.abs(c)
        # Pairwise distances
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        d_half = d / 2.0
        
        # Combine all constraints into one array
        vals = np.concatenate([wall.ravel(), d_half.ravel()])
        # Remove infinities
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 0.0
        # Soft-min with small beta for smoothness
        beta = 50.0
        m = np.min(vals)
        soft_min = m - np.log(np.mean(np.exp(-(vals - m) * beta))) / beta
        return -soft_min
    
    # Initial evaluation
    best_c = c0.copy()
    best_val = _radius(best_c)
    
    # Try a few random perturbations to escape local minima
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 10
    
    while attempts < max_attempts:
        attempts += 1
        # Start from current best or a perturbation
        if attempts == 1:
            start = c0.copy()
        else:
            start = best_c + rng.normal(0, 0.01, size=(n, 3))
            # Keep within bounds
            start = np.clip(start, -0.49, 0.49)
        
        try:
            res = minimize(objective, start.ravel(), method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
            new_c = res.x.reshape(n, 3)
            # Clip to ensure validity
            new_c = np.clip(new_c, -0.5, 0.5)
            new_val = _radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass
    
    return best_c, best_val

def construct():
    """
    Construct 50 sphere centres in the unit cube [-0.5, 0.5]^3.
    Uses a grid-based initial placement followed by local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Step 1: Create a fine grid of candidate positions
    # Use a 5x5x5 grid as starting point (125 points), then select 50
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Step 2: Greedy selection with random tie-breaking
    # We want to select 50 points that are well-spread
    rng = np.random.default_rng(123)
    
    # Shuffle grid
    indices = np.arange(len(grid))
    rng.shuffle(indices)
    grid_shuffled = grid[indices]
    
    # Greedy: select points that maximize minimum distance to already selected
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if time.time() > deadline - 10:
            break
        if len(selected) >= N:
            break
        point = grid_shuffled[i]
        # Check minimum distance to already selected
        dists = np.linalg.norm(np.array(selected) - point, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:  # Threshold to avoid too-close points
            selected.append(point)
    
    # If we don't have 50, add more from the remaining grid
    if len(selected) < N:
        remaining = [grid_shuffled[i] for i in range(len(grid_shuffled)) 
                     if not any(np.allclose(grid_shuffled[i], s) for s in selected)]
        for point in remaining:
            if len(selected) >= N:
                break
            selected.append(point)
    
    # If still not enough, generate random points
    while len(selected) < N:
        if time.time() > deadline - 5:
            break
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)
    
    c = np.array(selected[:N])
    
    # Step 3: Local optimization
    if time.time() < deadline - 30:
        c, val = _optimize_local(c, max_iter=300, tol=1e-9)
    
    # Step 4: Final polish with tighter tolerance
    if time.time() < deadline - 10:
        c, val = _optimize_local(c, max_iter=1000, tol=1e-12)
    
    # Ensure valid output
    c = np.clip(c, -0.5, 0.5)
    c = np.asarray(c, dtype=float)
    
    # Verify shape
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        c = grid[:N]
    
    return c
# EVOLVE-BLOCK-END
