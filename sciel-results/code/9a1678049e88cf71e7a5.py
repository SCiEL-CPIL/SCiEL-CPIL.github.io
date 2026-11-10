"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist

def _compute_score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    if n < 2:
        return 0.5 - np.max(np.abs(pts))
    
    # Min distance between any two centers
    # Using scipy for speed
    dmin = np.min(pdist(pts))
    
    # Min distance to walls
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)

def _project_to_constraint(pts, R, max_iter=1000, tol=1e-9):
    """
    Iteratively project points to satisfy:
    1. All points are inside [-0.5+R, 0.5-R]^3
    2. All pairwise distances >= 2R
    
    This is a simplified version of the KKT conditions for packing.
    We use a Gauss-Seidel style relaxation where we push overlapping pairs apart
    and clip to the box.
    """
    pts = pts.copy()
    n = len(pts)
    min_dist = 2 * R
    
    for _ in range(max_iter):
        changed = False
        
        # 1. Clip to box
        old_pts = pts.copy()
        pts = np.clip(pts, -0.5 + R, 0.5 - R)
        if not np.allclose(pts, old_pts, atol=tol):
            changed = True
            
        # 2. Resolve overlaps
        # To make this efficient, we only check pairs that are close.
        # For N=50, O(N^2) is 2500 checks, which is fast enough.
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < min_dist:
                    if d < 1e-10:
                        # Random jitter if coincident
                        d_vec = np.random.randn(3)
                        d = np.linalg.norm(d_vec)
                    
                    # Push apart by half the overlap
                    overlap = min_dist - d
                    correction = (overlap / 2.0) * (d_vec / d)
                    
                    pts[i] += correction
                    pts[j] -= correction
                    changed = True
                    
        if not changed:
            break
            
    return pts

def _check_feasibility(pts, R, tol=1e-6):
    """Check if configuration is feasible for radius R."""
    # Check walls
    if np.any(pts < -0.5 + R - tol) or np.any(pts > 0.5 - R - tol):
        return False
    # Check pairs
    dmin = np.min(pdist(pts))
    if dmin < 2 * R - tol:
        return False
    return True

def _solve_packing_R(initial_pts, R):
    """
    Try to find a valid packing for radius R starting from initial_pts.
    Returns (success, optimized_pts).
    """
    # We run the projection multiple times with random perturbations to escape local minima
    best_pts = initial_pts.copy()
    
    # Initial projection
    pts = _project_to_constraint(best_pts, R, max_iter=500)
    
    if _check_feasibility(pts, R):
        return True, pts
    
    # If it failed, try a few random restarts/perturbations
    for _ in range(10):
        # Perturb current best
        perturbed = best_pts + np.random.randn(N, 3) * 0.05
        perturbed = np.clip(perturbed, -0.45, 0.45)
        
        pts = _project_to_constraint(perturbed, R, max_iter=500)
        
        if _check_feasibility(pts, R):
            return True, pts
        else:
            # Keep the one with best score if we fail? 
            # Actually, if we fail feasibility, we can't use it for this R.
            # But maybe a slightly smaller R works?
            # We just return False for this R.
            pass
            
    return False, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate a good initial configuration
    # A 4x4x4 grid (64 points) is a good start. We take 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to avoid bias, then take first 50
    np.random.shuffle(grid)
    init_pts = grid[:N].copy()
    
    # We can also try a random init
    # init_pts_random = np.random.uniform(-0.4, 0.4, (N, 3))
    
    best_score = _compute_score(init_pts)
    best_pts = init_pts.copy()
    
    # Bisection search for R
    # Lower bound: current best score
    # Upper bound: theoretical max? 
    # Volume argument: 50 * (4/3)pi R^3 <= 1 => R <= (3/(400*pi))^(1/3) ~ 0.105 (too low, packing fraction < 1)
    # Actually, packing fraction for FCC is ~0.74. 
    # 50 * (4/3)pi R^3 <= 0.74 => R <= (0.74 * 3 / (200 * pi))^(1/3) ~ 0.138
    # So 0.14 is a reasonable upper bound.
    
    R_low = best_score
    R_high = 0.145
    
    # First, try to improve R_low using the current best points with a small increase
    # Actually, let's just bisect.
    
    # To save time, we limit the number of bisection steps.
    # Each step involves projection which is O(N^2 * Iter).
    # N=50, N^2=2500. Iter=500. Ops ~ 1.25M per projection.
    # 10 restarts per projection = 12.5M ops.
    # 15 bisection steps = 187M ops.
    # This might be tight in Python. Let's be careful.
    
    # We will use a fixed number of bisection steps, checking time.
    
    for step in range(20):
        if time.time() > deadline - 10:
            break
            
        R_mid = (R_low + R_high) / 2.0
        
        # Try to find a packing for R_mid
        success, pts_candidate = _solve_packing_R(init_pts, R_mid)
        
        if success:
            # We found a valid packing for R_mid
            # Calculate actual score (might be slightly higher due to tolerance)
            score_candidate = _compute_score(pts_candidate)
            if score_candidate > best_score:
                best_score = score_candidate
                best_pts = pts_candidate.copy()
            
            R_low = R_mid
        else:
            R_high = R_mid
            
        # If the interval is small enough, stop
        if R_high - R_low < 1e-6:
            break
            
    # Final check: maybe we can squeeze a bit more from best_pts?
    # Run a local optimization on best_pts to see if we can increase the min distance
    # This is a "polish" step.
    
    # Simple polish: run projection with R = best_score * 1.001 and see if it works
    # If it works, update best_score.
    
    # Let's try to increase R slightly from best_score
    R_try = best_score * 1.005
    if time.time() < deadline - 5:
        success, pts_polished = _solve_packing_R(best_pts, R_try)
        if success:
            score_polished = _compute_score(pts_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
