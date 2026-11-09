# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_feasibility(centers, sides, tol=1e-7):
    """Check if the configuration is feasible (all inside unit square, no overlaps)."""
    n = len(sides)
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    # Check no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _sum_sides(sides):
    return float(np.sum(sides))

def _penalty(centers, sides, tol=1e-7):
    """Calculate smooth penalty for infeasibility."""
    n = len(sides)
    penalty = 0.0
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return float('inf')
        # Boundary violations
        if cx - s / 2 < -tol:
            penalty += (-tol - (cx - s / 2)) ** 2
        if cx + s / 2 > 1 + tol:
            penalty += ((cx + s / 2) - (1 + tol)) ** 2
        if cy - s / 2 < -tol:
            penalty += (-tol - (cy - s / 2)) ** 2
        if cy + s / 2 > 1 + tol:
            penalty += ((cy + s / 2) - (1 + tol)) ** 2
    # Check no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                # Smooth penalty for overlap
                overlap_x = t - dx
                overlap_y = t - dy
                penalty += overlap_x * overlap_y
    return penalty

def _objective(centers, sides):
    """Objective function: maximize sum of sides, penalize infeasibility."""
    return _sum_sides(sides) - 1000.0 * _penalty(centers, sides)

def _flatten(centers, sides):
    n = len(sides)
    return np.concatenate([centers.flatten(), sides])

def _unflatten(x):
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return centers, sides

def _objective_flat(x):
    centers, sides = _unflatten(x)
    return -_objective(centers, sides) # Minimize negative

def _simulated_annealing(centers, sides, time_limit=5.0):
    """
    Simulated Annealing to maximize sum of side lengths.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_obj = _objective(current_centers, current_sides)
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    T = 0.1
    T_min = 1e-6
    alpha = 0.9995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        dc = rng.normal(0, 0.05, size=(n, 2))
        ds = rng.normal(0, 0.02, size=n)
        
        new_centers = current_centers + dc
        new_sides = current_sides + ds
        
        # Ensure sides are positive
        new_sides = np.maximum(new_sides, 1e-6)
        
        new_obj = _objective(new_centers, new_sides)
        
        delta_obj = new_obj - current_obj
        
        if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
            current_centers = new_centers
            current_sides = new_sides
            current_obj = new_obj
            
            if current_obj > best_obj:
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                best_obj = current_obj
                
        T *= alpha
        
    return best_centers, best_sides

def _nelder_mead_refine(centers, sides, time_limit=10.0):
    """
    Refine using Nelder-Mead on the flattened vector.
    """
    x0 = _flatten(centers, sides)
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective_flat(x)
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        return _unflatten(result.x)
    except Exception:
        return centers, sides

def _make_2big_8small():
    """
    The classic optimal (or near-optimal) configuration for 10 squares:
    2 squares of side 0.5 (at bottom-left and top-right quadrants),
    4 squares of side 0.25 filling the top-left quadrant,
    4 squares of side 0.25 filling the bottom-right quadrant.
    Sum = 2*0.5 + 8*0.25 = 3.0
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)

    # 2 large squares of side 0.5
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5

    centers[1] = [0.75, 0.75]
    sides[1] = 0.5

    # 4 small squares of side 0.25 in top-left quadrant
    centers[2] = [0.125, 0.625]
    sides[2] = 0.25
    centers[3] = [0.375, 0.625]
    sides[3] = 0.25
    centers[4] = [0.125, 0.875]
    sides[4] = 0.25
    centers[5] = [0.375, 0.875]
    sides[5] = 0.25

    # 4 small squares of side 0.25 in bottom-right quadrant
    centers[6] = [0.625, 0.125]
    sides[6] = 0.25
    centers[7] = [0.875, 0.125]
    sides[7] = 0.25
    centers[8] = [0.625, 0.375]
    sides[8] = 0.25
    centers[9] = [0.875, 0.375]
    sides[9] = 0.25

    return centers, sides

def _make_random(seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(10, 2))
    sides = rng.uniform(0.1, 0.4, size=10)
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n=10 axis-aligned squares with
    pairwise disjoint interiors inside the unit square [0,1]^2 so as to
    MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600 s limit

    n = 10
    candidates = []
    
    # --- Candidate 1: 2-big / 8-small (sum = 3.0) ---
    c1, s1 = _make_2big_8small()
    if _check_feasibility(c1, s1):
        # Refine with SA then NM
        c1, s1 = _simulated_annealing(c1, s1, time_limit=5.0)
        c1, s1 = _nelder_mead_refine(c1, s1, time_limit=10.0)
        if _check_feasibility(c1, s1):
            candidates.append((c1, s1, _sum_sides(s1)))

    # --- Candidate 2: Random starts ---
    for i in range(5):
        if time.time() > deadline:
            break
        c_r, s_r = _make_random(42 + i)
        c_r, s_r = _simulated_annealing(c_r, s_r, time_limit=5.0)
        c_r, s_r = _nelder_mead_refine(c_r, s_r, time_limit=10.0)
        if _check_feasibility(c_r, s_r):
            candidates.append((c_r, s_r, _sum_sides(s_r)))

    # Pick the best feasible candidate
    if not candidates:
        # Fallback to the known good one
        c1, s1 = _make_2big_8small()
        return c1, s1

    best_c, best_s, best_val = max(candidates, key=lambda x: x[2])
    return best_c, best_s

# EVOLVE-BLOCK-END
