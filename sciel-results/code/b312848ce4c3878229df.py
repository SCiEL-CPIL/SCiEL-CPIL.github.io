# EVOLVE-BLOCK-START
import numpy as np
import time

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
    """Calculate penalty for infeasibility."""
    n = len(sides)
    penalty = 0.0
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return float('inf')
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
                # Overlap penalty
                # If both dx and dy are less than t, there is an overlap.
                # The "depth" of overlap in x is t - dx, in y is t - dy.
                # A simple penalty is the area of the overlapping region.
                overlap_x = t - dx
                overlap_y = t - dy
                penalty += overlap_x * overlap_y
    return penalty

def _objective(centers, sides):
    """Objective function: maximize sum of sides, penalize infeasibility."""
    return _sum_sides(sides) - 1000.0 * _penalty(centers, sides)

def _simulated_annealing(centers, sides, time_limit=10.0):
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
    # bottom-left: [0,0.5] x [0,0.5], center (0.25, 0.25)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5

    # top-right: [0.5,1] x [0.5,1], center (0.75, 0.75)
    centers[1] = [0.75, 0.75]
    sides[1] = 0.5

    # 4 small squares of side 0.25 in top-left quadrant [0,0.5] x [0.5,1]
    # positions: (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875)
    centers[2] = [0.125, 0.625]
    sides[2] = 0.25
    centers[3] = [0.375, 0.625]
    sides[3] = 0.25
    centers[4] = [0.125, 0.875]
    sides[4] = 0.25
    centers[5] = [0.375, 0.875]
    sides[5] = 0.25

    # 4 small squares of side 0.25 in bottom-right quadrant [0.5,1] x [0,0.5]
    # positions: (0.625, 0.125), (0.875, 0.125), (0.625, 0.375), (0.875, 0.375)
    centers[6] = [0.625, 0.125]
    sides[6] = 0.25
    centers[7] = [0.875, 0.125]
    sides[7] = 0.25
    centers[8] = [0.625, 0.375]
    sides[8] = 0.25
    centers[9] = [0.875, 0.375]
    sides[9] = 0.25

    return centers, sides

def _make_3big_7small():
    """
    Try a configuration with 3 large squares.
    One large square of side 0.5 in bottom-left.
    One large square of side 0.5 in top-right.
    One large square of side 0.5 in top-left? No, that leaves bottom-right.
    Let's try:
    Square 0: side 0.5, bottom-left (0.25, 0.25)
    Square 1: side 0.5, top-right (0.75, 0.75)
    Square 2: side 0.5, top-left? No, overlaps with 0 if we put it there.
    
    Let's try a different layout.
    3 squares of side 1/3? No, sum is 1.
    
    How about:
    2 squares of side 0.4.
    8 squares of side 0.3?
    
    Let's just use random initialization for diversity.
    """
    rng = np.random.default_rng(42)
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
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].

    Strategy:
      1. Build the well-known 2-big/8-small configuration (sum = 3.0).
      2. Run Simulated Annealing from that configuration to try to push it higher.
      3. Also try random initializations with SA and keep the best.
      4. Return the best feasible configuration found.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600 s limit

    n = 10

    candidates = []
    
    # --- Candidate 1: 2-big / 8-small (sum = 3.0) ---
    c1, s1 = _make_2big_8small()
    if _check_feasibility(c1, s1):
        c1, s1 = _simulated_annealing(c1, s1, time_limit=10.0)
        if _check_feasibility(c1, s1):
            candidates.append((c1, s1, _sum_sides(s1)))

    # --- Candidate 2: Random start ---
    for i in range(5):
        c_r, s_r = _make_3big_7small()
        # Perturb random start slightly to ensure diversity
        rng = np.random.default_rng(42 + i)
        c_r += rng.normal(0, 0.1, size=(10, 2))
        s_r += rng.normal(0, 0.05, size=10)
        s_r = np.maximum(s_r, 0.1)
        
        c_r, s_r = _simulated_annealing(c_r, s_r, time_limit=10.0)
        if _check_feasibility(c_r, s_r):
            candidates.append((c_r, s_r, _sum_sides(s_r)))

    # Pick the best feasible candidate
    if not candidates:
        # Fallback to the known good one if SA failed all
        c1, s1 = _make_2big_8small()
        return c1, s1

    best_c, best_s, best_val = max(candidates, key=lambda x: x[2])
    return best_c, best_s

# EVOLVE-BLOCK-END
