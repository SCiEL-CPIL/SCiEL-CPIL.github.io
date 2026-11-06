import time
import numpy as np
import random

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 75.0

# Precompute triple indices for 30 points
_N = 30
_I, _J, _K = [], [], []
for i in range(_N):
    for j in range(i + 1, _N):
        for k in range(j + 1, _N):
            _I.append(i)
            _J.append(j)
            _K.append(k)
_I = np.array(_I, dtype=np.int32)
_J = np.array(_J, dtype=np.int32)
_K = np.array(_K, dtype=np.int32)
_NUM_TRIPLES = len(_I)

def _compute_areas(pts):
    """
    Compute doubled absolute areas for all triples.
    pts: numpy array of shape (30, 2)
    Returns: numpy array of shape (4060,) with doubled areas.
    """
    # pts[I], pts[J], pts[K]
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    
    # Vectors
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    # Determinant
    det = v1x * v2y - v1y * v2x
    return np.abs(det)

def _score(pts):
    """
    Calculate the maximum multiplicity of a single positive doubled area.
    """
    areas = _compute_areas(pts)
    pos_areas = areas[areas > 0]
    if len(pos_areas) == 0:
        return 0
    vals, counts = np.unique(pos_areas, return_counts=True)
    return int(counts.max())

def _generate_initial_grid():
    """
    Generate a 6x5 grid centered at (0,0).
    """
    points = []
    # 6 wide, 5 high
    for x in range(-2, 4):
        for y in range(-2, 3):
            points.append((x, y))
    return np.array(points, dtype=np.int64)

def _hill_climb_sa(pts, deadline, seed=0):
    """
    Simulated Annealing refinement.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    cur_pts = pts.copy()
    cur_score = _score(cur_pts)
    best_pts = cur_pts.copy()
    best_score = cur_score
    
    # Temperature parameters
    T_start = 2.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    
    while time.time() < deadline:
        # Choose a random point to move
        idx = rng.randint(0, _N - 1)
        ox, oy = cur_pts[idx]
        
        # Generate a new position
        # Strategy: try nearby moves mostly, occasionally jump
        if rng.random() < 0.8:
            # Local move
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = ox + dx, oy + dy
        else:
            # Global jump
            nx = rng.randint(-30, 30)
            ny = rng.randint(-30, 30)
        
        # Check bounds
        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
            
        # Check collision
        # Fast check: is (nx, ny) already in cur_pts?
        # Since N is small, a linear search or set check is fine.
        # We maintain a set for O(1) lookup? 
        # Let's just check against the array for simplicity, it's 30 items.
        # Or better, maintain a set of occupied cells.
        
        # To keep it simple and robust, we check collision by comparing to all other points.
        # If collision, skip.
        new_pts = cur_pts.copy()
        new_pts[idx] = [nx, ny]
        
        # Check if new position is occupied by another point
        # We can do this by checking if (nx, ny) exists in cur_pts excluding idx
        # This is O(N) per move. N=30, so it's cheap.
        occupied = False
        for i in range(_N):
            if i != idx:
                if cur_pts[i][0] == nx and cur_pts[i][1] == ny:
                    occupied = True
                    break
        if occupied:
            continue
            
        # Evaluate new score
        new_score = _score(new_pts)
        
        # Acceptance criterion
        delta = new_score - cur_score
        if delta > 0:
            # Always accept improvement
            cur_pts = new_pts
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_pts = cur_pts.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = np.exp(delta / T)
            if rng.random() < prob:
                cur_pts = new_pts
                cur_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Initial candidate: Grid
    init_pts = _generate_initial_grid()
    best_pts, best_score = _hill_climb_sa(init_pts, deadline, seed=42)
    
    # If time remains, try other initializations
    if time.time() < deadline - 10:
        # Try a slightly different grid or random cluster
        # Random cluster: points within a small box
        rng = random.Random(123)
        cluster_pts = []
        used = set()
        while len(cluster_pts) < 30:
            x = rng.randint(-3, 3)
            y = rng.randint(-3, 3)
            if (x, y) not in used:
                used.add((x, y))
                cluster_pts.append((x, y))
        cluster_arr = np.array(cluster_pts, dtype=np.int64)
        p2, s2 = _hill_climb_sa(cluster_arr, deadline, seed=123)
        if s2 > best_score:
            best_pts, best_score = p2, s2
            
    # Final validation and formatting
    seen = set()
    out = []
    for p in best_pts:
        x, y = int(p[0]), int(p[1])
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
            
    # Pad if necessary (should not happen)
    if len(out) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(out) >= 30:
                    break
                if (x, y) not in seen:
                    seen.add((x, y))
                    out.append((x, y))
                    
    return out[:30]
