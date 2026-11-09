import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0

    def compute_areas_vec(pts):
        """
        Compute absolute doubled areas for all triples using vectorization.
        Returns a dictionary of area -> count.
        """
        n = len(pts)
        if n < 3:
            return {}
        
        # Convert to numpy array for fast operations
        P = np.array(pts, dtype=np.int64)
        
        # We need to compute det for all (i, j, k) with i < j < k
        # Vectorizing all triples at once (30 choose 3 = 4060) is fast enough.
        # However, doing it in a loop over i is also fine and memory efficient.
        # Let's use a loop over i for clarity and moderate memory usage, 
        # but vectorize the j,k part.
        
        area_counts = defaultdict(int)
        
        for i in range(n - 2):
            pi = P[i]
            # Vectors from i to all j > i
            # We need pairs (j, k) with i < j < k
            # Let's extract sub-array of points from i+1 to end
            sub = P[i+1:]
            m = len(sub)
            if m < 2:
                continue
            
            # For each j in sub, we need k in sub[j+1:]
            # This is still a bit of a loop. 
            # Alternative: 
            # Area = 0.5 * | (xj-xi)(yk-yi) - (yj-yi)(xk-xi) |
            # Let dx_j = xj - xi, dy_j = yj - yi
            # Let dx_k = xk - xi, dy_k = yk - yi
            # Det = | dx_j * dy_k - dy_j * dx_k |
            
            # We can compute this for all pairs in sub using broadcasting?
            # sub is (m, 2). 
            # We want pairs (a, b) where index a < b in sub.
            # This is a lower triangular matrix of interactions.
            
            # Efficient way:
            # Create indices
            # Actually, for m=29, a simple nested loop in Python is slow.
            # Let's use NumPy broadcasting for the sub-problem.
            
            # dx, dy for all points in sub relative to pi
            dx = sub[:, 0] - pi[0]
            dy = sub[:, 1] - pi[1]
            
            # We need all pairs (u, v) with u < v in range(m)
            # We can create a matrix of differences?
            # Let's just do a loop over u in sub, and vectorize v.
            # m is at most 29. 29 iterations is nothing.
            
            for u in range(m - 1):
                du = dx[u]
                dv = dy[u]
                # v ranges from u+1 to m-1
                dv_vec = dy[u+1:]
                du_vec = dx[u+1:]
                
                # det = | du * dv_vec - dv * du_vec |
                dets = np.abs(du * dv_vec - dv * du_vec)
                
                # Filter out zeros
                non_zero = dets[dets > 0]
                if len(non_zero) > 0:
                    # Count occurrences
                    vals, counts = np.unique(non_zero, return_counts=True)
                    for val, cnt in zip(vals, counts):
                        area_counts[int(val)] += int(cnt)
                        
        return area_counts

    def score_of(pts):
        areas = compute_areas_vec(pts)
        if not areas:
            return 0
        return max(areas.values())

    # 1. Initialize with a good structured grid
    # 6x5 grid is a standard starting point
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = score_of(best_points)
    
    # Try a few other grid shapes to see if one is better initially
    grid_configs = [
        [(x, y) for x in range(10) for y in range(3)],
        [(x, y) for x in range(5) for y in range(6)],
        [(x, y) for x in range(15) for y in range(2)],
        # Hexagonal packing approximation?
        # For now, stick to rectangles.
    ]
    
    for config in grid_configs:
        if len(config) != 30:
            continue
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in config):
            s = score_of(config)
            if s > best_score:
                best_score = s
                best_points = config

    # 2. Local Search (Hill Climbing)
    # Strategy: Pick a random point, try moving it to a random nearby location.
    # If the score improves, accept.
    # To speed up, we can focus on improving the count of the *current* most frequent area,
    # but global score improvement is safer.
    
    current_points = list(best_points)
    current_score = best_score
    
    # Pre-compute current areas to identify the "target" area?
    # The task asks to maximize the *largest multiplicity*. 
    # So we just want to maximize max(counts).
    
    random.seed(42)
    step_size = 3  # Initial step size for random moves
    
    # Optimize: 
    # Instead of full recomputation (which is ~4000 ops), 
    # we can try to be smarter, but 4000 ops is very fast in NumPy.
    # 4000 * (small constant) is likely < 1ms.
    # 60s / 1ms = 60,000 iterations. This is plenty.
    
    while elapsed() < DEADLINE:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # Bias towards small moves to stay in the "basin"
        new_x = old_x + random.randint(-step_size, step_size)
        new_y = old_y + random.randint(-step_size, step_size)
        
        # Ensure bounds and distinctness
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check if new point collides with existing ones
        # We can do this quickly
        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j] == (new_x, new_y):
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Try the move
        current_points[idx] = (new_x, new_y)
        
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
            # Accept move
            # Maybe increase step size slightly? Or keep it.
            # Let's keep step size constant for stability, or decay it slowly.
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            
        # Occasionally try a larger jump to escape local maxima
        if random.random() < 0.01:
            step_size = random.randint(1, 10)
        else:
            # Decay step size slightly to refine
            if step_size > 1 and random.random() < 0.1:
                step_size = max(1, step_size - 1)

    return best_points
