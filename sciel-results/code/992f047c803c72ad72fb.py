"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _score(points):
    """Compute the max multiplicity of a positive doubled area using vectorized operations."""
    points = np.array(points, dtype=np.int32)
    n = len(points)
    if n < 3:
        return 0
    
    # Compute all pairwise differences
    # points[i] - points[j]
    # We need determinants for triples (i, j, k)
    # det = (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i)
    # Using broadcasting:
    # P1 = points[:, None, :] shape (n, 1, 2)
    # P2 = points[None, :, :] shape (1, n, 2)
    # P3 = points[None, None, :] shape (1, 1, n)
    # This creates arrays of shape (n, n, n) which might be too large for n=30?
    # 30^3 = 27000 elements. 27000 * 2 (coords) * 4 bytes (int32) = ~200KB. Very small.
    
    # Let's compute vectors v1 = P2 - P1 and v2 = P3 - P1
    # Actually, simpler:
    # For each i, j, k:
    # det = (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i)
    
    # We can compute this for all i,j,k at once.
    # To save memory and time, we can iterate over i, but vectorize j,k.
    # Or just do full broadcasting since n=30 is small.
    
    P = points
    # Shape: (n, 1, 1), (1, n, 1), (1, 1, n) for coordinates
    x_i = P[:, 0, None, None]  # (n, 1, 1)
    y_i = P[:, 1, None, None]  # (n, 1, 1)
    x_j = P[None, :, 0, None]  # (1, n, 1)
    y_j = P[None, :, 1, None]  # (1, n, 1)
    x_k = P[None, None, :, 0]  # (1, 1, n)
    y_k = P[None, None, :, 1]  # (1, 1, n)
    
    # dx_ji = x_j - x_i, dy_ji = y_j - y_i
    # dx_ki = x_k - x_i, dy_ki = y_k - y_i
    dx_ji = x_j - x_i
    dy_ji = y_j - y_i
    dx_ki = x_k - x_i
    dy_ki = y_k - y_i
    
    dets = dx_ji * dy_ki - dx_ki * dy_ji
    abs_dets = np.abs(dets)
    
    # Exclude zero area and i==j==k etc? 
    # The problem says "exclude zero-area triples". 
    # Also, technically i, j, k should be distinct indices.
    # If i==j, det is 0. If i==k, det is 0. If j==k, det is 0.
    # So non-zero dets imply distinct indices automatically.
    
    # We want the max frequency of any positive area value.
    # Since we took abs, we just look at non-zero values.
    valid_dets = abs_dets[abs_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    # Count frequencies
    # unique, counts = np.unique(valid_dets, return_counts=True)
    # return np.max(counts)
    
    # np.unique is fast enough for ~27000 elements.
    unique, counts = np.unique(valid_dets, return_counts=True)
    return int(np.max(counts))

def _random_points(n=30, bounds=30):
    pts = set()
    while len(pts) < n:
        x = random.randint(-bounds, bounds)
        y = random.randint(-bounds, bounds)
        pts.add((x, y))
    return list(pts)

def _grid_points(rows, cols, dx=1, dy=1, offset_x=0, offset_y=0):
    points = []
    for i in range(rows):
        for j in range(cols):
            x = offset_x + j * dx
            y = offset_y + i * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                points.append((x, y))
    return points

def construct():
    start_time = time.time()
    best_points = None
    best_score = 0
    
    # 1. Initialize with several diverse candidates
    candidates = []
    
    # Grids
    for rows in range(3, 11):
        for cols in range(3, 11):
            if rows * cols == 30:
                for dx in [1, 2]:
                    for dy in [1, 2]:
                        for ox in range(-5, 6, 2):
                            for oy in range(-5, 6, 2):
                                pts = _grid_points(rows, cols, dx, dy, ox, oy)
                                if len(pts) == 30 and len(set(pts)) == 30:
                                    candidates.append(pts)
                                    
    # Random starts
    for _ in range(50):
        candidates.append(_random_points(30))
        
    # 2. Local Search / Hill Climbing
    for pts in candidates:
        if time.time() - start_time > 50:
            break
            
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_points = pts[:]
            
        # Try to improve
        pts_arr = np.array(pts)
        improved = True
        # Limit iterations per candidate to avoid getting stuck
        max_iter = 100
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() - start_time > 55:
                break
                
            # Randomly pick a point to move
            idx = random.randint(0, 29)
            old_pt = pts[idx]
            
            # Try moving it to a nearby random location
            # Or just a random new location
            for _ in range(10):
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                new_pt = (new_x, new_y)
                
                # Check distinctness
                if new_pt in pts and new_pt != old_pt:
                    continue
                if new_pt == old_pt:
                    continue
                    
                # Swap
                pts[idx] = new_pt
                new_score = _score(pts)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = pts[:]
                    break
                else:
                    # Revert
                    pts[idx] = old_pt
            
            # If we didn't improve in this batch of 10 tries, maybe try a different point
            # The loop continues, picking a new idx each time
            
    # 3. Final Polish: try shifting all points by small offsets (if bounds allow)
    # This is a simple global move
    if best_points and time.time() - start_time < 58:
        # Try small translations
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                new_pts = [(x+dx, y+dy) for x, y in best_points]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in new_pts):
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_points = new_pts

    if best_points is None:
        # Fallback
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points
# EVOLVE-BLOCK-END
