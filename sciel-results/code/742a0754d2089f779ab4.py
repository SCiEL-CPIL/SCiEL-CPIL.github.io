import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0

    p = 7
    k = 3  # alpha^2 = 3

    # Field operations
    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_mul(e1, e2):
        a, b = e1
        c, d = e2
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    # Precompute all points and lines
    # Points are (xa, xb, ya, yb)
    # Map point to index
    points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = len(points)
                    points.append((xa, xb, ya, yb))
                    point_to_idx[(xa, xb, ya, yb)] = idx
    
    N_points = len(points)
    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    # v = (1, m) means (dx_a, dx_b) = (1, 0) for x part? No.
    # Point x = (x, y). v = (1, m).
    # x + t*v = (x_x + t, x_y + t*m).
    # x_x = (xa, xb). t = (ta, tb).
    # new_x = (xa + ta, xb + tb).
    # new_y = (ya + tm_a, yb + tm_b).
    
    lines = [] # List of lists of point indices
    point_lines = [[] for _ in range(N_points)] # List of line indices for each point
    
    # Add vertical lines: v = (0, 1)
    # x + t*(0,1) = (x_x, x_y + t)
    # t = (ta, tb). x_y + t = (ya + ta, yb + tb).
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # This is not efficient. A line is defined by x-coord and direction.
                    pass
    
    # Better: Iterate through all possible lines
    # Type 1: Vertical (dx=0, dy=1). Defined by x=(xa,xb).
    # Points: (xa, xb, ya, yb) for all ya, yb.
    # There are 49 such lines.
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_idx[(xa, xb, ya, yb)]
                    line_pts.append(idx)
            line_id = len(lines)
            lines.append(line_pts)
            for idx in line_pts:
                point_lines[idx].append(line_id)

    # Type 2: Slope m. v = (1, m). Defined by y - m*x = c.
    # Or just generate by picking a start point and direction.
    # To avoid duplicates, we can just generate all lines by picking a point and direction,
    # and using a set of frozensets to deduplicate? 
    # 2401 points * 49 slopes = 117k lines. 
    # Each line has 49 points.
    # Total line-point incidences: 2401 * 50 = 120050.
    # This is manageable.
    
    # We already added 49 vertical lines.
    # Now add 49 slopes * 49 x-coords? No.
    # A line with slope m is determined by its y-intercept?
    # y = m*x + b.
    # For each m (49 values), and each b (49 values), we have a line.
    # Total 49*49 = 2401 lines.
    
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    b = (ba, bb)
                    line_pts = []
                    # y = m*x + b
                    # For each x in F_49, y is determined.
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            y = f_add(f_mul(m, x), b)
                            ya, yb = y
                            idx = point_to_idx[(xa, xb, ya, yb)]
                            line_pts.append(idx)
                    
                    # Check if this line already exists (shouldn't if we iterate uniquely)
                    # But vertical lines are distinct from these.
                    line_id = len(lines)
                    lines.append(line_pts)
                    for idx in line_pts:
                        point_lines[idx].append(line_id)
    
    # Total lines: 49 + 2401 = 2450.
    # Each point is on 50 lines.
    
    # State
    C = set() # Set of point indices in Complement
    valid_lines_count = [50] * N_points # Each point has 50 lines, all valid initially
    
    # Line state: (count, point_idx_if_count_1)
    # Initially count=0
    line_count = [0] * len(lines)
    line_single_point = [-1] * len(lines)
    
    def try_add(x_idx):
        """Try to add x_idx to C. Returns True if successful."""
        if x_idx in C:
            return True
        # Check if adding x_idx violates any constraint
        # We need to simulate the effect.
        
        # For each line L through x_idx:
        for l_idx in point_lines[x_idx]:
            old_count = line_count[l_idx]
            
            # Determine which points lose validity
            # If old_count == 0:
            #   All y in L (y != x) lose validity.
            #   x keeps validity (was valid, still valid).
            # If old_count == 1:
            #   Let z be the single point.
            #   z loses validity.
            #   Others keep (were invalid).
            #   x keeps (was invalid, still invalid).
            # If old_count >= 2:
            #   No change.
            
            if old_count == 0:
                # All other points on this line lose this valid line
                for y_idx in lines[l_idx]:
                    if y_idx != x_idx:
                        valid_lines_count[y_idx] -= 1
                        if valid_lines_count[y_idx] < 0: # Should not happen
                            return False
            elif old_count == 1:
                z_idx = line_single_point[l_idx]
                valid_lines_count[z_idx] -= 1
                if valid_lines_count[z_idx] < 0:
                    return False
            # else: no change
        
        # If no counts went negative, the move is valid.
        # Apply the move.
        C.add(x_idx)
        for l_idx in point_lines[x_idx]:
            old_count = line_count[l_idx]
            if old_count == 0:
                line_count[l_idx] = 1
                line_single_point[l_idx] = x_idx
            elif old_count == 1:
                line_count[l_idx] = 2
                line_single_point[l_idx] = -1
            else:
                line_count[l_idx] = old_count + 1
                line_single_point[l_idx] = -1
        
        return True

    def try_remove(x_idx):
        """Try to remove x_idx from C. Returns True if successful."""
        if x_idx not in C:
            return True
        # Simulate removal
        for l_idx in point_lines[x_idx]:
            old_count = line_count[l_idx]
            
            if old_count == 1:
                # Must be x_idx
                # All other points on L gain validity
                for y_idx in lines[l_idx]:
                    if y_idx != x_idx:
                        valid_lines_count[y_idx] += 1
            elif old_count == 2:
                # Other point z gains validity
                # Find z
                for y_idx in lines[l_idx]:
                    if y_idx != x_idx and y_idx in C:
                        valid_lines_count[y_idx] += 1
                        break
            # else: no change
        
        # Apply removal
        C.remove(x_idx)
        for l_idx in point_lines[x_idx]:
            old_count = line_count[l_idx]
            if old_count == 1:
                line_count[l_idx] = 0
                line_single_point[l_idx] = -1
            elif old_count == 2:
                line_count[l_idx] = 1
                # Find the other point
                for y_idx in lines[l_idx]:
                    if y_idx != x_idx and y_idx in C:
                        line_single_point[l_idx] = y_idx
                        break
            else:
                line_count[l_idx] = old_count - 1
                line_single_point[l_idx] = -1
        
        return True

    # Search
    best_C_size = 0
    best_C = set()
    
    # Random restarts
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Reset state
        C = set()
        valid_lines_count = [50] * N_points
        line_count = [0] * len(lines)
        line_single_point = [-1] * len(lines)
        
        # Greedy Add
        indices = list(range(N_points))
        random.shuffle(indices)
        
        for x_idx in indices:
            if time.time() - start_time > deadline:
                break
            if try_add(x_idx):
                pass
        
        current_size = len(C)
        if current_size > best_C_size:
            best_C_size = current_size
            best_C = set(C)
            
        # Local Search: Try to add more by swapping?
        # Or just try to add points that were skipped?
        # Simple improvement: Try to add points that are currently not in C,
        # but allow temporary invalidity? No, must be valid.
        
        # Try to remove and re-add?
        # Let's try a random walk of removals and additions.
        for _ in range(1000):
            if time.time() - start_time > deadline:
                break
            if not C:
                break
            x_idx = random.choice(list(C))
            # Try removing x_idx
            # We need to check if removal is "safe" in terms of not blocking future additions?
            # No, we just want to maximize size.
            # If we remove, size decreases. We only remove if it allows adding more later?
            # This is complex.
            
            # Simpler: Just do more random restarts.
            pass

    # Construct N
    N_points_list = []
    for i in range(N_points):
        if i not in best_C:
            xa, xb, ya, yb = points[i]
            N_points_list.append([[xa, xb], [ya, yb]])
            
    if not N_points_list:
        # Fallback: Empty complement (N = all points)
        for i in range(N_points):
            xa, xb, ya, yb = points[i]
            N_points_list.append([[xa, xb], [ya, yb]])

    return np.array(N_points_list, dtype=np.int64)
