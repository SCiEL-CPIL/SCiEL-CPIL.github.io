# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    
    Uses efficient precomputation and greedy removal with missing_count tracking.
    """
    p = 7
    q = 49  # 7*7
    
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    # Precompute addition and multiplication tables for speed
    
    # All field elements as flat indices: idx = a*7 + b
    def add_table():
        tab = np.zeros((q, q), dtype=np.int32)
        for i in range(q):
            a1, b1 = i // 7, i % 7
            for j in range(q):
                a2, b2 = j // 7, j % 7
                tab[i, j] = ((a1 + a2) % 7) * 7 + ((b1 + b2) % 7)
        return tab
    
    def mul_table():
        tab = np.zeros((q, q), dtype=np.int32)
        for i in range(q):
            a1, b1 = i // 7, i % 7
            for j in range(q):
                a2, b2 = j // 7, j % 7
                ra = (a1 * a2 + 3 * b1 * b2) % 7
                rb = (a1 * b2 + a2 * b1) % 7
                tab[i, j] = ra * 7 + rb
        return tab
    
    add_tab = add_table()
    mul_tab = mul_table()
    
    # Points in F_49^2: point index = x_idx * 49 + y_idx
    total_points = 2401
    
    # Precompute for each point and direction, the other 48 points on the same line
    # Directions: 0-48 are slopes m (direction (1, m)), 49 is vertical (0, 1)
    # line_other[p, d] = array of 48 point indices (the other points on the line through p in direction d)
    
    # For efficiency, precompute using numpy vectorization where possible
    
    # For direction (1, m) where m is field element with index m_idx:
    # Line through (x, y): points are (x + t, y + m*t) for t in F_49
    # For each point p = (x_idx, y_idx) and slope m_idx:
    #   other points: (add_tab[x_idx, t_idx], add_tab[y_idx, mul_tab[m_idx, t_idx]]) for t_idx != 0
    
    # Precompute line_other as a 3D array: [total_points, 50, 48]
    line_other = np.zeros((total_points, 50, 48), dtype=np.int32)
    
    # For each slope direction (0-48)
    for m_idx in range(49):
        # For each point
        for p_idx in range(total_points):
            x_idx = p_idx // 49
            y_idx = p_idx % 49
            
            # Compute other 48 points
            for t_idx in range(1, 49):
                new_x = add_tab[x_idx, t_idx]
                new_y = add_tab[y_idx, mul_tab[m_idx, t_idx]]
                other_pt = new_x * 49 + new_y
                line_other[p_idx, m_idx, t_idx - 1] = other_pt
    
    # For vertical direction (0, 1), direction index 49
    for p_idx in range(total_points):
        x_idx = p_idx // 49
        y_idx = p_idx % 49
        
        for t_idx in range(1, 49):
            new_y = add_tab[y_idx, t_idx]
            other_pt = x_idx * 49 + new_y
            line_other[p_idx, 49, t_idx - 1] = other_pt
    
    # Check time
    if time.time() > deadline:
        # Return all points if precomputation took too long
        return np.zeros((2401, 2, 2), dtype=np.int64)
    
    # Greedy removal using missing_count
    in_N = np.ones(total_points, dtype=bool)
    # missing_count[x, d] = number of points on x's line in direction d that are NOT in N (excluding x)
    missing_count = np.zeros((total_points, 50), dtype=np.int32)
    # valid_count[x] = number of directions d where missing_count[x, d] == 0
    valid_count = np.full(total_points, 50, dtype=np.int32)
    
    # Process points in a fixed order (we could randomize, but let's keep it deterministic for now)
    # Try removing each point; if it would make any point have valid_count == 0, skip
    
    excluded = 0
    
    for p_idx in range(total_points):
        if time.time() > deadline:
            break
        
        if not in_N[p_idx]:
            continue
        
        # Tentatively remove p_idx
        # For each direction d, for each x in line_other[p_idx, d]:
        #   missing_count[x, d] += 1
        #   If this makes missing_count[x, d] == 1 (was 0), valid_count[x] -= 1
        #   If valid_count[x] becomes 0, we cannot remove p_idx
        
        can_remove = True
        # Track changes for rollback: list of (x, d, old_mc)
        changes = []
        
        for d in range(50):
            others = line_other[p_idx, d]  # array of 48 point indices
            for x in others:
                old_mc = missing_count[x, d]
                new_mc = old_mc + 1
                missing_count[x, d] = new_mc
                changes.append((x, d, old_mc))
                
                if old_mc == 0:  # valid_dir[x, d] was True, now becomes False
                    valid_count[x] -= 1
                    if valid_count[x] == 0:
                        can_remove = False
                        break
            if not can_remove:
                break
        
        if can_remove:
            in_N[p_idx] = False
            excluded += 1
        else:
            # Roll back
            for (cx, cd, cmd) in changes:
                missing_count[cx, cd] = cmd
                if cmd == 0:
                    valid_count[cx] += 1
    
    # Build result: list of points in N
    points = []
    for i in range(total_points):
        if in_N[i]:
            x_idx = i // 49
            y_idx = i % 49
            xa, xb = x_idx // 7, x_idx % 7
            ya, yb = y_idx // 7, y_idx % 7
            points.append([[xa, xb], [ya, yb]])
    
    if len(points) == 0:
        # Should not happen, but return all points as fallback
        return np.zeros((2401, 2, 2), dtype=np.int64)
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
