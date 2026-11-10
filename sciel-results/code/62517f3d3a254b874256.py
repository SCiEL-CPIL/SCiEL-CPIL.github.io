# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_points = q * q
    
    # Field elements as 0..48
    # x = xa + xb * alpha
    # We map (xa, xb) to xa + xb * 7
    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    
    lines = [] # list of 50 lists of sets of points
    
    # Vertical
    vert_lines = []
    for x in range(q):
        line = set()
        for y in range(q):
            line.add(x * q + y)
        vert_lines.append(line)
    lines.append(vert_lines)
    
    # Slopes m in F_49
    for m in range(q):
        slope_lines = []
        # For each c in F_49, line is y = m*x + c
        for c in range(q):
            line = set()
            for x in range(q):
                # y = m*x + c
                ma, mb = divmod(m, 7)
                xa, xb = divmod(x, 7)
                # m*x = (ma*xa + 3*mb*xb) + (ma*xb + mb*xa)*alpha
                ya = (ma * xa + 3 * mb * xb) % 7
                yb = (ma * xb + mb * xa) % 7
                y = ya + yb * 7
                # y = y + c
                ca, cb = divmod(c, 7)
                ya = (ya + ca) % 7
                yb = (yb + cb) % 7
                y = ya + yb * 7
                line.add(x * q + y)
            slope_lines.append(line)
        lines.append(slope_lines)
        
    # Precompute line indices for each point and direction
    line_idx = np.zeros((50, n_points), dtype=int)
    for d in range(50):
        for li, line in enumerate(lines[d]):
            for pt in line:
                line_idx[d, pt] = li
                
    def is_valid(S):
        shadow_mask = np.ones(n_points, dtype=bool)
        bad_mask = np.ones(n_points, dtype=bool)
        
        for d in range(50):
            line_sum = np.zeros(len(lines[d]), dtype=int)
            # Accumulate sums
            # Using numpy might be faster if we have a flat array
            # But loop is 2401, which is small.
            for pt in range(n_points):
                if S[pt]:
                    li = line_idx[d, pt]
                    line_sum[li] += 1
            
            li_for_all = line_idx[d, :]
            sums_for_all = line_sum[li_for_all]
            
            shadow_mask &= (sums_for_all > 0)
            
            sum_other = sums_for_all - S.astype(int)
            bad_mask &= (sum_other > 0)
            
        if np.any(shadow_mask & ~S):
            return False
        if np.any(bad_mask & S):
            return False
        return True
        
    # Construct baseline S
    S = np.zeros(n_points, dtype=bool)
    squares = {0, 1, 2, 4}
    for xa in range(7):
        for xb in range(7):
            x_idx = xa + xb * 7
            for ya in range(7):
                for yb in range(7):
                    y_idx = ya + yb * 7
                    pt = x_idx * q + y_idx
                    if ya not in squares:
                        S[pt] = True
                        
    # Greedy add
    current_S = S.copy()
    candidates = np.where(~current_S)[0]
    np.random.shuffle(candidates)
    
    for pt in candidates:
        new_S = current_S.copy()
        new_S[pt] = True
        if is_valid(new_S):
            current_S = new_S
            
    points = []
    for i in range(n_points):
        if current_S[i]:
            x_idx = i // q
            y_idx = i % q
            xa, xb = divmod(x_idx, 7)
            ya, yb = divmod(y_idx, 7)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
