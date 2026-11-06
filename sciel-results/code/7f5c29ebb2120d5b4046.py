import random
import itertools
import numpy as np
import time
import math

def _check_no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                a = yj - yi
                b = xi - xj
                c = xj*yi - xi*yj
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    xl, yl = points[l]
                    if a * xl + b * yl + c == 0:
                        return False
    return True

def _score(points):
    """Largest subset with no three collinear (exact branch and bound)."""
    n = len(points)
    if n == 0:
        return 0
        
    # Precompute collinearity masks
    # collinear_mask[i] is a bitmask of points j such that there exists a k 
    # where i, j, k are collinear? No.
    # We need: for a given set of chosen points C, and a candidate i,
    # i is invalid if there exist a, b in C such that a, b, i are collinear.
    # This is equivalent to: for each a in C, if we add i, 
    # we must ensure that no b in C is on the line (a, i).
    # Let line_mask[a][i] be the bitmask of all points b such that a, i, b are collinear.
    # Then i is invalid if (line_mask[a][i] & C_mask) != 0 for any a in C.
    
    line_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            # Find all k collinear with i, j
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    mask |= (1 << k)
            line_mask[i][j] = mask
            line_mask[j][i] = mask

    best = [0]
    
    def dfs(idx, current_mask, current_size):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            best[0] = max(best[0], current_size)
            return
        
        # Option 1: Skip idx
        dfs(idx + 1, current_mask, current_size)
        
        # Option 2: Take idx
        # Check if idx can be added to current_mask
        can_take = True
        # Iterate over set bits in current_mask
        m = current_mask
        while m:
            lsb = m & -m
            a_idx = lsb.bit_length() - 1
            # Check if any point in current_mask is collinear with a_idx and idx
            # line_mask[a_idx][idx] contains all points b such that a_idx, idx, b are collinear
            # If any such b is in current_mask, we have a collinear triple (a_idx, idx, b)
            if line_mask[a_idx][idx] & current_mask:
                can_take = False
                break
            m ^= lsb
            
        if can_take:
            dfs(idx + 1, current_mask | (1 << idx), current_size + 1)

    dfs(0, 0, 0)
    return best[0]

def construct():
    # Use Simulated Annealing to minimize the score
    # Start with a random valid configuration or a perturbed grid
    
    def get_initial_points():
        # 5x4 grid perturbed to break 4-collinear
        # Grid: x in 0..4, y in 0..3
        # Perturb each point by a small random amount
        # Ensure no 4 collinear by checking and re-perturbing if necessary
        random.seed(42)
        for _ in range(1000):
            pts = []
            for i in range(5):
                for j in range(4):
                    x = i * 10 + random.randint(-2, 2)
                    y = j * 10 + random.randint(-2, 2)
                    pts.append([x, y])
            if len(set(tuple(p) for p in pts)) != 20:
                continue
            if _check_no_four_collinear(pts):
                return pts
        # Fallback: parabola (score 20, but valid)
        return [[x, x * x] for x in range(-10, 10)]

    points = get_initial_points()
    current_score = _score(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for overhead
    
    T = 5.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        new_points = [p[:] for p in points]
        idx = random.randint(0, 19)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        # Check bounds
        if abs(new_points[idx][0]) > 100 or abs(new_points[idx][1]) > 100:
            continue
            
        # Check distinctness
        if len(set(tuple(p) for p in new_points)) != 20:
            continue
            
        # Check no 4 collinear
        if not _check_no_four_collinear(new_points):
            continue
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score < current_score:
            points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    points = new_points
                    current_score = new_score
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_points
