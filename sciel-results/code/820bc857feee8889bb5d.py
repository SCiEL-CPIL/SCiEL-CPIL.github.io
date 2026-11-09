"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Build candidate pool: dense grid in [-100, 100]
    # Use a subset to keep search tractable: sample from grid
    all_points = []
    for x in range(-50, 51, 2):
        for y in range(-50, 51, 2):
            all_points.append((x, y))
    random.shuffle(all_points)
    pool_size = min(500, len(all_points))
    pool = all_points[:pool_size]
    n_pool = len(pool)
    
    # Precompute line signatures for pairs in pool
    # For each pair (i, j), compute normalized line key
    # Use (dx, dy) reduced to primitive, then offset
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        g = abs(dx)
        if abs(dy) > g:
            g = abs(dy)
        # Compute GCD
        a, b = abs(dx), abs(dy)
        while b:
            a, b = b, a % b
        g = a
        if g == 0:
            g = 1
        dx_n = dx // g
        dy_n = dy // g
        # Canonical direction: ensure first non-zero is positive
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        # Offset: for vertical lines, use x; for horizontal, use y; else use a,b such that a*x + b*y = c
        # For line through p1 with direction (dx_n, dy_n): normal is (dy_n, -dx_n)
        # c = dy_n * x1 - dx_n * y1
        c = dy_n * p1[0] - dx_n * p1[1]
        return (dx_n, dy_n, c)
    
    # Build line map: for each line key, list of pool indices on that line
    line_map = {}
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            lk = line_key(pool[i], pool[j])
            if lk is None:
                continue
            if lk not in line_map:
                line_map[lk] = set()
            line_map[lk].add(i)
            line_map[lk].add(j)
    
    # For each line, store sorted list of indices
    lines = []
    for lk, idx_set in line_map.items():
        if len(idx_set) >= 3:
            lines.append(sorted(idx_set))
    
    # Objective: given a set of selected indices (size 20), compute:
    # - number of collinear triples (3 selected points on same line)
    # - penalty if any line has >= 4 selected points
    # We want to MAXIMIZE triples, MINIMIZE four-collinear violations
    # Score to minimize = largest subset with no 3 collinear (hard to compute exactly fast)
    # Proxy: minimize (four_violations * 1000 - triples)
    
    def evaluate(selected):
        """Return (penalty, -triples) where lower is better."""
        selected_set = set(selected)
        four_count = 0
        triple_count = 0
        for line in lines:
            count = 0
            for idx in line:
                if idx in selected_set:
                    count += 1
            if count >= 4:
                four_count += 1
            elif count == 3:
                triple_count += 1
        return (four_count, -triple_count)
    
    # Initial solution: random 20 points
    best_solution = random.sample(range(n_pool), 20)
    best_score = evaluate(best_solution)
    
    # Simulated annealing
    temp = 10.0
    cooling = 0.999
    min_temp = 0.01
    
    current_solution = best_solution[:]
    current_score = best_score
    
    step = 0
    while time.time() < deadline and temp > min_temp:
        # Generate neighbor: swap one selected point with one unselected
        if random.random() < 0.7:
            # Swap
            sel_idx = random.randint(0, 19)
            sel_point = current_solution[sel_idx]
            # Pick random unselected
            unselected = [i for i in range(n_pool) if i not in set(current_solution)]
            if not unselected:
                break
            new_point = random.choice(unselected)
            new_solution = current_solution[:]
            new_solution[sel_idx] = new_point
        else:
            # Single move: remove one, add one (keep size 20)
            sel_idx = random.randint(0, 19)
            sel_point = current_solution[sel_idx]
            unselected = [i for i in range(n_pool) if i not in set(current_solution)]
            if not unselected:
                break
            new_point = random.choice(unselected)
            new_solution = current_solution[:]
            new_solution[sel_idx] = new_point
        
        new_score = evaluate(new_solution)
        
        # Accept if better or with probability exp(-delta/T)
        delta = (new_score[0] - current_score[0]) * 1000.0 + (new_score[1] - current_score[1])
        if delta < 0 or (temp > 0 and random.random() < pow(2.718281828, -delta / temp)):
            current_solution = new_solution
            current_score = new_score
            if current_score < best_score:
                best_solution = current_solution[:]
                best_score = current_score
        
        temp *= cooling
        step += 1
    
    # Final verification: check no four collinear
    final_set = set(best_solution)
    valid = True
    for line in lines:
        count = 0
        for idx in line:
            if idx in final_set:
                count += 1
        if count >= 4:
            valid = False
            break
    
    if not valid:
        # Fallback: return parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return [[pool[i][0], pool[i][1]] for i in best_solution]
# EVOLVE-BLOCK-END
