import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _generate_line_pool(max_coeff=5):
    """Generate a pool of candidate lines with small integer coefficients."""
    pool = set()
    # Ensure we have a good variety of slopes
    # A*x + B*y + C = 0
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_coeff * 2, max_coeff * 2 + 1):
                norm = _normalize_line(A, B, C)
                # Avoid degenerate lines or duplicates
                if norm != (0, 0, 0):
                    pool.add(norm)
    return list(pool)

def _evaluate_line_set(lines, box_limit=30):
    """
    Evaluate a set of 27 lines.
    Returns (score, points_list).
    """
    n_lines = len(lines)
    # Find all integer intersection points
    # We only care about points within a reasonable box to avoid huge coordinates
    # and to ensure we can easily find 54 points if needed.
    
    intersections = {} # (x, y) -> count of lines passing through it
    
    for i in range(n_lines):
        A1, B1, C1 = lines[i]
        for j in range(i + 1, n_lines):
            A2, B2, C2 = lines[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            
            if num_x % det == 0 and num_y % det == 0:
                x = num_x // det
                y = num_y // det
                
                if -box_limit <= x <= box_limit and -box_limit <= y <= box_limit:
                    intersections[(x, y)] = 0 # Initialize
    
    # Now, for each unique intersection point, count how many lines pass through it
    # This is more accurate than just counting pair intersections because of concurrency
    # But checking all 27 lines for each point is O(N_points * 27).
    # N_points is at most 351. 351 * 27 is small.
    
    point_degrees = []
    for p in intersections.keys():
        x, y = p
        deg = 0
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                deg += 1
        point_degrees.append((deg, p))
        
    # Sort by degree descending
    point_degrees.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 54
    if len(point_degrees) < 54:
        # If we don't have enough integer intersections, we need to pad.
        # This will result in a lower score, so the search should avoid this.
        # But we must return 54 points.
        # We'll pad with arbitrary points far away that don't lie on any line.
        used_points = set([p for d, p in point_degrees])
        score = sum(d for d, p in point_degrees)
        pts = [p for d, p in point_degrees]
        
        # Pad
        while len(pts) < 54:
            px = random.randint(-100, 100)
            py = random.randint(-100, 100)
            if (px, py) not in used_points:
                pts.append((px, py))
                used_points.add((px, py))
                # These points contribute 0 to score
        return score, pts
    else:
        top_54 = point_degrees[:54]
        score = sum(d for d, p in top_54)
        pts = [p for d, p in top_54]
        return score, pts

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    # 1. Generate Pool
    # A larger pool allows better exploration.
    # max_coeff=4 gives a good balance.
    pool = _generate_line_pool(max_coeff=4)
    random.shuffle(pool)
    
    # 2. Initialization
    # Start with a grid configuration which is known to be good.
    # 7 vertical + 7 horizontal = 14 lines.
    # Add 13 diagonals.
    initial_lines = set()
    for c in range(7):
        initial_lines.add(_normalize_line(1, 0, -c))
        initial_lines.add(_normalize_line(0, 1, -c))
    
    # Add diagonals x+y=c and x-y=c
    added = 0
    for c in range(-10, 11):
        if added >= 13:
            break
        l1 = _normalize_line(1, 1, -c)
        if l1 not in initial_lines:
            initial_lines.add(l1)
            added += 1
            if added >= 13:
                break
        if added >= 13:
            break
        l2 = _normalize_line(1, -1, -c)
        if l2 not in initial_lines:
            initial_lines.add(l2)
            added += 1
            
    # If we still don't have 27, fill from pool
    if len(initial_lines) < 27:
        for l in pool:
            if len(initial_lines) >= 27:
                break
            if l not in initial_lines:
                initial_lines.add(l)
                
    # Ensure exactly 27
    if len(initial_lines) > 27:
        # Remove random ones
        lines_list = list(initial_lines)
        random.shuffle(lines_list)
        initial_lines = set(lines_list[:27])
        
    current_lines = list(initial_lines)
    current_score, current_points = _evaluate_line_set(current_lines)
    
    best_score = current_score
    best_lines = list(current_lines)
    best_points = current_points
    
    # 3. Local Search
    # We use a simple hill climbing with random restarts if stuck, or just aggressive swapping.
    # Since evaluation is fast, we can do many steps.
    
    pool_set = set(pool)
    # Remove lines already in current set from pool for swapping?
    # No, we can swap with any line from the pool.
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Select a line to remove
        idx_out = random.randint(0, 26)
        l_out = current_lines[idx_out]
        
        # Select a line to add
        # Bias towards lines in the pool that are not currently in the set?
        # Or just any line from the pool.
        l_in = random.choice(pool)
        
        if l_in == l_out:
            continue
            
        # Create new set
        new_lines = current_lines.copy()
        new_lines[idx_out] = l_in
        
        # Evaluate
        new_score, new_points = _evaluate_line_set(new_lines)
        
        if new_score > best_score:
            best_score = new_score
            best_lines = new_lines
            best_points = new_points
            current_lines = new_lines
            no_improve_steps = 0
        elif new_score > current_score:
            # Accept hill climb
            current_lines = new_lines
            current_score = new_score
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            
        # If stuck for a long time, restart from best?
        if no_improve_steps > max_no_improve:
            # Perturb current state significantly or restart
            # Let's try to restart from the best found so far with a random perturbation
            current_lines = best_lines.copy()
            # Swap 3 random lines
            for _ in range(3):
                idx = random.randint(0, 26)
                current_lines[idx] = random.choice(pool)
            current_score, current_points = _evaluate_line_set(current_lines)
            no_improve_steps = 0
            
        steps += 1
        if steps % 1000 == 0:
            # Check time
            pass

    # Finalize
    # Ensure points are exactly 54 and distinct
    final_points = best_points[:54]
    
    # Validation and padding just in case
    if len(final_points) < 54:
        used = set(final_points)
        while len(final_points) < 54:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in used:
                final_points.append(p)
                used.add(p)
                
    points_arr = np.array(final_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    # Double check shapes
    if points_arr.shape != (54, 2) or lines_arr.shape != (27, 3):
        # Fallback to a valid grid
        pts = []
        for i in range(7):
            for j in range(7):
                pts.append((i, j))
        for i in range(7, 12): # 5 more points
            pts.append((i, 0))
        points_arr = np.array(pts, dtype=np.int64)
        
        ls = []
        for i in range(8):
            ls.append((1, 0, -i))
        for j in range(7):
            ls.append((0, 1, -j))
        # Need 27 lines. 8+7=15. Add 12 more.
        for c in range(12):
             ls.append((1, 1, -c))
        lines_arr = np.array(ls, dtype=np.int64)

    return points_arr, lines_arr
