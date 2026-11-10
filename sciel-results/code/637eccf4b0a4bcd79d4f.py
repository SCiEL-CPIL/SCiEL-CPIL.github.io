import time
import random
import math
from collections import defaultdict
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return (0, 0, 1) if C != 0 else (0, 0, 0)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0:
        if B < 0:
            B = -B
            C = -C
        elif B == 0:
            if C < 0:
                C = -C
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _evaluate(points):
    """
    Returns (score, is_valid)
    score: number of lines with exactly 4 points
    is_valid: False if any line has >= 5 points
    """
    n = len(points)
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_points(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
            
    count_4 = 0
    is_valid = True
    for key, indices in line_map.items():
        cnt = len(indices)
        if cnt == 4:
            count_4 += 1
        elif cnt >= 5:
            is_valid = False
            # We can break early if we just want validity, 
            # but we might want the count for invalid states if we use penalty.
            # Here we return 0 score for invalid to be strict, or a penalty.
            # Let's return a penalty score: count_4 - 100 * (cnt - 4)
            # But for SA, rejecting invalid moves is often cleaner.
            # Let's just mark invalid.
    return count_4, is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Define the pool of candidate points
    # Range -10 to 10 gives 21x21 = 441 points
    # This allows for many different slopes
    pool = []
    pool_coords = {}
    for x in range(-10, 11):
        for y in range(-10, 11):
            idx = len(pool)
            pool.append([x, y])
            pool_coords[(x, y)] = idx
    
    N_POOL = len(pool)
    N_POINTS = 16
    
    # Precompute lines for all pairs in the pool?
    # No, we only need lines for the selected points during evaluation.
    # However, we can precompute the line keys for all pairs in the pool
    # to speed up evaluation?
    # Evaluation takes O(N_POINTS^2) which is 120 ops. Very fast.
    # So we don't need heavy precomputation for the search itself.
    
    # Initial solution: 4x4 grid
    # Map 4x4 grid [0..3]x[0..3] to pool indices
    # Pool is -10..10. So 0..3 is within range.
    init_indices = []
    for x in range(0, 4):
        for y in range(0, 4):
            init_indices.append(pool_coords[(x, y)])
            
    current_selection = init_indices[:]
    current_points = [pool[i] for i in current_selection]
    current_score, current_valid = _evaluate(current_points)
    
    best_score = current_score
    best_points = [p[:] for p in current_points]
    
    if not current_valid:
        # Fallback to random if grid is invalid (should not be)
        current_selection = random.sample(range(N_POOL), N_POINTS)
        current_points = [pool[i] for i in current_selection]
        current_score, current_valid = _evaluate(current_points)
        best_score = current_score
        best_points = [p[:] for p in current_points]

    random.seed(42)
    
    # Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    steps = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        steps += 1
        
        # Pick a random index to swap out
        out_idx_pos = random.randint(0, N_POINTS - 1)
        out_pool_idx = current_selection[out_idx_pos]
        
        # Pick a random index to swap in (not currently selected)
        in_pool_idx = random.randint(0, N_POOL - 1)
        while in_pool_idx in current_selection:
            in_pool_idx = random.randint(0, N_POOL - 1)
            
        # Apply swap
        new_selection = current_selection[:]
        new_selection[out_idx_pos] = in_pool_idx
        
        new_points = [pool[i] for i in new_selection]
        
        new_score, new_valid = _evaluate(new_points)
        
        # If invalid, we can treat score as very negative
        if not new_valid:
            new_score = -1000 # Penalty
            
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_selection = new_selection
            current_points = new_points
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in new_points]
        else:
            # Accept with probability exp(delta / T)
            # If delta is very negative, exp is 0
            if delta > -100: # Avoid underflow/overflow issues with large penalties
                if random.random() < math.exp(delta / T):
                    current_selection = new_selection
                    current_points = new_points
                    current_score = new_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Restart strategy
        if no_improve_count > max_no_improve:
            # Reheat
            T = 1.0
            no_improve_count = 0
            # Optionally, randomize current state slightly
            # Or keep it, just reheat
            pass
            
    return best_points
