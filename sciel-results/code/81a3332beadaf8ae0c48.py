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
    # Range -15 to 15 gives 31x31 = 961 points
    # This allows for many different slopes and configurations
    pool = []
    pool_coords = {}
    for x in range(-15, 16):
        for y in range(-15, 16):
            idx = len(pool)
            pool.append([x, y])
            pool_coords[(x, y)] = idx
    
    N_POOL = len(pool)
    N_POINTS = 16
    
    # Initial solution: 4x4 grid
    # Map 4x4 grid [0..3]x[0..3] to pool indices
    init_indices = []
    for x in range(0, 4):
        for y in range(0, 4):
            init_indices.append(pool_coords[(x, y)])
            
    best_score = -1
    best_points = None
    
    random.seed(42)
    
    # Multi-restart Simulated Annealing
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial state
        if r == 0:
            current_selection = init_indices[:]
        else:
            # Mix of random and perturbed grid
            if r % 2 == 0:
                current_selection = random.sample(range(N_POOL), N_POINTS)
            else:
                current_selection = init_indices[:]
                # Perturb 2-4 points
                for _ in range(random.randint(2, 4)):
                    pos = random.randint(0, N_POINTS - 1)
                    new_idx = random.randint(0, N_POOL - 1)
                    while new_idx in current_selection:
                        new_idx = random.randint(0, N_POOL - 1)
                    current_selection[pos] = new_idx
        
        current_points = [pool[i] for i in current_selection]
        current_score, current_valid = _evaluate(current_points)
        
        if not current_valid:
            # If initial is invalid, try to fix or just skip this restart?
            # Better to just skip or heavily penalize. Let's skip invalid starts to save time.
            # Or we can just run SA and let it fix it.
            # Given the pool is large, random starts are rarely invalid (5 collinear is rare).
            pass
        
        # SA Parameters
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        max_steps = 20000 # Limit steps per restart to ensure we do many restarts
        
        no_improve_count = 0
        max_no_improve = 500
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
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
                
            # Restart strategy within SA
            if no_improve_count > max_no_improve:
                # Reheat
                T = 1.0
                no_improve_count = 0
                # Optionally, randomize current state slightly
                # Or keep it, just reheat
                pass
                
    if best_points is None:
        # Fallback
        best_points = [pool[i] for i in init_indices]
        
    return best_points
