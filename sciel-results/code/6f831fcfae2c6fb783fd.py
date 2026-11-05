import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # We'll try n values from high to low
    n_values = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]
    
    # Time allocation: give more time to higher n values
    # Total time budget: deadline - start_time
    time_budget = deadline - start_time
    
    # Allocate time proportionally but with minimum for lower n
    # Higher n gets more time since finding a valid coloring is harder but more valuable
    weights = []
    for i, n in enumerate(n_values):
        # Exponential decay for lower n
        weights.append(max(1.0, 2.0 ** (len(n_values) - 1 - i)))
    
    total_weight = sum(weights)
    time_allocs = []
    for w in weights:
        time_allocs.append(time_budget * w / total_weight)
    
    # Minimum time for any n value
    min_time = 0.5
    max_time = time_budget * 0.6  # No single n gets more than 60% of total time
    
    for i, n in enumerate(n_values):
        if time.time() >= deadline - 2.0:
            break
            
        alloc = min(max(time_allocs[i], min_time), max_time)
        
        result = solve_n(n, start_time + alloc - start_time + (time.time() - start_time))
        if result is not None:
            return result
    
    # If all failed, return a valid small solution
    # For n=9, use the seed construction
    n_seed = 9
    triples_seed = list(combinations(range(n_seed), 3))
    colors_seed = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_seed]
    
    # Verify it's valid
    if _is_valid(n_seed, colors_seed):
        return {'n': n_seed, 'colors': colors_seed}
    
    # Fallback to even smaller
    for n_try in range(8, 0, -1):
        colors_try = [random.randint(0, 1) for _ in range(len(list(combinations(range(n_try), 3))))]
        if _is_valid(n_try, colors_try):
            return {'n': n_try, 'colors': colors_try}
    
    # Last resort: n=1
    return {'n': 1, 'colors': []}

def _is_valid(n, colors):
    """Check if coloring is valid: no red K4, no blue K5."""
    if n < 4:
        return True
    
    # Build triple to index map
    triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Check for red K4
    for k4 in combinations(range(n), 4):
        all_red = True
        for t in combinations(k4, 3):
            if colors[triple_to_idx[t]] != 0:
                all_red = False
                break
        if all_red:
            return False
    
    # Check for blue K5
    for k5 in combinations(range(n), 5):
        all_blue = True
        for t in combinations(k5, 3):
            if colors[triple_to_idx[t]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True

def solve_n(n, time_limit):
    """Solve for a valid coloring of n vertices within time_limit seconds."""
    start = time.time()
    deadline = start + time_limit
    
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4 and K5 structures
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    k4_triples = []
    for k4 in k4s:
        t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
        k4_triples.append(t_indices)
        
    k5_triples = []
    for k5 in k5s:
        t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
        k5_triples.append(t_indices)
    
    # Build incidence matrices
    M4 = np.zeros((len(k4_triples), num_triples), dtype=np.int8)
    for i, t_indices in enumerate(k4_triples):
        M4[i, t_indices] = 1
        
    M5 = np.zeros((len(k5_triples), num_triples), dtype=np.int8)
    for i, t_indices in enumerate(k5_triples):
        M5[i, t_indices] = 1
    
    def calc_cost_fast(colors):
        sums4 = M4.dot(colors)
        red_k4s = np.sum(sums4 == 0)
        sums5 = M5.dot(colors)
        blue_k5s = np.sum(sums5 == 10)
        return red_k4s + blue_k5s
    
    # Multiple initialization strategies
    best_cost = float('inf')
    best_colors = None
    
    random.seed(42)
    
    # Strategy 1: Random
    colors = np.random.randint(0, 2, num_triples).astype(np.int8)
    cost = calc_cost_fast(colors)
    if cost == 0:
        return {'n': n, 'colors': colors.tolist()}
    if cost < best_cost:
        best_cost = cost
        best_colors = colors.copy()
    
    # Strategy 2: Sum mod 2
    colors2 = np.array([1 if (a + b + c) % 2 == 0 else 0 for a, b, c in triples], dtype=np.int8)
    cost2 = calc_cost_fast(colors2)
    if cost2 == 0:
        return {'n': n, 'colors': colors2.tolist()}
    if cost2 < best_cost:
        best_cost = cost2
        best_colors = colors2.copy()
    
    # Strategy 3: Product-based
    colors3 = np.array([1 if ((a * b) % 3) == ((b * c) % 3) else 0 for a, b, c in triples], dtype=np.int8)
    cost3 = calc_cost_fast(colors3)
    if cost3 == 0:
        return {'n': n, 'colors': colors3.tolist()}
    if cost3 < best_cost:
        best_cost = cost3
        best_colors = colors3.copy()
    
    # Strategy 4: XOR-based
    colors4 = np.array([(a ^ b) ^ (b ^ c) for a, b, c in triples], dtype=np.int8)
    cost4 = calc_cost_fast(colors4)
    if cost4 == 0:
        return {'n': n, 'colors': colors4.tolist()}
    if cost4 < best_cost:
        best_cost = cost4
        best_colors = colors4.copy()
    
    # Strategy 5: Random with bias
    for bias in [0.3, 0.7]:
        colors5 = (np.random.random(num_triples) < bias).astype(np.int8)
        cost5 = calc_cost_fast(colors5)
        if cost5 == 0:
            return {'n': n, 'colors': colors5.tolist()}
        if cost5 < best_cost:
            best_cost = cost5
            best_colors = colors5.copy()
    
    # Strategy 6: Local patterns
    colors6 = np.zeros(num_triples, dtype=np.int8)
    for i, (a, b, c) in enumerate(triples):
        colors6[i] = (a + b + c) % 3 if (a + b + c) % 3 != 0 else 1
    cost6 = calc_cost_fast(colors6)
    if cost6 == 0:
        return {'n': n, 'colors': colors6.tolist()}
    if cost6 < best_cost:
        best_cost = cost6
        best_colors = colors6.copy()
    
    # Strategy 7: Random walk
    colors7 = np.zeros(num_triples, dtype=np.int8)
    state = 0
    for i in range(num_triples):
        state = (state + 1) % 2
        colors7[i] = state
    cost7 = calc_cost_fast(colors7)
    if cost7 == 0:
        return {'n': n, 'colors': colors7.tolist()}
    if cost7 < best_cost:
        best_cost = cost7
        best_colors = colors7.copy()
    
    # Strategy 8: Block-based
    colors8 = np.zeros(num_triples, dtype=np.int8)
    for i, (a, b, c) in enumerate(triples):
        # Divide vertices into blocks
        block_a = a // 5
        block_b = b // 5
        block_c = c // 5
        colors8[i] = 1 if block_a == block_b == block_c else 0
    cost8 = calc_cost_fast(colors8)
    if cost8 == 0:
        return {'n': n, 'colors': colors8.tolist()}
    if cost8 < best_cost:
        best_cost = cost8
        best_colors = colors8.copy()
    
    # Start with the best initial solution
    current_colors = best_colors.copy()
    current_cost = best_cost
    
    # Simulated Annealing
    T = max(1.0, best_cost * 0.1)
    cooling_rate = 0.9995
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Adaptive restart
        if no_improve > max_no_improve:
            T = max(1.0, best_cost * 0.1)
            no_improve = 0
            # Random perturbation
            for _ in range(min(50, num_triples // 10)):
                idx = random.randint(0, num_triples - 1)
                current_colors[idx] = 1 - current_colors[idx]
            current_cost = calc_cost_fast(current_colors)
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors.copy()
                no_improve = 0
        
        idx = random.randint(0, num_triples - 1)
        
        old_color = current_colors[idx]
        new_color = 1 - old_color
        current_colors[idx] = new_color
        
        new_cost = calc_cost_fast(current_colors)
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0.001 and random.random() < np.exp(-delta / T)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors.copy()
                no_improve = 0
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors.tolist()}
            else:
                no_improve += 1
        else:
            current_colors[idx] = old_color
            no_improve += 1
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    # Final check
    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Verify best_colors is actually valid
    if best_colors is not None and _is_valid(n, best_colors.tolist()):
        return {'n': n, 'colors': best_colors.tolist()}
    
    return None

# EVOLVE-BLOCK-END
