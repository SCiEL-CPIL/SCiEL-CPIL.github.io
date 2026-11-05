import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate all valid pairs (i, j) such that 2^i * 3^j <= 10000
    valid_pairs = []
    for j in range(9):  # 3^8 = 6561, 3^9 > 10000
        max_val = 10000 // (3 ** j)
        for i in range(14):  # 2^13 = 8192, 2^14 > 10000
            if (2 ** i) <= max_val:
                valid_pairs.append((i, j))
            else:
                break
    
    # We need to select 20 distinct pairs from valid_pairs
    # Initial guess: a 4x5 rectangle of exponents
    initial_set = set()
    for i in range(4):
        for j in range(5):
            initial_set.add((i, j))
            
    current_set = list(initial_set)
    
    # Precompute values for valid pairs for faster sum calculation
    pair_to_val = {}
    for p in valid_pairs:
        i, j = p
        pair_to_val[p] = 2**i * 3**j

    def evaluate(pairs):
        # Product set size from exponents
        exp_sum_set = set()
        for i1, j1 in pairs:
            for i2, j2 in pairs:
                exp_sum_set.add((i1 + i2, j1 + j2))
        prod_size = len(exp_sum_set)
        
        # Sum set size from values
        vals = [pair_to_val[p] for p in pairs]
        sum_set = set()
        for a in vals:
            for b in vals:
                sum_set.add(a + b)
        sum_size = len(sum_set)
        
        return max(prod_size, sum_size)
        
    current_score = evaluate(current_set)
    best_set = list(current_set)
    best_score = current_score
    
    # Simulated Annealing
    all_pairs_set = set(valid_pairs)
    current_set_set = set(current_set)
    
    T = 50.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Generate a move: swap one element
        idx_remove = random.randint(0, 19)
        elem_remove = current_set[idx_remove]
        
        available = all_pairs_set - current_set_set
        if not available:
            continue
        elem_add = random.choice(list(available))
        
        # Apply move
        new_set = list(current_set)
        new_set[idx_remove] = elem_add
        new_set_set = set(new_set)
        
        # Evaluate
        new_score = evaluate(new_set)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_set = new_set
            current_set_set = new_set_set
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            pass
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            
    # Return the best set found
    best_vals = [pair_to_val[p] for p in best_set]
    return sorted(best_vals)

# EVOLVE-BLOCK-END
