import itertools
import math
import time
import numpy as np
import random

_score_cache = {}

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    key = tuple(sorted(U))
    if key in _score_cache:
        return _score_cache[key]
    U_arr = np.array(U)
    plus = np.add.outer(U_arr, U_arr)
    minus = np.subtract.outer(U_arr, U_arr)
    n_plus = len(np.unique(plus))
    n_minus = len(np.unique(minus))
    if n_plus == 0:
        _score_cache[key] = 0.0
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        _score_cache[key] = 1.0
        return 1.0
    result = 1.0 + math.log(ratio) / math.log(2 * mx + 1)
    _score_cache[key] = result
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s wall clock

    best_score = 0.0
    best_set = [0, 1, 3]

    def try_set(U):
        nonlocal best_score, best_set
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)

    # Phase 1: Try structured sets
    structured_sets = []
    # Known good small sets
    structured_sets.append([0, 1, 2, 4])
    structured_sets.append([0, 1, 2, 4, 8])
    structured_sets.append([0, 1, 3, 4, 8])
    structured_sets.append([0, 1, 3])
    structured_sets.append([0, 1, 2, 3, 5])
    structured_sets.append([0, 1, 2, 4, 5, 8])
    structured_sets.append([0, 1, 3, 4, 9])
    
    # Powers of 2
    for n in range(2, 12):
        structured_sets.append([0] + [2**i for i in range(n)])
    # Small arithmetic progressions with gaps
    for a in range(1, 8):
        for d in range(1, 8):
            structured_sets.append([0] + [a * i * d for i in range(1, 7)])
    # Mixed: small + large
    for small in range(1, 6):
        for large in range(10, 50, 3):
            structured_sets.append([0, small, large])
            structured_sets.append([0, small, large, 2*large])
            
    # Fibonacci-like
    fib = [0, 1]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    for i in range(2, len(fib)):
        structured_sets.append(fib[:i+1])
    
    # Lacunary sequences (powers of 3, 4, etc.)
    for base in [3, 4, 5]:
        seq = [0]
        val = 1
        for i in range(8):
            seq.append(val)
            val *= base
        for i in range(2, len(seq)):
            structured_sets.append(seq[:i+1])
    
    # Hybrid: powers of 2 with some offsets
    for offset in range(0, 4):
        for n in range(3, 10):
            seq = [0]
            for i in range(1, n):
                seq.append(2**i + offset)
            structured_sets.append(seq)

    for U in structured_sets:
        if time.time() > deadline:
            return best_set
        try_set(U)

    # Phase 2: Limited exhaustive search over small ranges (only sizes 2-4)
    max_val = 20
    candidates = list(range(1, max_val + 1))

    for size in range(2, 5):
        if time.time() > deadline - 30:  # Reserve time for local search
            break
        for combo in itertools.combinations(candidates, size - 1):
            if time.time() > deadline - 30:
                break
            U = [0] + list(combo)
            try_set(U)

    # Phase 3: Robust Local Search with Simulated Annealing
    # Start from the best set found so far
    current = list(best_set)
    current_score = best_score
    
    # Define a range of candidates for moves
    max_candidate = 150
    
    # Simulated annealing parameters
    temperature = 0.01
    cooling_rate = 0.995
    min_temperature = 0.001
    sa_step = 0
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # 1. Try adding a new element
        candidates_to_try = set()
        for x in range(1, max_candidate + 1):
            candidates_to_try.add(x)
        # Add some structured candidates based on current set
        for x in current:
            for y in current:
                if x != y:
                    candidates_to_try.add(x + y)
                    candidates_to_try.add(abs(x - y))
                    if x * y < max_candidate:
                        candidates_to_try.add(x * y)
        
        # Filter candidates to those not in current set and within reasonable bounds
        valid_candidates = [x for x in candidates_to_try if x not in current and x <= max_candidate]
        
        best_add_score = current_score
        best_add_x = None
        for x in valid_candidates:
            if time.time() > deadline - 5.0: break
            new_U = sorted(current + [x])
            s = _score(new_U)
            if s > best_add_score:
                best_add_score = s
                best_add_x = x
        
        if best_add_x is not None:
            current = sorted(current + [best_add_x])
            current_score = best_add_score
            improved = True
            try_set(current)
            continue # Restart loop to re-evaluate other moves with new state

        # 2. Try removing an element
        best_rem_score = current_score
        best_rem_idx = None
        for i in range(1, len(current)):
            if time.time() > deadline - 5.0: break
            new_U = current[:i] + current[i+1:]
            s = _score(new_U)
            if s > best_rem_score:
                best_rem_score = s
                best_rem_idx = i
        
        if best_rem_idx is not None:
            current = current[:best_rem_idx] + current[best_rem_idx+1:]
            current_score = best_rem_score
            improved = True
            try_set(current)
            continue # Restart loop

        # 3. Try swapping an element
        best_swap_score = current_score
        best_swap_i = None
        best_swap_x = None
        for i in range(1, len(current)):
            if time.time() > deadline - 5.0: break
            # Try swapping current[i] with a candidate
            swap_candidates = [x for x in range(1, max_candidate + 1) if x not in current]
            for x in swap_candidates:
                if time.time() > deadline - 5.0: break
                new_U = current[:i] + [x] + current[i+1:]
                s = _score(new_U)
                if s > best_swap_score:
                    best_swap_score = s
                    best_swap_i = i
                    best_swap_x = x
        
        if best_swap_i is not None:
            current = current[:best_swap_i] + [best_swap_x] + current[best_swap_i+1:]
            current = sorted(current)
            current_score = best_swap_score
            improved = True
            try_set(current)
            continue # Restart loop

        if not improved:
            # No single move improves the score.
            # Try simulated annealing: accept worse moves with some probability
            sa_step += 1
            if temperature > min_temperature:
                # Generate a random move
                move_type = random.choice(['add', 'remove', 'swap'])
                new_U = None
                new_score = current_score
                
                if move_type == 'add' and len(current) < 20:
                    candidates = [x for x in range(1, max_candidate + 1) if x not in current]
                    if candidates:
                        x = random.choice(candidates)
                        new_U = sorted(current + [x])
                        new_score = _score(new_U)
                elif move_type == 'remove' and len(current) > 1:
                    i = random.randint(1, len(current) - 1)
                    new_U = current[:i] + current[i+1:]
                    new_score = _score(new_U)
                elif move_type == 'swap' and len(current) > 1:
                    i = random.randint(1, len(current) - 1)
                    candidates = [x for x in range(1, max_candidate + 1) if x not in current]
                    if candidates:
                        x = random.choice(candidates)
                        new_U = current[:i] + [x] + current[i+1:]
                        new_U = sorted(new_U)
                        new_score = _score(new_U)
                
                if new_U is not None:
                    # Accept if better, or with probability exp(-delta/T) if worse
                    delta = current_score - new_score
                    if delta < 0 or random.random() < math.exp(-delta / temperature):
                        current = new_U
                        current_score = new_score
                        try_set(current)
                        improved = True
                        temperature *= cooling_rate
                        continue
            
            # Try a "pair move": add one, remove one simultaneously
            best_pair_score = current_score
            best_pair_rem = None
            best_pair_add = None
            
            removal_candidates = list(range(1, len(current)))
            if len(removal_candidates) > 10:
                removal_candidates = removal_candidates[:10]
            
            for i in removal_candidates:
                if time.time() > deadline - 5.0: break
                base_U = current[:i] + current[i+1:]
                add_candidates = [x for x in range(1, 80) if x not in base_U]
                for x in add_candidates:
                    if time.time() > deadline - 5.0: break
                    new_U = sorted(base_U + [x])
                    s = _score(new_U)
                    if s > best_pair_score:
                        best_pair_score = s
                        best_pair_rem = i
                        best_pair_add = x
            
            if best_pair_rem is not None:
                new_U = current[:best_pair_rem] + current[best_pair_rem+1:]
                new_U = sorted(new_U + [best_pair_add])
                current = new_U
                current_score = best_pair_score
                improved = True
                try_set(current)
                continue
            
            # If no pair move improved and SA is cooling, restart from best
            if temperature <= min_temperature:
                # Final polish: try more exhaustive local moves from best_set
                current = list(best_set)
                current_score = best_score
                temperature = 0.005  # Small restart
                sa_step = 0
            else:
                temperature *= cooling_rate

    # Final polish: Ensure we return the best set found
    return best_set
