import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases: Greedy
    return _generic_greedy(q, n, radius, deadline)

def _generic_greedy(q, n, radius, deadline):
    # Simple greedy for non-specific cases
    total = q ** n
    # Precompute balls (expensive for large n, but n is small in this task usually)
    # For generic case, we might just return a trivial code if n is large
    if n > 10:
        return [list(c) for c in itertools.product(range(q), repeat=n)]
        
    weights = [q ** (n - 1 - i) for i in range(n)]
    balls = []
    for w in range(total):
        ball = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                ball.append(w + (new_digit - digit) * weight)
        balls.append(ball)
        
    uncovered = np.ones(total, dtype=bool)
    selected = []
    while np.any(uncovered) and time.time() < deadline - 5.0:
        # Greedy: pick codeword covering most uncovered words
        counts = uncovered[balls].sum(axis=1)
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[balls[best_idx]] = False
        
    result = []
    for w in selected:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
    return result

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n # 17
    
    # Precompute balls as numpy array
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(3):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
                
    # Precompute ball sets for fast membership/diff checks if needed, 
    # but numpy indexing is faster for bulk ops.
    # For swap checks, we need to check specific words.
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    # Helper: Greedy construction
    def greedy_init(seed_val):
        np.random.seed(seed_val)
        uncovered = np.ones(total, dtype=bool)
        selected = []
        while np.any(uncovered) and time.time() < deadline - 20.0:
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            uncovered[balls[best_idx]] = False
        return np.array(selected, dtype=np.int32)
    
    # Helper: Local Search Optimization
    def optimize(selected, time_limit):
        selected = np.array(selected, dtype=np.int32)
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[selected]] += 1
        
        # Phase 1: Greedy Removal
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Check each codeword
            for i in range(len(selected)):
                c = selected[i]
                # If min cover count in ball(c) is > 1, we can remove c
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected = np.delete(selected, i)
                    changed = True
                    break # Restart loop after removal
            
        # Phase 2: Swap-based Local Search (Simulated Annealing style)
        current_size = len(selected)
        T = 1.0
        T_min = 0.1
        alpha = 0.99
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_limit and T > T_min:
            T *= alpha
            if len(selected) == 0:
                break
                
            # Pick a random codeword to swap
            c_idx = np.random.randint(len(selected))
            c = selected[c_idx]
            
            # Pick a random replacement
            c_prime = np.random.randint(total)
            if c_prime == c:
                continue
            if c_prime in selected:
                continue
                
            # Check validity of swap
            # Words in B(c) not in B(c_prime) must have cover_count > 1
            bc = balls[c]
            bcp = balls[c_prime]
            
            # Create a set of bcp for fast lookup? 
            # Or use numpy: 
            # We need to check words in bc that are NOT in bcp.
            # Since balls are small (17), we can do this manually or with a boolean mask.
            # Let's use a temporary boolean array for bcp membership? 
            # Faster: 
            # unique_words = bc[~np.isin(bc, bcp)]
            # if len(unique_words) == 0: valid
            # else: if np.min(cover_count[unique_words]) > 1: valid
            
            unique_words = bc[~np.isin(bc, bcp)]
            valid = True
            if len(unique_words) > 0:
                if np.min(cover_count[unique_words]) <= 1:
                    valid = False
            
            if not valid:
                continue
                
            # Calculate delta in size (0 for swap)
            # But we might enable removals later.
            # For now, just swap if valid.
            # To accept "worse" swaps, we'd need a metric. 
            # Since size doesn't change, we accept if it leads to future removals.
            # We'll just accept all valid swaps to explore the space.
            
            # Perform swap
            cover_count[balls[c]] -= 1
            cover_count[balls[c_prime]] += 1
            selected[c_idx] = c_prime
            
            # Try to remove redundant codewords immediately
            removals = 0
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                for i in range(len(selected)):
                    c_rem = selected[i]
                    if np.min(cover_count[balls[c_rem]]) > 1:
                        cover_count[balls[c_rem]] -= 1
                        selected = np.delete(selected, i)
                        removals += 1
                        changed = True
                        break
            
            if removals > 0:
                no_improve_count = 0
                if len(selected) < current_size:
                    current_size = len(selected)
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
        return selected, cover_count

    # Main Loop: Multiple Restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with greedy
        selected = greedy_init(42 + i)
        
        # Optimize
        time_limit_for_this_run = min(deadline - 10.0, time.time() + 2.0)
        selected, _ = optimize(selected, time_limit_for_this_run)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # If we found a very good code, we can spend more time polishing it
            if best_size <= 440:
                # Intensive local search on the best found
                time_limit_for_polish = deadline - 5.0
                best_set, _ = optimize(best_set, time_limit_for_polish)
                best_size = len(best_set)
                
    if best_set is None:
        best_set = greedy_init(42)
        
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
