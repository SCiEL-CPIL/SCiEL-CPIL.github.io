import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
                
    # Precompute a set of balls for fast intersection checks?
    # Or just use boolean masks.
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    
    while time.time() < deadline - 10.0:
        # 1. Greedy
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            counts = uncovered[balls].sum(axis=1)
            # Random tie break
            # Add small random noise to break ties
            # Or just pick random from max
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            uncovered[balls[best_idx]] = False
            
            if time.time() > deadline - 15.0:
                break
        
        if np.any(uncovered):
            continue
            
        selected = np.array(selected, dtype=np.int32)
        
        # 2. Pruning
        # Calculate cover counts
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[selected]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            # mins for each selected codeword
            mins = np.min(cover_count[balls[selected]], axis=1)
            removable_mask = mins > 1
            
            if not np.any(removable_mask):
                break
                
            remove_idx = selected[removable_mask]
            selected = selected[~removable_mask]
            
            cover_count[balls[remove_idx]] -= 1
            changed = True
            
        # 3. Local Search (Swap)
        # Try to swap codewords to enable further pruning or reduce size
        # We only accept swaps that keep the code valid.
        # Then we re-prune.
        
        # Optimization: Precompute ball membership as sets? Too slow in Python.
        # Use boolean vectors.
        
        for _ in range(50): # Limit swap attempts per restart
            if time.time() > deadline - 5.0:
                break
            if len(selected) == 0:
                break
                
            # Pick random codeword to swap out
            c_idx = np.random.randint(len(selected))
            c = selected[c_idx]
            
            # Pick random candidate to swap in
            c_prime = np.random.randint(total)
            if c_prime == c:
                continue
                
            # Check if swap is valid
            # Words in B_c that are NOT in B_c_prime must have cover_count > 1
            # Words in B_c that ARE in B_c_prime are safe (covered by c_prime)
            
            # Create mask for B_c_prime
            # This is expensive: 17 ops.
            # We can check:
            # unique_in_c = balls[c] where cover_count == 1
            # If any of those are NOT in balls[c_prime], swap fails.
            
            unique_words = balls[c][cover_count[balls[c]] == 1]
            
            if len(unique_words) == 0:
                # c is redundant? Should have been pruned.
                # Just swap and hope?
                # Actually if min > 1 was false, there is at least one unique.
                # So this case shouldn't happen if pruning was perfect.
                pass
            else:
                # Check if all unique_words are in balls[c_prime]
                # balls[c_prime] is small.
                # We can check membership by broadcasting?
                # unique_words (K, 1), balls[c_prime] (1, B)
                # This is O(K*B). K is small (<=17).
                
                # Simple check:
                # Create a set of balls[c_prime]
                # Set creation in Python loop is slow.
                # Use numpy:
                # in_cp = np.isin(unique_words, balls[c_prime])
                # if not np.all(in_cp): continue
                
                # isin might be slower for small arrays.
                # Let's use a simple loop for the few unique words.
                valid_swap = True
                cp_set = set(balls[c_prime].tolist())
                for w in unique_words:
                    if w not in cp_set:
                        valid_swap = False
                        break
                
                if not valid_swap:
                    continue
            
            # Perform swap
            # Remove c
            cover_count[balls[c]] -= 1
            # Add c_prime
            cover_count[balls[c_prime]] += 1
            
            # Update selected
            selected[c_idx] = c_prime
            
            # Re-prune
            changed = True
            while changed and time.time() < deadline - 5.0:
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if not np.any(removable_mask):
                    break
                remove_idx = selected[removable_mask]
                selected = selected[~removable_mask]
                cover_count[balls[remove_idx]] -= 1
                changed = True
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break

    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
        
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
