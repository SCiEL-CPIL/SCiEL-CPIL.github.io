import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls
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
    
    # Convert balls to a list of sets for faster membership tests
    ball_sets = [set(row.tolist()) for row in balls]
    
    # Precompute for each word, the list of codewords whose ball contains it
    word_to_codewords = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            word_to_codewords[w].append(c)
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        # 1. Greedy construction with vectorized counting
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            # Vectorized count of uncovered words covered by each candidate
            counts = np.zeros(total, dtype=np.int32)
            for w in range(ball_size):
                counts += uncovered[balls[:, w]].astype(np.int32)
            
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
        
        # 2. Pruning: remove redundant codewords using vectorized cover counts
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            # For each selected codeword, check if all words in its ball have cover_count > 1
            removable_mask = np.array([
                np.min(cover_count[balls[c]]) > 1 for c in selected
            ])
            
            if not np.any(removable_mask):
                break
            
            remove_indices = selected[removable_mask]
            selected = selected[~removable_mask]
            for c in remove_indices:
                cover_count[balls[c]] -= 1
            changed = True
        
        # 3. Local search: improved swap strategy
        swap_attempts = 0
        max_swaps_per_round = 50
        
        while swap_attempts < max_swaps_per_round and time.time() < deadline - 5.0:
            if len(selected) == 0:
                break
            
            swap_attempts += 1
            
            # Pick a random codeword to potentially swap
            c_idx = np.random.randint(len(selected))
            c = selected[c_idx]
            
            # Try a few random candidates
            found_valid = False
            for _ in range(5):
                c_prime = np.random.randint(total)
                if c_prime == c:
                    continue
                
                # Check if c_prime is already selected
                if c_prime in selected:
                    continue
                
                # Check validity: words in ball(c) not in ball(c_prime) must have cover_count > 1
                bc = ball_sets[c]
                bcp = ball_sets[c_prime]
                
                valid = True
                for w in bc:
                    if w not in bcp:
                        if cover_count[w] == 1:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Perform swap
                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                selected[c_idx] = c_prime
                found_valid = True
                
                # Re-prune
                changed = True
                while changed and time.time() < deadline - 5.0:
                    removable_mask = np.array([
                        np.min(cover_count[balls[c]]) > 1 for c in selected
                    ])
                    if not np.any(removable_mask):
                        break
                    remove_indices = selected[removable_mask]
                    selected = selected[~removable_mask]
                    for c in remove_indices:
                        cover_count[balls[c]] -= 1
                    changed = True
                
                break
        
        # 4. Final polish: try to remove any remaining redundant codewords
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            if len(selected) == 0:
                break
            
            # For each selected codeword, check if it can be removed
            removable_found = False
            for i in range(len(selected)):
                c = selected[i]
                bc = ball_sets[c]
                can_remove = True
                for w in bc:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove this codeword
                    cover_count[balls[c]] -= 1
                    selected = np.delete(selected, i)
                    removable_found = True
                    break
            
            if not removable_found:
                break
        
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
