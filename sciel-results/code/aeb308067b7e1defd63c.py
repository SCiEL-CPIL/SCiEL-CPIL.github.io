import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls: for each word, its 17 neighbors (itself + 16 at distance 1)
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
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    best_size = total
    best_set = None
    rng = np.random.default_rng(42)
    
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # === PHASE 1: Randomized Greedy Construction ===
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
            
            # Count how many uncovered words each candidate covers
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(int(best_idx))
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            # Incomplete due to timeout - fix by adding codewords for uncovered words
            while np.any(uncovered_mask):
                if time.time() > deadline - 8.0:
                    break
                uncov_idx = int(np.where(uncovered_mask)[0][0])
                if uncov_idx not in selected:
                    selected.append(uncov_idx)
                    cover_count[balls[uncov_idx]] += 1
                    uncovered_mask = (cover_count == 0)
                else:
                    # This shouldn't happen, but break to avoid infinite loop
                    break
        
        if np.any(uncovered_mask):
            continue  # Still invalid, try again
        
        # === PHASE 2: Local Search ===
        
        # 2a: Remove all removable codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # 2b: 1-Swap (Remove 1, Add 1)
        improved = True
        max_swap_rounds = 30
        for round_num in range(max_swap_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Shuffle order for diversity
            order = rng.permutation(len(selected))
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                if idx >= len(selected):
                    continue
                    
                c = selected[idx]
                
                # Try to remove
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    del selected[idx]
                    improved = True
                    # Adjust order since we removed an element
                    if idx < len(order) - 1:
                        order = np.delete(order, idx)
                    continue
                
                # Try to swap with a codeword that covers all critical words
                cover_count[balls[c]] -= 1
                critical = balls[c][cover_count[balls[c]] == 1]
                
                if len(critical) == 0:
                    # c was removable
                    del selected[idx]
                    improved = True
                    if idx < len(order) - 1:
                        order = np.delete(order, idx)
                    continue
                
                # Find candidates that cover all critical words
                candidates = set(balls[int(critical[0])].tolist())
                for cw in critical[1:]:
                    candidates.intersection_update(balls[int(cw)].tolist())
                    if not candidates:
                        break
                
                candidates.discard(c)
                
                if candidates:
                    c_prime = int(list(candidates)[rng.integers(len(candidates))])
                    cover_count[balls[c_prime]] += 1
                    selected[idx] = c_prime
                    improved = True
                else:
                    # Revert the removal of c
                    cover_count[balls[c]] += 1
        
        # 2c: 2-Exchange (Remove 2, Add 1) with Random Sampling
        if len(selected) > 486 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                num_trials = min(3000, len(selected) * (len(selected) - 1) // 2)
                
                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break
                    
                    idx1 = rng.integers(len(selected))
                    idx2 = rng.integers(len(selected))
                    if idx1 == idx2:
                        continue
                        
                    c1 = selected[idx1]
                    c2 = selected[idx2]
                    
                    # Temporarily remove
                    cover_count[balls[c1]] -= 1
                    cover_count[balls[c2]] -= 1
                    
                    uncovered_after = (cover_count == 0)
                    
                    if np.any(uncovered_after):
                        uncovered_words = np.where(uncovered_after)[0]
                        
                        # Find c3 that covers all uncovered words
                        candidates = set(balls[int(uncovered_words[0])].tolist())
                        for uw in uncovered_words[1:]:
                            candidates.intersection_update(balls[int(uw)].tolist())
                            if not candidates:
                                break
                        
                        candidates.discard(c1)
                        candidates.discard(c2)
                        
                        if candidates:
                            c3 = int(list(candidates)[rng.integers(len(candidates))])
                            cover_count[balls[c3]] += 1
                            # Remove c1, c2 from selected and add c3
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            new_selected.append(c3)
                            selected = new_selected
                            improved_2ex = True
                            break
                        else:
                            # Revert
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Both can be removed
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        selected = new_selected
                        improved_2ex = True
                        break
        
        # 2d: Final cleanup - remove any newly removable codewords
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break
    
    # Convert to output format
    if best_set is None:
        # Fallback: return all words (should never happen)
        return [list(c) for c in itertools.product(range(q), repeat=n)]
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
