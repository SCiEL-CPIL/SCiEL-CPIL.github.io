import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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
    
    # Run multiple restarts
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue  # Invalid due to timeout
        
        # 2. Local Search
        # Phase 1: Remove redundant codewords
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
            
        # Phase 2: 1-Swap (Remove 1, Add 1)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find candidates that cover all critical words
                candidates = set(balls[critical_words[0]].tolist())
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break
                
                candidates.discard(c)
                
                if candidates:
                    # Score candidates by how much redundancy they create
                    best_c = None
                    best_score = -1
                    for c_prime in list(candidates)[:50]:  # Limit for speed
                        score = np.sum(cover_count[balls[c_prime]] > 1)
                        if score > best_score:
                            best_score = score
                            best_c = c_prime
                    if best_c is not None:
                        cover_count[balls[c]] -= 1
                        cover_count[balls[best_c]] += 1
                        new_selected.append(best_c)
                        improved = True
                    else:
                        new_selected.append(c)
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1) with Random Sampling
        if len(selected) > 486 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                # Sample a subset of pairs to try
                num_trials = min(3000, len(selected) * (len(selected) - 1) // 2)
                
                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break
                    
                    idx1 = random.randint(0, len(selected) - 1)
                    idx2 = random.randint(0, len(selected) - 1)
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
                        if len(uncovered_words) <= 8:
                            candidates = set(balls[uncovered_words[0]].tolist())
                            for uw in uncovered_words[1:]:
                                candidates.intersection_update(balls[uw].tolist())
                                if not candidates:
                                    break
                            
                            candidates.discard(c1)
                            candidates.discard(c2)
                            
                            if candidates:
                                c3 = random.choice(list(candidates))
                                cover_count[balls[c3]] += 1
                                # Remove c1, c2 from selected
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
                        
                    if not improved_2ex:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
                        
        # Phase 4: Final cleanup - remove any remaining redundant codes
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
            
        # Verify coverage
        final_cov = np.zeros(total, dtype=np.int32)
        for c in selected:
            final_cov[balls[c]] += 1
            
        if np.any(final_cov == 0):
            # Invalid - skip this iteration
            continue
            
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = list(selected)
            
        # Early exit if we've hit the known best
        if best_size <= 486:
            break
    
    # Return best result
    if best_set is None:
        # Fallback: return full space
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
