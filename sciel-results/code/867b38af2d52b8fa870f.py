import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    import itertools
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute balls: balls[w] is the list of words within distance 1 of w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
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
    
    # Verify coverage function
    def is_covered(selected_set, balls):
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected_set:
            cover_count[balls[c]] += 1
        return np.all(cover_count > 0)
    
    best_size = total
    best_set = None
    
    # Strategy 1: Greedy with randomization + local search (2-opt swaps)
    # We run multiple iterations.
    
    def greedy_and_optimize(random_seed=None, deadline_check=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy phase
        while np.any(uncovered_mask):
            if deadline_check and time.time() > deadline_check:
                return None
            # Count how many uncovered words each candidate covers
            # candidates: 0..total-1
            # score[c] = sum(uncovered_mask[balls[c]])
            scores = uncovered_mask[balls].sum(axis=1)
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        if np.any(uncovered_mask):
            # Should not happen if greedy is correct, but safety
            for w in range(total):
                if uncovered_mask[w]:
                    selected.append(w)
                    cover_count[balls[w]] += 1
                    uncovered_mask = (cover_count == 0)
        
        # Local Search: Removal
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Local Search: 1-1 Swaps (try to replace a codeword with another to reduce size or improve structure for future removals)
        # Actually, 1-1 swap keeps size same. We want to enable more removals.
        # A better local search for set cover is often 1-1 replacement where we remove one and add one, 
        # hoping that the new one covers more "critical" spots allowing another removal later.
        # But simple 1-1 doesn't reduce size directly.
        # Let's do "try to remove c, if fails, try to find c' such that adding c' and removing c allows removing some other c''"
        # This is 2-1 move.
        
        # Simpler effective heuristic: Random 1-1 swap with acceptance if it doesn't break coverage, 
        # followed by removal pass. This is like simulated annealing but deterministic removal.
        
        improved = True
        while improved and (not deadline_check or time.time() < deadline_check - 2.0):
            improved = False
            # Try random swaps
            for i in range(len(selected)):
                if deadline_check and time.time() > deadline_check - 2.0:
                    break
                c = selected[i]
                # Try to find a c_prime to replace c
                # We want c_prime to cover some words that are only covered by c, 
                # or generally improve the "slack" of the system.
                # Critical words for c: those with cover_count == 1
                critical_words = balls[c][cover_count[balls[c]] == 1]
                
                # Find a random c_prime
                c_prime = np.random.randint(total)
                if c_prime == c:
                    continue
                if c_prime in selected:
                    continue
                
                # Check if c_prime covers all critical words of c
                # critical_words is an array of word indices
                # balls[c_prime] is the set of words c_prime covers
                # We need all(critical_words in balls[c_prime])
                
                # Efficient check:
                # Create a boolean array for words covered by c_prime
                # But doing this for every swap is expensive.
                # Alternative: precomputed membership?
                # balls is shape (total, 17).
                # For a given critical word w, is w in balls[c_prime]?
                # We can check if w == balls[c_prime][j] for some j.
                
                # Since critical_words is usually small, let's just iterate
                valid = True
                for w in critical_words:
                    if w not in balls[c_prime]:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    selected[i] = c_prime
                    improved = True
                    
                    # Try to remove redundant codewords after swap
                    removed = True
                    while removed:
                        removed = False
                        new_selected = []
                        for j in range(len(selected)):
                            cj = selected[j]
                            if np.min(cover_count[balls[cj]]) > 1:
                                cover_count[balls[cj]] -= 1
                                removed = True
                            else:
                                new_selected.append(cj)
                        selected = new_selected
                    break # Restart inner loop if improved
        
        if selected is not None and len(selected) > 0:
            return selected
        return None

    # Run multiple greedy+local search iterations
    num_iterations = 50
    for i in range(num_iterations):
        if time.time() > deadline - 10.0:
            break
        
        result = greedy_and_optimize(random_seed=i, deadline_check=deadline - 5.0)
        if result is not None:
            # Verify
            if is_covered(result, balls):
                if len(result) < best_size:
                    best_size = len(result)
                    best_set = result.copy()
                    if best_size <= 486:
                        break
            else:
                # Should not happen if logic is correct, but handle it
                pass

    # If we found a good solution, return it
    if best_set is not None:
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

    # Fallback: If local search failed or was too slow, return a valid but larger code
    # Simple greedy without optimization
    selected = []
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered_mask = np.ones(total, dtype=bool)
    
    while np.any(uncovered_mask):
        scores = uncovered_mask[balls].sum(axis=1)
        max_score = np.max(scores)
        if max_score == 0:
            break
        candidates = np.where(scores == max_score)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)
        
    if np.any(uncovered_mask):
        for w in range(total):
            if uncovered_mask[w]:
                selected.append(w)
                cover_count[balls[w]] += 1
                uncovered_mask = (cover_count == 0)

    # Quick removal pass
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected

    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result
