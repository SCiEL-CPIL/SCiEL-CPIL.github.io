import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks
    if q == 3 and n >= 4:
        # Tetracode based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors (balls) for all 6561 words
    weights = [3 ** (n - 1 - i) for i in range(n)]

    balls = np.zeros((total, num_neighbors), dtype=np.int32)
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

    # Precompute ball sets for O(1) membership check
    ball_sets = [set(balls[w].tolist()) for w in range(total)]

    best_size = total
    best_set = None

    def remove_redundant(codewords):
        if not codewords:
            return []
        
        current_set = set(codewords)
        cover_count = np.zeros(total, dtype=np.int32)
        
        # Initial count
        for c in current_set:
            cover_count[balls[c]] += 1
            
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            # Find a codeword c such that all words in its ball have count > 1
            # i.e., no word in ball(c) has count == 1.
            
            # If min_count >= 2, any codeword is removable
            if np.min(cover_count) >= 2:
                c_to_remove = next(iter(current_set))
                current_set.remove(c_to_remove)
                cover_count[balls[c_to_remove]] -= 1
                changed = True
                continue
            
            # If min_count == 1, we need to find a codeword c that does NOT cover any word with count 1.
            unique_words_mask = (cover_count == 1)
            
            # Iterate through current_set to find a removable codeword
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                # Check if ball(c) intersects U (words with count 1)
                if not np.any(unique_words_mask[balls[c]]):
                    current_set.remove(c)
                    cover_count[balls[c]] -= 1
                    changed = True
                    break
                    
        return list(current_set)

    def local_search(codewords):
        current_set = set(remove_redundant(codewords))
        if not current_set:
            return []
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[balls[c]] += 1
            
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Try to find a codeword to swap or remove
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline - 1.0:
                    break
                if c not in current_set:
                    continue
                
                # Find words covered ONLY by c
                c_ball = balls[c]
                U_c = c_ball[cover_count[c_ball] == 1]
                
                if len(U_c) == 0:
                    # c is redundant
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break # Restart loop with new set
                
                # If U_c is not empty, we must cover U_c if we remove c.
                # We look for a candidate c' not in current_set such that U_c subset of ball(c').
                
                found_swap = False
                U_c_list = list(U_c)
                
                # Strategy: Intersect the balls of all words in U_c to get potential candidates.
                # A candidate c' must be in ball(w) for all w in U_c.
                # Since U_c is usually small (1 or 2), this is fast.
                
                # Start with candidates from the first word in U_c
                candidate_set = set(ball_sets[U_c_list[0]])
                for w in U_c_list[1:]:
                    candidate_set.intersection_update(ball_sets[w])
                
                # Remove existing codewords
                candidate_set -= current_set
                
                if candidate_set:
                    # Pick a candidate that maximizes the number of NEW words covered
                    # Or simply random. Let's try a heuristic: prefer candidates that cover many words
                    # which are currently covered by exactly 1 codeword (not c) or 0?
                    # Actually, since we are removing c, the "new" coverage is just ball(c_prime) \ ball(c).
                    # But we want to enable future removals.
                    # Let's just pick the one that covers the most words with count == 1 (excluding c's ball? no, c is removed).
                    # Simpler: random choice is okay, but let's try to pick one that covers many "tight" words.
                    
                    # Heuristic: Maximize sum of (1/cover_count) for words in ball(c_prime)?
                    # Or just count of words in ball(c_prime) that are in U_c (already guaranteed) + count of other words?
                    # Let's stick to random for speed, or maybe max gain?
                    # Given the small number of candidates, we can check gain.
                    
                    best_prime = None
                    max_gain = -1
                    # Limit check to first 10 candidates for speed?
                    cand_list = list(candidate_set)
                    if len(cand_list) > 10:
                        cand_list = random.sample(cand_list, 10)
                        
                    for c_prime in cand_list:
                        # Gain is number of words in ball(c_prime) that are NOT in ball(c)
                        # But more importantly, does it help other codewords become redundant?
                        # Let's just use a simple score: number of words in ball(c_prime)
                        # that are currently covered by exactly 1 codeword (which will be 2 after swap)
                        # vs words covered by 0 (impossible) or >1.
                        # Actually, the main benefit is if it allows ANOTHER codeword to be removed.
                        # Hard to predict. Let's just pick randomly.
                        pass
                    
                    c_prime = random.choice(cand_list)
                    
                    # Perform swap
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    
                    current_set.add(c_prime)
                    cover_count[balls[c_prime]] += 1
                    
                    improved = True
                    found_swap = True
                    break # Restart outer loop
                    
                if found_swap:
                    break # Restart while loop with new set
                    
        return list(current_set)

    # Run multiple restarts
    num_restarts = 100 
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Greedy Initialization
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break
            
            # Vectorized count of uncovered words covered by each candidate
            # counts[c] = number of uncovered words in ball(c)
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue
            
        # Local Search
        refined_set = local_search(selected)
        
        if len(refined_set) < best_size:
            best_size = len(refined_set)
            best_set = refined_set
            
            # If we reach a very good size, we can break early or continue polishing
            if best_size <= 486:
                break

    # Fallback
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
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
