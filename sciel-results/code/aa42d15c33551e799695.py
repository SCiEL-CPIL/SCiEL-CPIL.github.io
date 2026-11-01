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
    
    # Precompute neighbors (balls) for all 6561 words
    # A ball of radius 1 has 1 + n*(q-1) = 1 + 8*2 = 17 words.
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = []
    for w in range(total):
        ball = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                ball.append(new_w)
        balls.append(ball)
    
    balls_np = np.array(balls, dtype=np.int32)
    
    # Precompute "conflict" information? 
    # For 2-exchange, we need to know if two codewords can be removed and replaced.
    
    best_size = total
    best_set = None

    # Helper: Remove redundant codewords (1-exchange removal)
    def remove_redundant(codewords, cover_count=None):
        if not codewords:
            return [], np.zeros(total, dtype=np.int32)
        
        current_set = set(codewords)
        if cover_count is None:
            cover_count = np.zeros(total, dtype=np.int32)
            for c in current_set:
                cover_count[balls_np[c]] += 1
        else:
            # Ensure cover_count matches current_set
            cover_count = np.zeros(total, dtype=np.int32)
            for c in current_set:
                cover_count[balls_np[c]] += 1

        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            # Find a codeword c such that all words in its ball have count > 1
            # i.e., min(cover_count[ball(c)]) > 1
            
            # Optimization: If global min cover count >= 2, any codeword is redundant.
            min_count_global = np.min(cover_count)
            if min_count_global >= 2:
                c_to_remove = next(iter(current_set))
                current_set.remove(c_to_remove)
                cover_count[balls_np[c_to_remove]] -= 1
                changed = True
                continue
            
            # Otherwise, find c where min(cover_count[ball(c)]) > 1
            # We can check this for each c in current_set.
            # To speed up, iterate.
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline - 0.5:
                    break
                min_in_ball = np.min(cover_count[balls_np[c]])
                if min_in_ball > 1:
                    current_set.remove(c)
                    cover_count[balls_np[c]] -= 1
                    changed = True
                    break
                    
        return list(current_set), cover_count

    # Helper: 1-Exchange Local Search (Swap one codeword for another)
    def local_search_1exchange(codewords, cover_count):
        current_set, cover_count = remove_redundant(codewords, cover_count)
        if not current_set:
            return [], cover_count
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline - 0.5:
                    break
                
                c_ball = balls_np[c]
                # Words covered ONLY by c
                U_c = c_ball[cover_count[c_ball] == 1]
                
                if len(U_c) == 0:
                    # Redundant
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break # Restart loop with new set
                
                # Try to find c' not in current_set such that U_c subset of ball(c')
                # And ideally, c' covers many other uncovered/low-count words?
                # Actually, we just need validity.
                # To minimize size, we just swap. Size stays same.
                # But maybe we can find a c' that allows further reduction?
                # For now, just find any valid c'.
                
                # Optimization: If |U_c| is large, it's hard to find c'.
                # If |U_c| == 1, easy.
                
                # Iterate through non-selected codewords
                # To avoid O(N^2) in worst case, we can be smart.
                # But N=6561 is small enough if we break early.
                
                found_swap = False
                # Randomize order of c' search?
                # Or just linear scan.
                
                # Heuristic: Look for c' that is "far" from current set? 
                # Or just check all.
                
                for c_prime in range(total):
                    if c_prime in current_set:
                        continue
                    
                    # Check if c_prime covers U_c
                    # Vectorized:
                    # Create mask of U_c in ball(c_prime)
                    # Since U_c is small, loop is fine.
                    valid = True
                    for w in U_c:
                        if w not in balls_np[c_prime]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        current_set.remove(c)
                        cover_count[c_ball] -= 1
                        
                        current_set.add(c_prime)
                        cover_count[balls_np[c_prime]] += 1
                        
                        improved = True
                        found_swap = True
                        break # Restart outer loop
                    
                if found_swap:
                    break # Restart while loop with new set
                    
        return current_set, cover_count

    # Helper: 2-Exchange Local Search (Swap two codewords for two others)
    # This is more expensive but escapes deeper minima.
    def local_search_2exchange(codewords, cover_count):
        current_set, cover_count = remove_redundant(codewords, cover_count)
        if not current_set:
            return [], cover_count
        
        # First do 1-exchange to get close to local optimum
        current_set, cover_count = local_search_1exchange(list(current_set), cover_count)
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            
            candidates = list(current_set)
            random.shuffle(candidates)
            
            # Try pairs (c1, c2)
            # To limit complexity, we can limit the number of pairs tried per iteration.
            # Or just try all pairs if N is small enough? N ~ 400. Pairs ~ 80k.
            # Each pair check involves finding c3, c4.
            # This might be too slow.
            
            # Strategy: Pick a random c1. Find c2 such that U_c1 U U_c2 is coverable by two new words.
            
            c1 = candidates[0]
            c1_ball = balls_np[c1]
            U_c1 = c1_ball[cover_count[c1_ball] == 1]
            
            if len(U_c1) == 0:
                # Should have been removed by 1-exchange, but just in case
                current_set.remove(c1)
                cover_count[c1_ball] -= 1
                improved = True
                break
            
            # Try to find c2
            # We need to cover U_c1 and U_c2 with c3 and c4.
            # This is a set packing/covering problem.
            
            # Simplified 2-exchange:
            # Remove c1.
            # If U_c1 is not empty, we need to cover it.
            # Find c3 that covers U_c1.
            # Then see if we can find c2, c4 such that removing c1, c2 and adding c3, c4 is valid.
            
            # Let's try a simpler heuristic:
            # Pick c1.
            # Find all c3 such that U_c1 subset ball(c3).
            # For each c3, tentatively add c3.
            # Now some words might be uncovered (those that were only covered by c1 and not by c3? No, U_c1 is covered by c3).
            # But adding c3 might make some other words redundant?
            # Actually, the standard move is: Remove c1, c2. Add c3, c4.
            # Condition: Every word covered by c1 or c2 must be covered by c3 or c4 or other existing codewords.
            
            # Given time constraints, a full 2-exchange is risky.
            # Let's stick to a "Perturbation" strategy:
            # Randomly remove k codewords (e.g., 2 or 3) and re-add them greedily to cover the holes.
            
            pass # Disabled complex 2-exchange for safety, relying on restarts and 1-exchange + perturbation

    # Helper: Perturbation (Remove k, re-add greedily)
    def perturb(codewords, cover_count, k=2):
        current_set, cover_count = remove_redundant(codewords, cover_count)
        if not current_set:
            return [], cover_count
        
        # Remove k random codewords
        to_remove = random.sample(list(current_set), min(k, len(current_set)))
        for c in to_remove:
            current_set.remove(c)
            cover_count[balls_np[c]] -= 1
        
        # Re-add greedily to cover uncovered words
        uncovered = np.where(cover_count == 0)[0]
        
        while len(uncovered) > 0 and time.time() < deadline - 0.5:
            # Find best candidate to cover most uncovered
            # counts[c] = number of uncovered words in ball(c)
            counts = np.zeros(total, dtype=np.int32)
            for w in uncovered:
                # Add 1 to all codewords that cover w
                # This is slow if done naively.
                # Better: For each candidate c, count how many of its ball are in uncovered.
                # Vectorized:
                # counts = np.sum(uncovered_mask[balls_np], axis=1)
                # But uncovered is a list of indices.
                # Let's create a mask.
                pass
            
            # Efficient way:
            # uncovered_mask = np.zeros(total, dtype=bool)
            # uncovered_mask[uncovered] = True
            # counts = uncovered_mask[balls_np].sum(axis=1)
            
            uncovered_mask = np.zeros(total, dtype=bool)
            uncovered_mask[uncovered] = True
            counts = uncovered_mask[balls_np].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break # Should not happen if graph is connected, but just in case
            
            candidates = np.where(counts == max_count)[0]
            # Prefer candidates not in current_set?
            # If a candidate is already in current_set, adding it does nothing (count is already >=1).
            # So we must pick one NOT in current_set.
            
            valid_candidates = [c for c in candidates if c not in current_set]
            if not valid_candidates:
                # If no new candidate covers any uncovered word, we are stuck.
                # This implies the removed codewords were essential and no single new one can replace them.
                # We might need to add multiple.
                # For simplicity, pick the one with max count even if it's in set? No, that does nothing.
                # If valid_candidates is empty, it means no single codeword covers any uncovered word.
                # This is impossible because each uncovered word has a ball of 17 candidates.
                # Unless those candidates are all in current_set?
                # If c is in current_set, it covers its ball. So if w is uncovered, no c in current_set covers w.
                # So all candidates covering w are NOT in current_set.
                # So valid_candidates should not be empty.
                
                # Wait, if c is in current_set, it covers w. So w would not be uncovered.
                # So yes, valid_candidates must be non-empty.
                
                pass
            
            best_c = valid_candidates[np.random.randint(len(valid_candidates))]
            current_set.add(best_c)
            cover_count[balls_np[best_c]] += 1
            
            # Update uncovered
            # New uncovered = old uncovered \ ball(best_c)
            # Or just recompute.
            uncovered = np.where(cover_count == 0)[0]
        
        return current_set, cover_count

    # Run multiple restarts
    # Strategy:
    # 1. Start with a good algebraic base (e.g., 81 words from tetracode product? No, 9x9=81 is too small? 
    #    K3(4,1)=9. Direct sum gives code of size 81. 
    #    Does it cover H(8,3)? 
    #    Distance in direct sum is sum of distances. 
    #    If d1 <= 1 and d2 <= 1, total d <= 2. 
    #    We need radius 1. So direct sum of radius 1 codes gives radius 2. 
    #    So 81 words is NOT a covering code for radius 1. 
    #    We need a code where every word is within 1.
    #    So we need to start with a valid code.
    #    Greedy is the safest start.
    
    num_restarts = 200 # High number of restarts, each short
    
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Greedy Initialization
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 2.0:
                break
            
            # Vectorized count of uncovered words covered by each candidate
            counts = uncovered_mask[balls_np].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls_np[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue
            
        # Local Search 1
        refined_set, cover_count = local_search_1exchange(selected, cover_count)
        
        # Perturbation and Local Search 2
        # Try a few perturbations
        for p in range(3):
            if time.time() > deadline - 2.0:
                break
            perturbed_set, pc = perturb(list(refined_set), cover_count.copy(), k=2)
            refined_set, cover_count = local_search_1exchange(perturbed_set, pc)
            
        if len(refined_set) < best_size:
            best_size = len(refined_set)
            best_set = list(refined_set)
            
            # If we hit a very good solution, do more local search on it
            if best_size < 450:
                for extra in range(10):
                    if time.time() > deadline - 2.0:
                        break
                    refined_set, cover_count = local_search_1exchange(list(refined_set), cover_count)
                    if len(refined_set) < best_size:
                        best_size = len(refined_set)
                        best_set = list(refined_set)

    # Fallback
    if best_set is None:
        # Return a valid code, even if large
        # Just use the greedy result from the last run or a simple product
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls_np].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0: break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            cover_count[balls_np[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected
        best_size = len(best_set)

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
