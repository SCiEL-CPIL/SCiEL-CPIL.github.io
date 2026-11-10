import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        # Simple greedy set cover
        total_words = q ** n
        
        def int_to_tuple(w):
            digits = []
            for _ in range(n):
                digits.append(w % q)
                w //= q
            return tuple(reversed(digits))
        
        def tuple_to_int(t):
            w = 0
            mult = 1
            for d in reversed(t):
                w += d * mult
                mult *= q
            return w

        neighbors = [None] * total_words
        for w in range(total_words):
            t = int_to_tuple(w)
            ball = {w}
            for i in range(n):
                orig = t[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_t = list(t)
                    new_t[i] = new_val
                    ball.add(tuple_to_int(tuple(new_t)))
            neighbors[w] = list(ball)
        
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available_candidates = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            # Vectorized gain calculation is hard in pure python fallback, so loop
            for c in available_candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available_candidates.remove(best_candidate)
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        
        result = [list(int_to_tuple(c)) for c in codewords_list]
        return result

    # Specialized solver for q=3, n=7, r=1
    total_words = 2187
    
    # Precompute neighbors as list of lists of ints
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    neighbors_list = [None] * total_words
    for w in range(total_words):
        ball = [w]
        for i in range(n):
            p = powers[i]
            digit = (w // p) % 3
            if digit != 0:
                ball.append(w - digit * p)
            if digit != 1:
                ball.append(w + (1 - digit) * p)
            if digit != 2:
                ball.append(w + (2 - digit) * p)
        neighbors_list[w] = ball

    # Precompute covers matrix: covers[c, w] is True if c covers w
    # Using a sparse representation or dense bool array?
    # Dense bool array is 2187x2187 * 1 byte ~ 4.7 MB. Fast access.
    covers = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        covers[c, neighbors_list[c]] = True
    
    # Precompute neighbor indices as numpy arrays for vectorized ops
    neighbors_np = [np.array(neighbors_list[c], dtype=np.int32) for c in range(total_words)]

    def greedy_set_cover():
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        codewords_list = []
        available = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            
            # To speed up, we can sample if available is large, 
            # but 2187 is small enough to check all if we vectorize the scoring.
            
            # We iterate through available candidates. 
            # Score is number of newly covered points.
            # new_covered = covers[c] & ~covered
            # score = sum(new_covered)
            
            # Optimization: If we maintain a list of candidates, 
            # we can't easily vectorize the outer loop without building a matrix.
            # However, we can just loop. 2187 * 2187 ops is ~5M ops. 
            # In numpy, a sum of 2187 elements is very fast.
            
            # We will iterate through available candidates.
            # To make it faster, we can shuffle and break early if we find a perfect score (15).
            candidates = list(available)
            random.shuffle(candidates)
            
            for c in candidates:
                # Vectorized gain calculation
                # covers[c] is a row of the covers matrix
                gain = np.sum(covers[c] & ~covered)
                
                if gain > best_score:
                    best_score = gain
                    best_candidate = c
                    if best_score == 15: # Max possible gain
                        break
            
            if best_candidate == -1:
                break
            
            codewords_list.append(best_candidate)
            available.remove(best_candidate)
            
            # Update covered
            newly_covered = covers[best_candidate] & ~covered
            covered[neighbors_np[best_candidate]] = True 
            # Wait, covers[best_candidate] IS the neighbors.
            # So covered[neighbors_np[best_candidate]] = True is equivalent to covered |= covers[best_candidate]
            # But we need to update uncovered_count
            # Let's just update covered and recount? No, recount is expensive.
            # newly_covered count is best_score.
            uncovered_count -= best_score
            
        return codewords_list

    def local_search(code_list, time_limit):
        code_set = set(code_list)
        cover_count = np.zeros(total_words, dtype=np.int16)
        for c in code_set:
            cover_count[neighbors_np[c]] += 1
            
        start_time = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        no_improvement_count = 0
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            indices = list(range(len(code_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit:
                    break
                c = code_list[idx]
                # If all neighbors are covered by at least 2 codewords, we can remove c
                if np.min(cover_count[neighbors_np[c]]) > 1:
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_np[c]] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement_count = 0
                    break
            
            if improved:
                continue
                
            # 2. Try Swaps
            if code_list:
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find words covered ONLY by c
                # unique_words = neighbors[c] where cover_count == 1
                unique_mask = (cover_count[neighbors_np[c]] == 1)
                unique_words = neighbors_np[c][unique_mask]
                
                if len(unique_words) > 0:
                    # We need to find a candidate c' NOT in code_set 
                    # such that c' covers ALL unique_words.
                    
                    # Vectorized check:
                    # For each candidate c', check if covers[c', unique_words] are all True.
                    # This is equivalent to: np.all(covers[:, unique_words], axis=1)
                    
                    # This creates a boolean vector of length 2187.
                    candidate_mask = np.all(covers[:, unique_words], axis=1)
                    
                    # Exclude existing codewords
                    for c_exist in code_set:
                        candidate_mask[c_exist] = False
                    
                    valid_candidates = np.where(candidate_mask)[0]
                    
                    if len(valid_candidates) > 0:
                        # We found valid candidates.
                        # Choose the one that maximizes "newly covered" or minimizes "newly uncovered"?
                        # Actually, we remove c and add c'.
                        # Net change in size is 0.
                        # We want to make future removals easier?
                        # Score: How many NEW words does c' cover that were previously covered by 0? (None, we are maintaining cover)
                        # Score: How many words with cover_count == 1 does c' cover? (Increases robustness)
                        
                        best_c_prime = -1
                        best_score = -1000
                        
                        # Penalty: c' should not cover words that are currently covered ONLY by c?
                        # No, c' MUST cover unique_words (by construction).
                        # So those words will have cover_count 1 (from c') instead of 1 (from c).
                        
                        # Heuristic: Prefer c' that covers many words with low cover count (2 or 3).
                        for c_prime in valid_candidates:
                            # Calculate score
                            # Words covered by c'
                            n_prime = neighbors_np[c_prime]
                            # Count how many have cover_count 1, 2, etc.
                            # Since c' covers unique_words, those counts don't change relative to c.
                            # We look at n_prime \ unique_words
                            
                            # Simple heuristic: maximize number of neighbors with cover_count == 1
                            # This might not be right.
                            
                            # Better: Minimize the "fragility" of the new code.
                            # Let's just pick one at random or the one with highest total coverage?
                            # Actually, any valid swap maintains the cover.
                            # The goal is to find a state from which we can REMOVE words.
                            # So we want to increase the minimum cover count.
                            
                            # Score: Sum of cover_count for neighbors of c_prime?
                            score = np.sum(cover_count[n_prime])
                            
                            if score > best_score:
                                best_score = score
                                best_c_prime = c_prime
                            
                        if best_c_prime != -1:
                            # Perform swap
                            del code_list[idx]
                            code_set.remove(c)
                            cover_count[neighbors_np[c]] -= 1
                            
                            code_list.append(best_c_prime)
                            code_set.add(best_c_prime)
                            cover_count[neighbors_np[best_c_prime]] += 1
                            
                            improved = True
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                break
            
            # 3. Perturbation if stuck
            if no_improvement_count > 10 and code_list:
                # Remove a random subset
                num_to_remove = max(1, len(code_list) // 10)
                remove_indices = random.sample(range(len(code_list)), num_to_remove)
                remove_indices.sort(reverse=True)
                
                for idx in remove_indices:
                    c = code_list[idx]
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_np[c]] -= 1
                
                # Re-fill greedily
                covered = (cover_count > 0)
                uncovered_count = int(np.sum(~covered))
                
                available = set(range(total_words)) - code_set
                while uncovered_count > 0 and time.time() < time_limit:
                    best_candidate = -1
                    best_score = -1
                    
                    candidates = list(available)
                    random.shuffle(candidates)
                    
                    for c in candidates:
                        gain = np.sum(covers[c] & ~covered)
                        if gain > best_score:
                            best_score = gain
                            best_candidate = c
                            if best_score == 15:
                                break
                    
                    if best_candidate == -1:
                        break
                    
                    code_list.append(best_candidate)
                    code_set.add(best_candidate)
                    uncovered_count -= best_score
                    cover_count[neighbors_np[best_candidate]] += 1
                    available.remove(best_candidate)
                
                no_improvement_count = 0

        return best_code

    # Execution
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Run multiple greedy passes
    best_initial = None
    best_initial_size = float('inf')
    
    while time.time() < deadline - 40.0:  # Leave 40s for local search
        initial_code = greedy_set_cover()
        if len(initial_code) < best_initial_size:
            best_initial_size = len(initial_code)
            best_initial = initial_code
            if best_initial_size <= 188:
                break
    
    if best_initial is None:
        best_initial = greedy_set_cover()
        
    # 2. Local Search
    final_code = local_search(best_initial, deadline - 5.0)
    
    # 3. Final Cleanup
    code_set = set(final_code)
    cover_count = np.zeros(total_words, dtype=np.int16)
    for c in code_set:
        cover_count[neighbors_np[c]] += 1
        
    final_list = list(code_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(final_list)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            c = final_list[idx]
            if np.min(cover_count[neighbors_np[c]]) > 1:
                del final_list[idx]
                code_set.remove(c)
                cover_count[neighbors_np[c]] -= 1
                improved = True

    # Convert to list of lists
    result = []
    for c in final_list:
        digits = []
        w = c
        for _ in range(n):
            digits.append(w % q)
            w //= q
        result.append(digits[::-1])
        
    return result
