import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
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
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for 120s limit

    # Precompute neighbors
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

    # Precompute neighbor arrays for vectorized operations
    # Shape: (2187, 15)
    neighbor_arrays = np.array(neighbors_list, dtype=np.int32)
    
    # Precompute covers matrix (2187, 2187) boolean
    # covers[c, w] is True if c is in the ball of w (i.e., w covers c? No, c covers w)
    # Actually, let's stick to the definition: covers[c, w] = (w in neighbors_list[c])
    covers = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        covers[c, neighbors_list[c]] = True

    def generate_linear_code():
        # Generate a random linear code of dimension 4 over GF(3)
        # Generator matrix G is 4x7
        G = np.random.randint(0, 3, size=(4, 7))
        # Ensure rank is 4 (highly likely for random, but let's check)
        # For small matrices, we can just use it. If rank < 4, size is smaller, which is fine for a seed.
        # But we want size 81.
        # Let's just use the first 4 rows if rank is sufficient? 
        # Actually, for a seed, any subset of a linear code works.
        # Let's generate all 3^4 = 81 codewords.
        codewords = []
        for i in range(81):
            coeffs = []
            temp = i
            for _ in range(4):
                coeffs.append(temp % 3)
                temp //= 3
            # coeffs is [c0, c1, c2, c3]
            # codeword = c0*G[0] + c1*G[1] + c2*G[2] + c3*G[3]
            cw = np.zeros(7, dtype=np.int32)
            for j in range(4):
                cw += coeffs[j] * G[j]
            cw = cw % 3
            # Convert to integer
            val = 0
            for k in range(7):
                val += cw[k] * powers[k]
            codewords.append(val)
        return list(set(codewords)) # Remove duplicates if rank < 4

    def greedy_set_cover(seed_codes=None):
        if seed_codes is None:
            seed_codes = []
        
        # Initialize coverage
        cover_count = np.zeros(total_words, dtype=np.int16)
        code_set = set(seed_codes)
        
        # Add seed codes
        for c in code_set:
            cover_count[neighbors_list[c]] += 1
            
        uncovered_mask = (cover_count == 0)
        uncovered_count = np.sum(uncovered_mask)
        
        codewords_list = list(code_set)
        
        # Precompute scores for all candidates? No, it changes.
        # But we can use vectorization to find the best candidate quickly.
        
        # available candidates are all words not in code_set
        # We want to pick c that maximizes sum(~uncovered_mask[neighbors_list[c]])
        
        # Let's create a matrix of scores?
        # scores[c] = sum(~uncovered_mask[neighbors_list[c]])
        # This is a matrix-vector product if we use covers.
        # scores = covers @ ~uncovered_mask
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            # Calculate score for each candidate
            # ~uncovered_mask is boolean. covers is boolean.
            # We need to count how many uncovered words are covered by each candidate.
            # scores = covers.dot(~uncovered_mask.astype(np.int32))
            
            # However, we should not pick candidates already in code_set.
            # We can set their scores to -1.
            
            scores = covers.dot(~uncovered_mask.astype(np.int32))
            scores[list(code_set)] = -1
            
            best_candidate = np.argmax(scores)
            best_score = scores[best_candidate]
            
            if best_score <= 0:
                break # No improvement possible (should not happen if uncovered_count > 0)
                
            codewords_list.append(best_candidate)
            code_set.add(best_candidate)
            
            # Update coverage
            new_covered = neighbors_list[best_candidate]
            for w in new_covered:
                if cover_count[w] == 0:
                    cover_count[w] = 1
                    uncovered_count -= 1
                else:
                    cover_count[w] += 1
            
            # Update mask
            uncovered_mask[new_covered] = (cover_count[new_covered] == 0)

        return codewords_list

    def local_search(code_list, time_limit):
        code_set = set(code_list)
        cover_count = np.zeros(total_words, dtype=np.int16)
        for c in code_set:
            cover_count[neighbors_list[c]] += 1
            
        start_time_ls = time.time()
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
                # Check if any neighbor has cover_count 1
                if np.min(cover_count[neighbors_list[c]]) > 1:
                    # Remove
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_list[c]] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement_count = 0
                    break # Restart scan
            
            if improved:
                continue
                
            # 2. Try Swaps
            if code_list:
                # Pick a random codeword to potentially swap out
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find unique words (cover_count == 1) for c
                unique_words = [w for w in neighbors_list[c] if cover_count[w] == 1]
                
                if unique_words:
                    # Find candidate c' not in code_set that covers all unique_words
                    best_c_prime = -1
                    best_score = -1
                    
                    # To speed this up, we can pre-filter candidates?
                    # Or just scan all. 2187 is small.
                    # But checking `covers[c_prime, unique_words]` is fast with numpy.
                    
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check if c_prime covers all unique_words
                        if not np.all(covers[c_prime, unique_words]):
                            continue
                        
                        # Score the swap
                        # We want to increase the "redundancy" of other codewords.
                        # Words with cover_count 1 are "critical".
                        
                        # Let's calculate the "damage" and "gain".
                        # Damage: words in neighbors[c] \ unique_words that have count 2. They will become 1.
                        # Gain: words in neighbors[c'] that have count 1 or 2. They will become 2 or 3.
                        
                        penalty = 0
                        for w in neighbors_list[c]:
                            if w not in unique_words and cover_count[w] == 2:
                                penalty += 2
                        
                        score = 0
                        for w in neighbors_list[c_prime]:
                            if cover_count[w] == 1:
                                score += 2
                            elif cover_count[w] == 2:
                                score += 1
                                
                        net_score = score - penalty
                        
                        if net_score > best_score:
                            best_score = net_score
                            best_c_prime = c_prime
                            if best_score > 10: # Good enough
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        del code_list[idx]
                        code_set.remove(c)
                        cover_count[neighbors_list[c]] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        cover_count[neighbors_list[best_c_prime]] += 1
                        
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Should have been removed, but maybe floating point or logic error?
                    # If no unique words, it should be removable.
                    no_improvement_count += 1
            else:
                break
            
            # 3. Perturbation if stuck
            if no_improvement_count > 20 and code_list:
                # Remove a random few codewords
                num_to_remove = max(1, len(code_list) // 10)
                remove_indices = random.sample(range(len(code_list)), num_to_remove)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    c = code_list[idx]
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_list[c]] -= 1
                
                # Re-fill greedily
                covered = (cover_count > 0)
                uncovered_count = np.sum(~covered)
                
                # Greedy fill
                available = set(range(total_words)) - code_set
                while uncovered_count > 0 and time.time() < time_limit:
                    best_candidate = -1
                    best_score = -1
                    
                    # Vectorized greedy step for refill
                    scores = covers.dot((~covered).astype(np.int32))
                    scores[list(code_set)] = -1
                    
                    candidate = np.argmax(scores)
                    score = scores[candidate]
                    if score <= 0:
                        break
                        
                    code_list.append(candidate)
                    code_set.add(candidate)
                    for w in neighbors_list[candidate]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                        cover_count[w] += 1
                    
                    no_improvement_count = 0

        return best_code

    # Execution
    best_initial = None
    best_initial_size = float('inf')
    
    # 1. Run multiple iterations with different linear code seeds
    while time.time() < deadline - 10.0: # Leave 10s for local search on best
        seed = generate_linear_code()
        initial_code = greedy_set_cover(seed)
        if len(initial_code) < best_initial_size:
            best_initial_size = len(initial_code)
            best_initial = initial_code
            # If we find a very good one, we can break early?
            # 186 is the target. If we find < 190, it's good.
            if best_initial_size <= 188:
                break
        
    # 2. Local Search on the best initial code
    if best_initial is None:
        best_initial = greedy_set_cover()
        
    final_code = local_search(best_initial, deadline)
    
    # 3. Final Cleanup
    code_set = set(final_code)
    cover_count = np.zeros(total_words, dtype=np.int16)
    for c in code_set:
        cover_count[neighbors_list[c]] += 1
        
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
            if np.min(cover_count[neighbors_list[c]]) > 1:
                del final_list[idx]
                code_set.remove(c)
                cover_count[neighbors_list[c]] -= 1
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

if __name__ == "__main__":
    # Quick test
    res = construct(3, 7, 1)
    print(f"Length: {len(res)}")
