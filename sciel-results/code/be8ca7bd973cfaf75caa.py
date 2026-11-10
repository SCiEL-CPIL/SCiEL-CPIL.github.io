import time
import random
import sys

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
    
    # Precompute neighbors
    # Ball size is 1 + 7*2 = 15
    neighbors = [None] * total_words
    # Use a fast conversion
    # Represent words as integers 0..2186
    # Digits are base-3.
    # To find neighbors, we can precompute the "digit mask" or just iterate.
    # Since total_words is small, precomputation is fine.
    
    # Efficient neighbor generation
    # For each word, generate its 14 neighbors.
    # We can precompute the "powers" of 3.
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    for w in range(total_words):
        ball = [w]
        for i in range(n):
            p = powers[i]
            # Get digit at position i
            digit = (w // p) % 3
            # Neighbor with digit 0
            if digit != 0:
                ball.append(w - digit * p)
            # Neighbor with digit 1
            if digit != 1:
                ball.append(w + (1 - digit) * p)
            # Neighbor with digit 2
            if digit != 2:
                ball.append(w + (2 - digit) * p)
        neighbors[w] = ball
        
        # Also precompute a boolean array or set for fast "is neighbor" check?
        # For local search, we need to check if a candidate covers unique words.
        # Unique words are a small list.
        # Checking `if w in neighbors[c]` is O(15) with list, or O(1) with set.
        # Let's use sets for neighbors for faster lookup during swap search.
        neighbors[w] = set(ball)
        
    # But for iterating neighbors, list is faster.
    # Let's keep both? Or just list and use a global boolean matrix?
    # 2187x2187 matrix is ~4.7M booleans. In Python, a list of lists of bools is heavy.
    # A numpy array would be fast.
    import numpy as np
    # adj[i][j] is True if i covers j (i is neighbor of j or i==j)
    # Actually, we need: does codeword c cover word w? i.e., is w in neighbors[c]?
    # Let's create a numpy boolean matrix `covers` where covers[c, w] is True if c covers w.
    covers = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        for w in neighbors[c]:
            covers[c, w] = True
            
    # Now `covers[c]` is a boolean vector.
    # `cover_count[w]` can be computed as `np.sum(covers[:, w])`? No, that's slow.
    # We maintain `cover_count` manually.
    
    # Greedy Set Cover
    def greedy_set_cover():
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        codewords_list = []
        
        # To speed up greedy, we can precompute the "uncovered mask" for each candidate?
        # No, it changes.
        # We iterate candidates. 2187 candidates * 15 neighbors = ~32k ops per step.
        # ~180 steps. ~6M ops. Fast enough.
        
        available = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            # Random tie-breaking?
            candidates = list(available)
            random.shuffle(candidates)
            
            for c in candidates:
                # Count uncovered neighbors
                # neighbors[c] is a set of ints.
                # We need to count how many are not in `covered`.
                # Using numpy: `np.sum(~covered[list(neighbors[c])])`
                # This creates a list and indexes numpy.
                # Alternative: precomputed list of neighbors.
                # Let's use the list version for speed in this loop.
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == 15:
                        break
            
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available.remove(best_candidate)
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        return codewords_list

    def local_search(code_list, time_limit):
        code_set = set(code_list)
        cover_count = np.zeros(total_words, dtype=np.int8)
        for c in code_set:
            # Add 1 to cover_count for all neighbors of c
            # cover_count[list(neighbors[c])] += 1
            # This is vectorized and fast.
            cover_count[list(neighbors[c])] += 1
            
        start_time = time.time()
        no_improvement_count = 0
        best_size = len(code_list)
        best_code = list(code_list)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            # Iterate in random order
            indices = list(range(len(code_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit:
                    break
                c = code_list[idx]
                # Check if any neighbor has cover_count 1
                # min_count = np.min(cover_count[list(neighbors[c])])
                # if min_count > 1: remove
                # Using python loop might be faster for small list?
                # Let's try numpy min.
                if np.min(cover_count[list(neighbors[c])]) > 1:
                    # Remove
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[list(neighbors[c])] -= 1
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
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find unique words (cover_count == 1)
                # unique_mask = (cover_count == 1)
                # We need neighbors of c that are unique.
                # unique_words = [w for w in neighbors[c] if cover_count[w] == 1]
                unique_words = []
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if unique_words:
                    # Find candidate c' not in code_set that covers all unique_words
                    best_c_prime = -1
                    best_score = -1
                    
                    # Sample candidates? Or scan all?
                    # Scanning all 2187 is fast.
                    # For each c', check if it covers all unique_words.
                    # And score it.
                    
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check coverage of unique_words
                        # all(cover_count[w] > 0 for w in unique_words) is not right, we need c_prime to cover them.
                        # i.e., covers[c_prime, w] is True for all w in unique_words.
                        # Using numpy: np.all(covers[c_prime, unique_words])
                        if not np.all(covers[c_prime, unique_words]):
                            continue
                        
                        # Score:
                        # We want to minimize "rigidity".
                        # Heuristic: cover more words with cover_count == 2 or 1?
                        # If we swap c -> c':
                        # Words in neighbors[c] \ unique_words lose 1 count.
                        # Words in neighbors[c'] \ unique_words gain 1 count.
                        # Words in unique_words stay at count 1 (covered by c' instead of c).
                        
                        # A good swap makes other codewords removable.
                        # So we want to increase cover_count of words that are currently "critical" for other codewords?
                        # No, we want to make words have higher cover count so they are less critical.
                        # Words with cover_count 1 are critical.
                        # If c' covers a word w with cover_count 1 (where w is not in unique_words, so w is covered by some other c_other),
                        # then w becomes cover_count 2. c_other becomes less critical.
                        # So we prefer c' that covers many words with cover_count == 1.
                        
                        # Also, if c' covers words with cover_count 2, they become 3. Also good.
                        # But count 1 is more "valuable" to fix.
                        
                        # Let's score:
                        # score = sum(2 if cover_count[w]==1 else 1 if cover_count[w]==2 else 0 for w in neighbors[c_prime])
                        # But we must exclude unique_words?
                        # For w in unique_words, cover_count is 1. After swap, it's 1. No change in "criticality" of w itself, but the codeword c is gone.
                        # Actually, for w in unique_words, the "criticality" shifts from c to c'.
                        # The score should reflect the change in the rest of the code.
                        
                        # Let's just sum scores for all neighbors of c_prime.
                        # We can vectorize this scoring?
                        # scores = (cover_count[neighbors[c_prime]] == 1) * 2 + (cover_count[neighbors[c_prime]] == 2) * 1
                        # This creates arrays.
                        
                        # Let's stick to python loop for scoring, it's only 15 items.
                        score = 0
                        for w in neighbors[c_prime]:
                            if cover_count[w] == 1:
                                score += 2
                            elif cover_count[w] == 2:
                                score += 1
                                
                        if score > best_score:
                            best_score = score
                            best_c_prime = c_prime
                            if best_score > 8: # Good enough
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        del code_list[idx]
                        code_set.remove(c)
                        cover_count[list(neighbors[c])] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        cover_count[list(neighbors[best_c_prime])] += 1
                        
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Should have been removed
                    pass
            else:
                break
                
            if no_improvement_count > 50:
                # Try to restart with a slightly perturbed code?
                # Or just continue.
                pass

        return best_code

    # Execution
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Greedy
    initial_code = greedy_set_cover()
    
    # 2. Local Search
    final_code = local_search(initial_code, deadline)
    
    # 3. Final Cleanup
    code_set = set(final_code)
    cover_count = np.zeros(total_words, dtype=np.int8)
    for c in code_set:
        cover_count[list(neighbors[c])] += 1
        
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
            if np.min(cover_count[list(neighbors[c])]) > 1:
                del final_list[idx]
                code_set.remove(c)
                cover_count[list(neighbors[c])] -= 1
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
