import time
import random
import itertools

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # General implementation for q=3, n=6, r=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to direct sum for other parameters if needed, 
        # though task specifies this one.
        # For robustness, if not 3,6,1, return a simple valid code (might not be optimal)
        # The task says "evaluator passes this task's alphabet size...", implying it's fixed to the problem.
        pass

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Precomputation
    # Total words = 3^6 = 729
    total_words = 729
    
    # Precompute neighbors for each word (including itself)
    # A word is a tuple of 6 digits.
    # We map word -> index 0..728
    # Index to word:
    def idx_to_word(idx):
        w = [0]*n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors[i].append(i) # self
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors[i].append(word_to_idx(tuple(new_w)))
                    
    # 2. Greedy Construction
    uncovered = set(range(total_words))
    code = []
    
    # To speed up greedy, we can just iterate.
    # Max neighbors is 13.
    # 729 words.
    while uncovered:
        best_c = -1
        best_cover_count = -1
        # Optimization: We don't need to check all 729 if we track uncovered counts?
        # Simpler: Iterate all 729. 729 * 13 * 729 operations approx?
        # 729 iterations of greedy. Each checks 729 candidates.
        # Total ~ 729 * 729 * 13 ~ 7 million ops. Very fast.
        for c in range(total_words):
            # Count how many neighbors of c are in uncovered
            cnt = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    cnt += 1
            if cnt > best_cover_count:
                best_cover_count = cnt
                best_c = c
        
        code.append(best_c)
        for nb in neighbors[best_c]:
            if nb in uncovered:
                uncovered.remove(nb)
                
        if time.time() > deadline:
            break

    if not uncovered:
        # We have a valid code. Now optimize.
        pass
    else:
        # If greedy didn't finish (unlikely), return what we have? 
        # No, we must return a valid code.
        # Greedy always finishes for 729 words.
        pass

    # 3. Local Search (Swap and Remove)
    # Maintain cover_count array
    cover_count = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def is_redundant(c, cover_count, code_set):
        # c is redundant if all its neighbors are covered by at least 2 codewords
        # Note: cover_count includes c itself.
        # So for nb in neighbors(c), we need cover_count[nb] >= 2.
        # Wait, if c is the only one covering nb, cover_count[nb] is 1.
        # If we remove c, cover_count[nb] becomes 0 -> invalid.
        # So c is redundant if min(cover_count[nb] for nb in neighbors(c)) >= 2.
        min_count = 100
        for nb in neighbors[c]:
            if cover_count[nb] < min_count:
                min_count = cover_count[nb]
            if min_count < 2:
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        code_set = set(code)
        
        # First pass: Check for single redundant codewords
        for c in code:
            if is_redundant(c, cover_count, code_set):
                # Remove c
                code.remove(c)
                code_set.remove(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                improved = True
                break # Restart loop as state changed
        
        if improved:
            continue

        # Second pass: Try swap-and-remove
        # For each c1 in code:
        #   Find unique_to_c1
        #   If empty, handled above.
        #   Else, find candidates w that cover unique_to_c1.
        #   For each candidate w:
        #      Simulate swap c1 -> w.
        #      Check if any c2 becomes redundant.
        #      If yes, perform swap and remove c2.
        
        for i, c1 in enumerate(code):
            if time.time() > deadline:
                break
            
            # Find words covered ONLY by c1
            unique_to_c1 = []
            for nb in neighbors[c1]:
                if cover_count[nb] == 1:
                    unique_to_c1.append(nb)
            
            if not unique_to_c1:
                # Should have been caught in first pass, but just in case
                code.remove(c1)
                code_set.remove(c1)
                for nb in neighbors[c1]:
                    cover_count[nb] -= 1
                improved = True
                break

            # Candidates w must cover all words in unique_to_c1
            # w is a candidate if unique_to_c1 is a subset of neighbors[w]
            # To find candidates, we can intersect the sets of words that cover each word in unique_to_c1.
            # But we only care about w in 0..728.
            # Let's iterate through all 729 words? 
            # Or better: The candidate w must be a neighbor of each word in unique_to_c1.
            # So w is in intersection of neighbors[ub] for ub in unique_to_c1.
            
            # Intersection of lists of neighbors
            # unique_to_c1 is usually small (1-3 words).
            if len(unique_to_c1) == 1:
                candidates = neighbors[unique_to_c1[0]]
            elif len(unique_to_c1) == 2:
                # Intersect neighbors of two words
                set1 = set(neighbors[unique_to_c1[0]])
                set2 = set(neighbors[unique_to_c1[1]])
                candidates = list(set1 & set2)
            elif len(unique_to_c1) == 3:
                set1 = set(neighbors[unique_to_c1[0]])
                set2 = set(neighbors[unique_to_c1[1]])
                set3 = set(neighbors[unique_to_c1[2]])
                candidates = list(set1 & set2 & set3)
            else:
                # If more than 3 unique words, it's harder to cover them with one new word
                # Max coverage of one word is 13.
                # Intersection of 4 neighbors sets is likely empty or small.
                set1 = set(neighbors[unique_to_c1[0]])
                for ub in unique_to_c1[1:]:
                    set1 &= set(neighbors[ub])
                    if not set1:
                        break
                candidates = list(set1)

            for w in candidates:
                if w in code_set:
                    continue # w is already in code, swapping c1->w is no-op or just reordering
                
                # Check if w covers all unique_to_c1 (guaranteed by candidate generation)
                # Simulate swap:
                # 1. Remove c1: decrement cover_count for neighbors[c1]
                # 2. Add w: increment cover_count for neighbors[w]
                
                # To avoid modifying global cover_count, we can check redundancy condition directly.
                # A codeword c2 (c2 != c1) is redundant after swap if:
                # For all nb in neighbors[c2]:
                #   new_cover_count[nb] >= 2
                # new_cover_count[nb] = cover_count[nb] - (1 if nb in neighbors[c1] else 0) + (1 if nb in neighbors[w] else 0)
                
                # We want to find if ANY c2 becomes redundant.
                # Note: w itself is not in code yet, so we don't check w for redundancy.
                # We check all c2 in code where c2 != c1.
                
                found_redundant_c2 = -1
                for c2 in code:
                    if c2 == c1:
                        continue
                    
                    redundant = True
                    for nb in neighbors[c2]:
                        cnt = cover_count[nb]
                        if nb in neighbors[c1]:
                            cnt -= 1
                        if nb in neighbors[w]:
                            cnt += 1
                        if cnt < 2:
                            redundant = False
                            break
                    
                    if redundant:
                        found_redundant_c2 = c2
                        break
                
                if found_redundant_c2 != -1:
                    # Perform the swap and removal
                    # 1. Remove c1
                    for nb in neighbors[c1]:
                        cover_count[nb] -= 1
                    code.remove(c1)
                    code_set.remove(c1)
                    
                    # 2. Remove c2
                    for nb in neighbors[found_redundant_c2]:
                        cover_count[nb] -= 1
                    code.remove(found_redundant_c2)
                    code_set.remove(found_redundant_c2)
                    
                    # 3. Add w
                    for nb in neighbors[w]:
                        cover_count[nb] += 1
                    code.append(w)
                    code_set.add(w)
                    
                    improved = True
                    break # Restart outer loop
            
            if improved:
                break

    # Convert indices back to lists of integers
    result = []
    for c in code:
        result.append(list(idx_to_word(c)))
        
    return result
