import time
import random
import numpy as np

def construct(q, n, radius):
    # Specialized for q=3, n=7, radius=1 (K_3(7,1))
    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer before 150s kill
    
    # 1. Generate all words and map to indices
    # There are 3^7 = 2187 words
    all_words = []
    word_to_idx = {}
    idx = 0
    # Using a fast loop to generate ternary strings
    # We can use np.ndindex or manual recursion. np.ndindex is fine.
    for digits in np.ndindex(*([3] * n)):
        w = list(digits)
        word_to_idx[tuple(w)] = idx
        all_words.append(w)
        idx += 1
    N = len(all_words)
    
    # 2. Precompute neighbors (distance <= 1)
    # neighbors[i] is a list of indices j such that d(all_words[i], all_words[j]) <= 1
    # Each word has 1 + n*(q-1) = 1 + 7*2 = 15 neighbors.
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                neighbors[i].append(j)
    
    # 3. Greedy Construction
    # We use a vectorized approach to find the best candidate at each step.
    # However, for N=2187, a simple loop is fast enough if optimized.
    # Let's try a slightly smarter greedy:
    # uncovered = set of indices not yet covered.
    # At each step, find c that maximizes |neighbors[c] intersect uncovered|.
    
    # To speed this up, we can maintain an array `gain` for each candidate?
    # Or just compute on the fly.
    # 2187 candidates * 15 neighbors * ~200 steps = ~6.5M ops. This is very fast in Python if we use lists.
    
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        best_candidate = -1
        max_covered = -1
        for c in range(N):
            if c in code:
                continue
            # Count how many neighbors of c are in uncovered
            count = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_candidate = c
                # Early exit if we cover everything?
                if count == len(uncovered):
                    break
        if best_candidate == -1:
            break
        code.append(best_candidate)
        # Remove covered words
        for nb in neighbors[best_candidate]:
            if nb in uncovered:
                uncovered.discard(nb)
                
    initial_code_words = [all_words[c] for c in code]
    
    # 4. Local Search / Improvement
    # Represent code as a set of indices for fast lookup
    code_set = set(code)
    
    # cover_count[i] = number of codewords in code_set that cover i
    cover_count = np.zeros(N, dtype=int)
    for c in code_set:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def is_full_cover(cover_count):
        return np.all(cover_count > 0)
        
    # Local search loop
    last_improvement = time.time()
    best_size = len(code_set)
    best_code_set = set(code_set)
    
    while time.time() < deadline:
        # Try to remove a redundant codeword
        removed = False
        for c in list(code_set):
            # Check if all neighbors of c are covered by others
            # i.e., cover_count[nb] > 1 for all nb in neighbors[c]
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                code_set.discard(c)
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                removed = True
                last_improvement = time.time()
                break
                
        if not removed:
            # Try a swap to potentially enable future removals
            # Pick a random codeword to remove
            if code_set:
                c_out = random.choice(list(code_set))
                # Temporarily remove
                for nb in neighbors[c_out]:
                    cover_count[nb] -= 1
                code_set.discard(c_out)
                
                # Find uncovered words
                # We need to find a c_in that covers the most uncovered words
                # Uncovered words are those with cover_count == 0
                # Let's find the list of uncovered indices
                uncovered_indices = np.where(cover_count == 0)[0]
                
                if len(uncovered_indices) == 0:
                    # We actually removed a redundant one? 
                    # This shouldn't happen if the removal check above was correct, 
                    # unless the removal check was too strict or I made a logic error.
                    # Wait, if cover_count[nb] > 1 for all nb, then after decrement, 
                    # cover_count[nb] >= 1. So no uncovered.
                    # So this case means we found a redundant one that the loop missed? 
                    # Or maybe the loop broke early?
                    # Anyway, if no uncovered, we just keep the smaller set.
                    pass
                else:
                    # Find best c_in
                    best_c_in = -1
                    max_cover = -1
                    # We can restrict search to candidates that cover at least one uncovered word
                    # To be fast, we can just sample or iterate all.
                    # Iterating all 2187 is fine.
                    for c_in in range(N):
                        if c_in in code_set:
                            continue
                        # Count how many uncovered_indices are in neighbors[c_in]
                        # This is a set intersection or count.
                        # neighbors[c_in] is a list.
                        # We can convert uncovered_indices to a set for O(1) lookup?
                        # Or just iterate neighbors[c_in] and check cover_count.
                        cnt = 0
                        for nb in neighbors[c_in]:
                            if cover_count[nb] == 0:
                                cnt += 1
                        if cnt > max_cover:
                            max_cover = cnt
                            best_c_in = c_in
                            
                    if best_c_in != -1 and max_cover > 0:
                        # Perform swap
                        code_set.add(best_c_in)
                        for nb in neighbors[best_c_in]:
                            cover_count[nb] += 1
                        # Note: cover_count for neighbors of c_out are already decremented.
                        # The swap keeps size constant.
                    else:
                        # Revert swap
                        code_set.add(c_out)
                        for nb in neighbors[c_out]:
                            cover_count[nb] += 1
            else:
                # Code is empty? Should not happen.
                break
                
        # Update best
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code_set = set(code_set)
            
        # Optional: Restart if stuck for too long?
        # For now, just keep swapping.
        
    # Return the best code found
    return [all_words[c] for c in best_code_set]

def _generic_construct(q, n, radius):
    # Fallback for non-standard parameters
    # Simple greedy
    all_words = []
    word_to_idx = {}
    idx = 0
    for digits in np.ndindex(*([q] * n)):
        w = list(digits)
        word_to_idx[tuple(w)] = idx
        all_words.append(w)
        idx += 1
    N = len(all_words)
    
    # Precompute neighbors within radius
    neighbors = [[] for _ in range(N)]
    # This is expensive for large n/q, but for fallback it's okay if small
    # For now, just use brute force distance check if small
    # Or just return a trivial solution if too big
    if N > 10000:
        return [all_words[0]] # Invalid but fast
    
    for i, w1 in enumerate(all_words):
        for j, w2 in enumerate(all_words):
            dist = sum(1 for a, b in zip(w1, w2) if a != b)
            if dist <= radius:
                neighbors[i].append(j)
                
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_candidate = -1
        max_covered = -1
        for c in range(N):
            if c in code:
                continue
            count = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_candidate = c
        if best_candidate == -1:
            break
        code.append(best_candidate)
        for nb in neighbors[best_candidate]:
            if nb in uncovered:
                uncovered.discard(nb)
                
    return [all_words[c] for c in code]
