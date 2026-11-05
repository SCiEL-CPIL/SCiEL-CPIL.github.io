import itertools
import numpy as np
import time

def construct(q, n, radius):
    # For this specific task, we assume q=3, n=7, radius=1
    # But the function signature is general.
    # If parameters differ, we might need a fallback.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we only need to handle the specific case or a general case.
    # Given the time limit and complexity, I will specialize for q=3, n=7, r=1
    # and provide a fallback for other cases (e.g., greedy only).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple Greedy
        return _greedy_construct(q, n, radius)

    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Map word index to word tuple for later conversion
    # Index is just the integer representation in base q
    
    # Precompute balls as bitsets
    # Ball of i: i and all words at distance 1
    balls = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    # Find index of new_w
                    # Since we generated in product order, index can be computed
                    # Or use a dictionary
                    pass
        
    # Using dictionary for index lookup is safer/faster for arbitrary q,n
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    balls = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                new_w = w.copy()
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        balls[i] = mask

    all_mask = (1 << N) - 1
    
    # 1. Greedy Initialization (Vectorized)
    uncovered = all_mask
    code = []
    
    # Precompute ball arrays as numpy arrays of indices for vectorized coverage counting?
    # Actually, bitsets are hard to vectorize for "count bits in intersection".
    # Alternative: Represent balls as sparse arrays of indices?
    # For N=2187, a boolean matrix of size 2187x2187 is ~4.7M booleans = ~4.7MB.
    # This is very manageable.
    # Let's create a boolean matrix `ball_matrix` where ball_matrix[i][j] is True if j is in ball(i).
    # Then coverage of candidate i is sum(ball_matrix[i][uncovered_indices]).
    # But updating uncovered set is easier with bitsets or lists.
    
    # Let's stick to bitsets for the local search but optimize the greedy.
    # Greedy:
    # Instead of iterating all i and counting bits, we can maintain a `coverage_count` array.
    # Initially all 0.
    # When we pick i, we decrement coverage_count for all j in ball(i) that are currently uncovered?
    # No, standard greedy:
    # 1. Pick i that maximizes |Ball(i) \ Covered|.
    # 2. Mark Ball(i) as Covered.
    
    # Vectorized Greedy:
    # We can maintain an array `uncovered_mask` (numpy bool array).
    # For each step, compute `cov = np.sum(ball_matrix[:, uncovered_mask], axis=1)`.
    # This is O(N^2) per step if done naively.
    # However, N=2187. N steps * N^2 ops = 2187 * 4.7M ~ 10^10 ops. Too slow in Python/NumPy?
    # NumPy sum is fast. 2187 steps. Each step does a sum over a 2187x2187 matrix subset.
    # It might be okay. Let's try a different approach for Greedy:
    # Randomized Greedy or just simple greedy with bitsets but optimized inner loop.
    
    # Let's use the bitset greedy but optimize the "best_i" selection.
    # We can precompute the list of indices in each ball to iterate faster?
    # Or just use the bit_count method but it was slow.
    
    # Let's try a simple heuristic: Pick the word that covers the most *uncovered* words.
    # We'll use a numpy boolean array for `uncovered`.
    uncovered_np = np.ones(N, dtype=bool)
    
    # Precompute ball indices as lists of ints for fast iteration in greedy?
    # Or just use the bitset and extract bits?
    # Let's stick to the bitset approach for consistency with local search, but optimize the loop.
    
    # Actually, the previous greedy was:
    # for i in range(N): cov = (balls[i] & uncovered).bit_count()
    # This is 2187 * 2187 bit operations. Python `int` bit operations are C-speed.
    # `bit_count` is also C-speed.
    # 4.7M iterations of a tight loop might take ~0.5s - 1s.
    # 2187 steps * 1s = 2000s. Too slow.
    
    # We need a faster greedy.
    # Idea: Use a priority queue? No, coverage changes.
    # Idea: Randomized greedy? Pick a random subset, evaluate?
    # Idea: Vectorized coverage using NumPy.
    
    # Let's build `ball_indices`: list of lists.
    ball_indices = [[] for _ in range(N)]
    for i in range(N):
        mask = balls[i]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            ball_indices[i].append(idx)
            mask ^= lsb
            
    # Now Greedy:
    # uncovered is a set or list?
    # If we use a boolean array `is_uncovered`, updating it is O(1) per element in ball.
    # To find best i:
    # We need count of j in ball_indices[i] such that is_uncovered[j] is True.
    # This is still O(size_of_ball) per candidate i.
    # Total O(N * size_of_ball) per step.
    # size_of_ball = 1 + 2*n = 15.
    # 2187 * 15 = 32k ops per step.
    # 2187 steps * 32k = 70M ops.
    # In Python, 70M simple ops might take ~5-10 seconds.
    # This is feasible within 120s.
    
    is_uncovered = np.ones(N, dtype=bool)
    code = []
    
    while True:
        if not is_uncovered.any():
            break
        best_i = -1
        best_cov = 0
        for i in range(N):
            # Count uncovered in ball i
            # Using numpy sum on boolean array slice?
            # is_uncovered[ball_indices[i]] -> array of bools
            # np.sum(...)
            # This creates a temporary array. 2187 times per step.
            # Might be slow due to allocation.
            # Alternative: manual loop in Python?
            # for j in ball_indices[i]: if is_uncovered[j]: c+=1
            # 15 iterations. 2187 * 15 = 32k Python iterations.
            # 2187 steps * 32k = 70M Python iterations.
            # 70M Python iterations is likely ~10-20 seconds.
            # Acceptable.
            
            c = 0
            for j in ball_indices[i]:
                if is_uncovered[j]:
                    c += 1
            if c > best_cov:
                best_cov = c
                best_i = i
                
        code.append(best_i)
        # Mark covered
        for j in ball_indices[best_i]:
            is_uncovered[j] = False
            
    # 2. Local Search
    # Data structures for fast updates
    cover_count = np.zeros(N, dtype=np.int32)
    for i in code:
        for j in ball_indices[i]:
            cover_count[j] += 1
            
    # Masks for words with count 1 and 2
    mask_count_1 = 0
    mask_count_2 = 0
    for i in range(N):
        if cover_count[i] == 1:
            mask_count_1 |= (1 << i)
        elif cover_count[i] == 2:
            mask_count_2 |= (1 << i)

    improved = True
    while improved:
        improved = False
        
        # Pass 1: Single Removal
        for idx_in_code, i in enumerate(code):
            if (balls[i] & mask_count_1) == 0:
                # Remove i
                code.pop(idx_in_code)
                # Update cover_count and masks
                for j in ball_indices[i]:
                    old_c = cover_count[j]
                    cover_count[j] -= 1
                    new_c = cover_count[j]
                    if old_c == 1:
                        mask_count_1 &= ~(1 << j)
                    elif old_c == 2:
                        mask_count_2 &= ~(1 << j)
                    if new_c == 1:
                        mask_count_1 |= (1 << j)
                    elif new_c == 2:
                        mask_count_2 |= (1 << j)
                improved = True
                break # Restart loop after modification
                
        if improved:
            continue

        # Pass 2: Single Replacement
        # Try to replace i with k if Unique(i) subset of Ball(k)
        # Unique(i) = Ball(i) & mask_count_1
        # We need a k such that Unique(i) is covered by k.
        # i.e., (Unique(i) & ~balls[k]) == 0
        # Also, we must ensure that removing i doesn't leave holes that k doesn't cover?
        # If Unique(i) is covered by k, then all words that ONLY i covers are now covered by k.
        # Words covered by both i and others remain covered.
        # So yes, if Unique(i) subset Ball(k), then replacing i with k is valid.
        # Note: k can be any word, not necessarily in code.
        
        for idx_in_code, i in enumerate(code):
            unique_i = balls[i] & mask_count_1
            if unique_i == 0:
                continue # Would have been removed in Pass 1
            
            # Find a k that covers unique_i
            # Candidates for k: any word that intersects unique_i?
            # Actually, k must cover ALL of unique_i.
            # So k must be in the intersection of balls of all words in unique_i?
            # No, k is a codeword. Ball(k) must contain unique_i.
            # This implies k is close to all words in unique_i.
            # Let's pick one word w from unique_i.
            # k must be in Ball(w).
            # Also, for any other word u in unique_i, dist(k, u) <= 1.
            
            # Iterate through candidates k in Ball(w) for w in unique_i?
            # Let's just pick one w from unique_i and check all k in Ball(w).
            lsb = unique_i & -unique_i
            w_idx = lsb.bit_length() - 1
            
            candidates = ball_indices[w_idx]
            
            found_k = -1
            for k in candidates:
                if (unique_i & ~balls[k]) == 0:
                    # Check if k is already in code?
                    # If k is in code, replacing i with k is just removing i (since k is already there).
                    # But if k is in code, then unique_i is already covered by k?
                    # If k is in code, and k covers unique_i, then those words in unique_i are covered by i AND k.
                    # So their count is at least 2.
                    # But unique_i is defined as words with count 1 (covered ONLY by i).
                    # So if k is in code, it cannot cover any word in unique_i, because then that word would have count >= 2.
                    # Thus, k cannot be in code.
                    
                    found_k = k
                    break
                    
            if found_k != -1:
                # Replace i with found_k
                code[idx_in_code] = found_k
                # Update counts: Remove i, Add found_k
                # Remove i
                for j in ball_indices[i]:
                    old_c = cover_count[j]
                    cover_count[j] -= 1
                    new_c = cover_count[j]
                    if old_c == 1: mask_count_1 &= ~(1 << j)
                    elif old_c == 2: mask_count_2 &= ~(1 << j)
                    if new_c == 1: mask_count_1 |= (1 << j)
                    elif new_c == 2: mask_count_2 |= (1 << j)
                
                # Add found_k
                for j in ball_indices[found_k]:
                    old_c = cover_count[j]
                    cover_count[j] += 1
                    new_c = cover_count[j]
                    if old_c == 1: mask_count_1 &= ~(1 << j)
                    elif old_c == 2: mask_count_2 &= ~(1 << j)
                    if new_c == 1: mask_count_1 |= (1 << j)
                    elif new_c == 2: mask_count_2 |= (1 << j)
                    
                improved = True
                break

        if improved:
            continue

        # Pass 3: Pair Replacement
        len_code = len(code)
        for i_idx in range(len_code):
            i = code[i_idx]
            for j_idx in range(i_idx + 1, len_code):
                j = code[j_idx]
                
                Bi = balls[i]
                Bj = balls[j]
                
                Bi_only = Bi & ~Bj
                Bj_only = Bj & ~Bi
                Bi_Bj = Bi & Bj
                
                U = (Bi_only & mask_count_1) | (Bj_only & mask_count_1) | (Bi_Bj & mask_count_2)
                
                if U == 0:
                    # Both redundant
                    code.pop(j_idx)
                    for jj in ball_indices[j]:
                        old_c = cover_count[jj]
                        cover_count[jj] -= 1
                        new_c = cover_count[jj]
                        if old_c == 1: mask_count_1 &= ~(1 << jj)
                        elif old_c == 2: mask_count_2 &= ~(1 << jj)
                        if new_c == 1: mask_count_1 |= (1 << jj)
                        elif new_c == 2: mask_count_2 |= (1 << jj)
                        
                    code.pop(i_idx)
                    for ii in ball_indices[i]:
                        old_c = cover_count[ii]
                        cover_count[ii] -= 1
                        new_c = cover_count[ii]
                        if old_c == 1: mask_count_1 &= ~(1 << ii)
                        elif old_c == 2: mask_count_2 &= ~(1 << ii)
                        if new_c == 1: mask_count_1 |= (1 << ii)
                        elif new_c == 2: mask_count_2 |= (1 << ii)
                        
                    improved = True
                    break
                    
                # Check if U is covered by some k
                lsb = U & -U
                w_idx = lsb.bit_length() - 1
                candidates = ball_indices[w_idx]
                
                found_k = -1
                for k in candidates:
                    if k == i or k == j:
                        continue
                    if (U & ~balls[k]) == 0:
                        found_k = k
                        break
                        
                if found_k != -1:
                    # Replace i, j with k
                    code.pop(j_idx)
                    for jj in ball_indices[j]:
                        old_c = cover_count[jj]
                        cover_count[jj] -= 1
                        new_c = cover_count[jj]
                        if old_c == 1: mask_count_1 &= ~(1 << jj)
                        elif old_c == 2: mask_count_2 &= ~(1 << jj)
                        if new_c == 1: mask_count_1 |= (1 << jj)
                        elif new_c == 2: mask_count_2 |= (1 << jj)
                        
                    code.pop(i_idx)
                    for ii in ball_indices[i]:
                        old_c = cover_count[ii]
                        cover_count[ii] -= 1
                        new_c = cover_count[ii]
                        if old_c == 1: mask_count_1 &= ~(1 << ii)
                        elif old_c == 2: mask_count_2 &= ~(1 << ii)
                        if new_c == 1: mask_count_1 |= (1 << ii)
                        elif new_c == 2: mask_count_2 |= (1 << ii)
                        
                    code.append(found_k)
                    for kk in ball_indices[found_k]:
                        old_c = cover_count[kk]
                        cover_count[kk] += 1
                        new_c = cover_count[kk]
                        if old_c == 1: mask_count_1 &= ~(1 << kk)
                        elif old_c == 2: mask_count_2 &= ~(1 << kk)
                        if new_c == 1: mask_count_1 |= (1 << kk)
                        elif new_c == 2: mask_count_2 |= (1 << kk)
                        
                    improved = True
                    break
                    
            if improved:
                break
                
    return [words[i] for i in code]

def _greedy_construct(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    balls = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        # Generate neighbors within radius
        # For general radius, use BFS or recursive generation
        # Here we assume small radius
        def gen_neighbors(pos, current, dist):
            if dist == 0:
                yield current
                return
            if pos == n:
                return
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                new_val = (current[pos] + delta) % q
                if new_val != current[pos]:
                    new_w = current.copy()
                    new_w[pos] = new_val
                    yield from gen_neighbors(pos + 1, new_w, dist - 1)
        # This generation is inefficient for general radius.
        # For the specific task, we don't use this.
        # For fallback, just use simple greedy without bitsets if N is small, or random.
        pass
        
    # Simple Greedy without bitsets for fallback
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            w = words[i]
            cov = 0
            for u_idx in uncovered:
                u = words[u_idx]
                if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        # Remove covered
        w = words[best_i]
        to_remove = []
        for u_idx in uncovered:
            u = words[u_idx]
            if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                to_remove.append(u_idx)
        uncovered.difference_update(to_remove)
    return [words[i] for i in code]
