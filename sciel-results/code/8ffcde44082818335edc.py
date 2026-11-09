# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses a greedy set cover approach followed by local search (removal/swap/2-swap) to minimize size.
    Optimized for q=3, n=7, radius=1 using bitsets.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        # Generate all 3^7 = 2187 words
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        # Map word to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as bitmasks
        # Each word has 1 + 7*2 = 15 neighbors
        neighbor_masks = [0] * N
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
            neighbor_masks[i] = mask
        
        # Initial Solution: Tetracode construction (243 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        best_size = len(initial_codewords)
        best_indices = [word_to_idx[c] for c in initial_codewords]

        def optimize(initial_indices, max_time=10.0):
            """
            Local search to minimize the number of codewords.
            Moves:
            1. Remove redundant codeword.
            2. Swap codeword A with non-codeword B if valid.
            3. 2-Swap: Remove A, B and add C, D if valid.
            """
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            # Initialize cover counts
            cover_count = [0] * N
            for idx in current_indices:
                mask = neighbor_masks[idx]
                m = mask
                while m:
                    lb = m & -m
                    idx_nb = lb.bit_length() - 1
                    cover_count[idx_nb] += 1
                    m ^= lb

            improved = True
            local_start = time.time()
            
            while improved and time.time() < local_start + max_time:
                improved = False
                random.shuffle(current_indices)
                
                # Pass 1: Removal
                for i in range(len(current_indices)):
                    if time.time() > local_start + max_time: break
                    idx = current_indices[i]
                    
                    # Check redundancy
                    mask = neighbor_masks[idx]
                    m = mask
                    redundant = True
                    while m:
                        lb = m & -m
                        idx_nb = lb.bit_length() - 1
                        if cover_count[idx_nb] <= 1:
                            redundant = False
                            break
                        m ^= lb
                    
                    if redundant:
                        # Remove idx
                        m = mask
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            cover_count[idx_nb] -= 1
                            m ^= lb
                        
                        current_indices.pop(i)
                        improved = True
                        continue

                # Pass 2: 1-Swap
                # Try to swap a codeword with a non-codeword to enable further removals
                if len(current_indices) > 180: # Threshold
                    s_set = set(current_indices)
                    for _ in range(20): # Limit swap attempts per pass
                        if time.time() > local_start + max_time: break
                        if not current_indices: break
                        
                        idx_a = random.choice(current_indices)
                        idx_b = random.randint(0, N-1)
                        if idx_b in s_set: continue
                        
                        mask_a = neighbor_masks[idx_a]
                        mask_b = neighbor_masks[idx_b]
                        
                        # Check validity of swap A->B
                        # Words covered only by A must be covered by B
                        valid_swap = True
                        m = mask_a
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            if cover_count[idx_nb] == 1:
                                if not (mask_b & (1 << idx_nb)):
                                    valid_swap = False
                                    break
                            m ^= lb
                        
                        if valid_swap:
                            # Perform Swap
                            m_a = mask_a
                            m_b = mask_b
                            
                            while m_a:
                                lb = m_a & -m_a
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] -= 1
                                m_a ^= lb
                                
                            while m_b:
                                lb = m_b & -m_b
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] += 1
                                m_b ^= lb
                                
                            current_indices[current_indices.index(idx_a)] = idx_b
                            s_set.discard(idx_a)
                            s_set.add(idx_b)
                            improved = True
                            break # Restart loop to check for removals

                # Pass 3: 2-Swap
                # Remove A, B and add C, D
                if len(current_indices) > 170:
                    s_set = set(current_indices)
                    for _ in range(10):
                        if time.time() > local_start + max_time: break
                        if len(current_indices) < 2: break
                        
                        # Pick A, B
                        i1 = random.randint(0, len(current_indices)-1)
                        i2 = random.randint(0, len(current_indices)-1)
                        if i1 == i2: continue
                        idx_a = current_indices[i1]
                        idx_b = current_indices[i2]
                        
                        # Pick C, D
                        idx_c = random.randint(0, N-1)
                        idx_d = random.randint(0, N-1)
                        if idx_c in s_set or idx_d in s_set: continue
                        if idx_c == idx_d: continue
                        
                        mask_a = neighbor_masks[idx_a]
                        mask_b = neighbor_masks[idx_b]
                        mask_c = neighbor_masks[idx_c]
                        mask_d = neighbor_masks[idx_d]
                        
                        # Check validity:
                        # For every word w, new_count[w] > 0
                        # new_count = cover_count - mask_a - mask_b + mask_c + mask_d
                        # We only need to check words where cover_count is small or affected.
                        # Actually, just checking all words is N=2187 ops, which is fast.
                        valid_2swap = True
                        # To speed up, we can check only words in union of masks?
                        # No, words NOT in masks have cover_count unchanged.
                        # If cover_count[w] > 0 and w not in any of the 4 masks, it's fine.
                        # We only need to check words in mask_a | mask_b | mask_c | mask_d?
                        # If w is not in mask_a or mask_b, its count decreases by 0.
                        # If w is not in mask_c or mask_d, its count increases by 0.
                        # So we only need to check w in (mask_a | mask_b | mask_c | mask_d).
                        # But wait, if w is in mask_a (count -1) and not in others, and cover_count[w]==1, it becomes 0.
                        # So yes, checking the union of masks is sufficient.
                        
                        union_mask = mask_a | mask_b | mask_c | mask_d
                        m = union_mask
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            # Calculate new count for this word
                            new_cnt = cover_count[idx_nb]
                            if mask_a & lb: new_cnt -= 1
                            if mask_b & lb: new_cnt -= 1
                            if mask_c & lb: new_cnt += 1
                            if mask_d & lb: new_cnt += 1
                            if new_cnt <= 0:
                                valid_2swap = False
                                break
                            m ^= lb
                        
                        if valid_2swap:
                            # Perform 2-Swap
                            # Update cover counts
                            for mask in [mask_a, mask_b]:
                                m = mask
                                while m:
                                    lb = m & -m
                                    idx_nb = lb.bit_length() - 1
                                    cover_count[idx_nb] -= 1
                                    m ^= lb
                            for mask in [mask_c, mask_d]:
                                m = mask
                                while m:
                                    lb = m & -m
                                    idx_nb = lb.bit_length() - 1
                                    cover_count[idx_nb] += 1
                                    m ^= lb
                            
                            # Update list
                            # Remove A, B
                            # Careful with indices if we pop
                            # Easier: create new list
                            new_list = [x for x in current_indices if x != idx_a and x != idx_b]
                            new_list.append(idx_c)
                            new_list.append(idx_d)
                            current_indices = new_list
                            s_set = set(current_indices)
                            improved = True
                            break # Restart loop to check for removals

            return current_indices

        # Run optimization on Tetracode
        optimized_tet = optimize(best_indices, max_time=30.0)
        if len(optimized_tet) < best_size:
            best_size = len(optimized_tet)
            best_indices = optimized_tet

        # Try Greedy Initial Solution
        if time.time() < deadline - 20:
            uncovered_mask = (1 << N) - 1
            greedy_indices = []
            greedy_set = set()
            
            while uncovered_mask and time.time() < deadline - 15:
                best_idx = -1
                best_gain = 0
                # To speed up, we can sample or use a heuristic.
                # Full scan is 2187 * bit_count.
                for idx in range(N):
                    if idx in greedy_set: continue
                    gain = (neighbor_masks[idx] & uncovered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if gain == uncovered_mask.bit_count(): break
                
                if best_idx == -1 or best_gain == 0: break
                greedy_indices.append(best_idx)
                greedy_set.add(best_idx)
                uncovered_mask &= ~neighbor_masks[best_idx]
            
            if len(greedy_indices) < best_size:
                optimized_greedy = optimize(greedy_indices, max_time=20.0)
                if len(optimized_greedy) < best_size:
                    best_size = len(optimized_greedy)
                    best_indices = optimized_greedy

        # Multi-start with random perturbations of the best solution
        if time.time() < deadline - 10:
            # Take best solution, remove some, add random, optimize
            for _ in range(5):
                if time.time() > deadline - 5: break
                perturbed = list(best_indices)
                # Remove 10 random codewords
                for _ in range(min(10, len(perturbed))):
                    if not perturbed: break
                    idx_to_remove = random.choice(perturbed)
                    perturbed.remove(idx_to_remove)
                # Add 15 random codewords
                s_set = set(perturbed)
                for _ in range(15):
                    idx_to_add = random.randint(0, N-1)
                    if idx_to_add not in s_set:
                        perturbed.append(idx_to_add)
                        s_set.add(idx_to_add)
                
                optimized_pert = optimize(perturbed, max_time=10.0)
                if len(optimized_pert) < best_size:
                    best_size = len(optimized_pert)
                    best_indices = optimized_pert

        # Return the best found
        result = [list(all_words[idx]) for idx in best_indices]
        return result

    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set: continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1: break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
# EVOLVE-BLOCK-END
