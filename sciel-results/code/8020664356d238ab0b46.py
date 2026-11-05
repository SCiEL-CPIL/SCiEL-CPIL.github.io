import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        total_words = q ** n
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx
            
        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        # Precompute neighbors
        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        # Simple Greedy Set Cover
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx
        
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return tuple(w)

    # Precompute neighbors in a NumPy array for fast vectorized operations
    # Shape: (total_words, 17)
    # 17 = 1 (self) + 8 * 2 (flips)
    neighbors_matrix = np.empty((total_words, 17), dtype=np.int32)
    
    for idx in range(total_words):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        neighbors_matrix[idx, 0] = idx
        k = 1
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = w.copy()
                    nw[i] = val
                    neighbors_matrix[idx, k] = word_to_idx(tuple(nw))
                    k += 1

    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe within 150s

    # Phase 1: Greedy Set Cover Construction with random tie-breaking
    uncovered = np.ones(total_words, dtype=bool)
    selected = np.zeros(total_words, dtype=bool)
    code_list = []
    
    while uncovered.any():
        if time.time() > deadline - 30:  # Leave time for local search
            break
            
        # Vectorized count of uncovered neighbors for all candidates
        counts = uncovered[neighbors_matrix].sum(axis=1)
        
        # Mask out already selected codewords
        counts[selected] = -1
        
        # Find all candidates with max count for random tie-breaking
        max_count = counts.max()
        if max_count <= 0:
            break
            
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[random.randint(0, len(candidates) - 1)]
        
        code_list.append(best_idx)
        selected[best_idx] = True
        # Update uncovered status
        uncovered[neighbors_matrix[best_idx]] = False

    # Phase 2: Local Search Refinement
    code_set = set(code_list)
    
    # Initialize cover_count
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_matrix[c]] += 1

    # Helper to check if a codeword is redundant
    def is_redundant(c):
        return np.min(cover_count[neighbors_matrix[c]]) >= 2

    # Removal Phase
    removed_any = True
    while removed_any and time.time() < deadline - 10:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            if is_redundant(c):
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors_matrix[c]] -= 1
                removed_any = True
                break

    # Swap Phase (1-for-1) to enable further removals
    # We try to swap a codeword c with c' such that:
    # 1. c' is not in code_set
    # 2. After removing c, the uncovered points U are covered by c'
    # 3. c' leads to higher redundancy potential
    
    max_swap_iters = 5000
    for _ in range(max_swap_iters):
        if time.time() > deadline - 10:
            break
            
        if not code_list:
            break
            
        # Pick a random codeword to potentially swap
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Temporarily remove c
        cover_count[neighbors_matrix[c]] -= 1
        
        # Find uncovered points U
        u_mask = (cover_count[neighbors_matrix[c]] == 0)
        if not np.any(u_mask):
            # c was redundant, should have been removed. Revert.
            cover_count[neighbors_matrix[c]] += 1
            continue
        
        u_indices = neighbors_matrix[c][u_mask]
        
        # Find candidates c' that cover all points in u_indices
        candidates = set(neighbors_matrix[u_indices[0]].tolist())
        
        for u in u_indices[1:]:
            candidates.intersection_update(neighbors_matrix[u].tolist())
            if not candidates:
                break
        
        if candidates:
            candidates.difference_update(code_set)
            
            if candidates:
                # Score candidates by how many of their neighbors have cover_count >= 2
                # This promotes swaps that increase redundancy
                best_cp = None
                best_score = -1
                for cp in candidates:
                    score = np.sum(cover_count[neighbors_matrix[cp]] >= 2)
                    if score > best_score:
                        best_score = score
                        best_cp = cp
                
                if best_cp is not None:
                    # Perform swap
                    code_list[idx_c] = best_cp
                    code_set.add(best_cp)
                    code_set.discard(c)
                    
                    # Update cover counts
                    cover_count[neighbors_matrix[best_cp]] += 1
                    
                    # Try to remove redundant words immediately after a successful swap
                    for j in range(len(code_list) - 1, -1, -1):
                        cj = code_list[j]
                        if is_redundant(cj):
                            code_list.pop(j)
                            code_set.discard(cj)
                            cover_count[neighbors_matrix[cj]] -= 1
                            break
            else:
                cover_count[neighbors_matrix[c]] += 1
        else:
            cover_count[neighbors_matrix[c]] += 1

    # Final cleanup
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            if is_redundant(c):
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors_matrix[c]] -= 1
                removed_any = True
                break

    return [list(idx_to_word(c)) for c in code_list]
