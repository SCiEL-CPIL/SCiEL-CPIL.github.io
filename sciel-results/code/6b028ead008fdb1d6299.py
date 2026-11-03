import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Find candidates that cover all holes using vectorized approach
                    # For each candidate, check if it covers all holes
                    hole_neighbors = neighbors_arr[holes, :]  # shape: (num_holes, cover_count)
                    
                    # A candidate covers a hole if the hole is in candidate's neighborhood
                    # Equivalent: for each hole h, candidate c covers h iff h is in neighbors_arr[c]
                    # Use the fact that neighbors are symmetric for radius 1
                    # Actually for Hamming distance 1, if h is in N(c) then c is in N(h)
                    
                    # Vectorized: for each candidate c, count how many holes it covers
                    # candidate c covers hole h iff h is in neighbors_arr[c]
                    # This is equivalent to: for each h in holes, check if c is in neighbors_arr[h]
                    
                    # More efficient: build a boolean matrix
                    # num_holes x total_words: does candidate j cover hole i?
                    # candidate j covers hole h iff h is in neighbors_arr[j]
                    # i.e., neighbors_arr[j] contains h
                    # Equivalent: for each j, count how many of holes are in neighbors_arr[j]
                    
                    # We can compute this as: for each hole h, mark which candidates cover it
                    # Then sum across holes
                    # candidate j covers hole h iff h is in neighbors_arr[j]
                    # iff j is in neighbors_arr[h] (by symmetry of Hamming distance 1)
                    
                    # So: coverage_matrix[h, j] = 1 iff j is in neighbors_arr[h]
                    # We want candidates j where sum over h of coverage_matrix[h, j] == len(holes)
                    
                    # Vectorized: for each hole, get its neighbor list, then accumulate
                    hole_neighbor_sets = neighbors_arr[holes, :]  # (num_holes, cover_count)
                    
                    # Create a 2D boolean: for each candidate, how many holes does it cover?
                    # candidate j covers hole h iff j is in neighbors_arr[h]
                    # So for each hole h, the candidates that cover it are neighbors_arr[h]
                    
                    # Accumulate counts
                    cover_count_per_candidate = np.zeros(total_words, dtype=np.int16)
                    for h in holes:
                        cover_count_per_candidate[neighbors_arr[h, :]] += 1
                    
                    # Candidates that cover all holes
                    all_holes = len(holes)
                    candidates = np.where(cover_count_per_candidate == all_holes)[0]
                    candidates = candidates[candidates != c_remove]
                    
                    if len(candidates) == 0:
                        counts[neighbors_c] += 1
                        continue
                    
                    cand_list = candidates.tolist()
                    if len(cand_list) > 100:
                        cand_list = random.sample(cand_list, 100)
                    
                    # Score: prefer candidates that maximize coverage of already-covered words
                    # (i.e., minimize additional overlap), but all cover the holes
                    # So score = number of words already covered that this candidate also covers
                    # Lower is better (less overlap)
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    
                    # Score = sum of counts over the candidate's neighborhood
                    # But we want to minimize this (less overlap with existing coverage)
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    
                    # Choose the one with minimum score (least overlap)
                    # With some randomness to escape local optima
                    min_score = np.min(scores)
                    best_cands = np.where(scores == min_score)[0]
                    
                    if len(best_cands) > 1 and random.random() < 0.3:
                        # Sometimes pick a random one from top candidates for diversity
                        top_cands = np.where(scores <= min_score + 2)[0]
                        chosen_idx = random.choice(top_cands.tolist())
                    else:
                        chosen_idx = random.choice(best_cands.tolist())
                    
                    best_cand = cand_list[chosen_idx]
                    
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        code_arr = np.array(code, dtype=np.int32)
        all_neighbors = neighbors_arr[code_arr, :]
        scores = np.sum(counts[all_neighbors], axis=1)
        
        # Remove codewords with highest overlap (most redundant)
        top_k_factor = 2
        num_to_consider = min(len(code), k * top_k_factor)
        top_indices = np.argsort(scores)[-num_to_consider:]
        
        top_indices_list = top_indices.tolist()
        random.shuffle(top_indices_list)
        remove_indices = top_indices_list[:k]
        
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        uncovered = counts == 0
        
        while uncovered.any() and time.time() < deadline:
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            counts[neighbors_arr[chosen, :]] += 1
            uncovered = counts == 0
            
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if time.time() < deadline - 15 and best_code is not None:
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 10)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 5:
                perturbed = local_search(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
