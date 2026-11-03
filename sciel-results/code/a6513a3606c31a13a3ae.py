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
    # For radius 1, neighbors are self + 1-bit changes
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    # Vectorized generation of neighbors is tricky due to variable length logic in python loops,
    # but total_words is 2187, so a loop is fast enough.
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

    # Precompute reverse neighbors: for each word w, list of codewords that cover w
    # This is the same as neighbors_arr since the relation is symmetric for radius 1
    # (if c covers w, then w covers c). So reverse_neighbors = neighbors_arr.
    # But let's be explicit for clarity and potential future use.
    reverse_neighbors_arr = neighbors_arr  # symmetric for radius 1

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        counts = np.zeros(total_words, dtype=np.int16)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            # Primary: number of uncovered words covered
            # Secondary: number of singly-covered words (to avoid redundancy)
            gains_primary = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains_primary > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains_primary[valid])
            best_candidates = np.where((gains_primary == max_gain) & valid)[0]
            
            if len(best_candidates) > 1:
                # Tie-break: prefer candidates that cover fewer already-singly-covered words
                # counts[w] == 1 means w is covered by exactly one codeword
                singly_covered = (counts == 1)
                # For each candidate, count how many of its neighbors are singly covered
                # We want to MINIMIZE this to avoid creating redundancy
                redundancy_penalty = np.sum(singly_covered[neighbors_arr[best_candidates]], axis=1)
                # Also add a small random component for diversity
                random_scores = np.random.random(len(best_candidates)) * 0.1
                scores = -redundancy_penalty + random_scores
                best_idx = np.argmin(redundancy_penalty + random_scores * 100)  # minimize penalty + noise
                chosen = best_candidates[best_idx]
            else:
                chosen = best_candidates[0]
            
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
            counts[neighbors_arr[chosen, :]] += 1
        
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

    def exhaustive_reduce(code, deadline):
        """Try every codeword for removal with fast replacement search."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    if len(holes) <= 10:
                        # Find candidates covering all holes
                        candidate_set = None
                        for h in holes:
                            h_candidates = set(reverse_neighbors_arr[h, :].tolist())
                            if candidate_set is None:
                                candidate_set = h_candidates
                            else:
                                candidate_set &= h_candidates
                        
                        if candidate_set:
                            candidate_set.discard(c_remove)
                            if candidate_set:
                                # Score and pick best
                                cand_list = list(candidate_set)
                                if len(cand_list) > 30:
                                    cand_list = random.sample(cand_list, 30)
                                
                                best_cand = None
                                best_score = -1
                                code_set = set(code)
                                
                                for cand in cand_list:
                                    cand_neigh = neighbors_arr[cand, :]
                                    redundant = np.sum(counts[cand_neigh] > 0)
                                    if cand in code_set:
                                        redundant -= 100
                                    if redundant > best_score:
                                        best_score = redundant
                                        best_cand = cand
                                
                                if best_cand is not None:
                                    code[i] = best_cand
                                    counts[neighbors_arr[best_cand, :]] += 1
                                    improved = True
                                    break
                    counts[neighbors_c] += 1
                else:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def swap_reduce(code, deadline):
        """Try pairwise swaps to reduce code size."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                if i >= len(code):
                    continue
                
                c1 = code[i]
                neighbors_c1 = neighbors_arr[c1, :]
                counts[neighbors_c1] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Use reverse neighbors to find candidates that cover all holes
                    candidate_set = None
                    for h in holes:
                        h_candidates = set(reverse_neighbors_arr[h, :].tolist())
                        if candidate_set is None:
                            candidate_set = h_candidates
                        else:
                            candidate_set &= h_candidates
                    
                    if candidate_set:
                        # Remove the word we're trying to remove
                        candidate_set.discard(c1)
                        
                        if candidate_set:
                            # Find candidates that are in the current code
                            code_set = set(code)
                            valid_candidates = candidate_set & code_set
                            
                            if valid_candidates:
                                # Pick one (prefer one with max redundancy)
                                best_cand = None
                                best_score = -1
                                for cand in valid_candidates:
                                    cand_neigh = neighbors_arr[cand, :]
                                    redundant = np.sum(counts[cand_neigh] > 0)
                                    if redundant > best_score:
                                        best_score = redundant
                                        best_cand = cand
                                
                                if best_cand is not None:
                                    # Find the index of best_cand in code
                                    j = code.index(best_cand)
                                    code[i] = best_cand
                                    code[j] = c1
                                    counts[neighbors_c1] += 1
                                    counts[neighbors_arr[best_cand, :]] -= 1
                                    improved = True
                                    break
                            else:
                                # Try any candidate from the full set
                                cand_list = list(candidate_set)
                                if len(cand_list) > 20:
                                    cand_list = random.sample(cand_list, 20)
                                
                                best_cand = None
                                best_score = -1
                                for cand in cand_list:
                                    cand_neigh = neighbors_arr[cand, :]
                                    redundant = np.sum(counts[cand_neigh] > 0)
                                    if redundant > best_score:
                                        best_score = redundant
                                        best_cand = cand
                                
                                if best_cand is not None:
                                    # Replace c1 with best_cand (not a swap, but an addition)
                                    # This doesn't reduce size, so skip
                                    pass
                    
                    counts[neighbors_c1] += 1
                else:
                    # Can remove c1
                    code.pop(i)
                    improved = True
                    break
        
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
                    # Identify holes
                    holes = np.where(counts == 0)[0]
                    
                    if len(holes) > 12:
                        # Too many holes to efficiently find a replacement
                        counts[neighbors_c] += 1
                        continue
                    
                    # Find candidates covering all holes using reverse neighbors
                    # A candidate c covers all holes if c is in reverse_neighbors[h] for all h in holes
                    # i.e., c is in the intersection of reverse_neighbors[h] for all h in holes
                    
                    # Start with all possible candidates, then filter
                    # For small hole sets, intersect the candidate sets
                    candidate_set = None
                    for h in holes:
                        h_candidates = set(reverse_neighbors_arr[h, :].tolist())
                        if candidate_set is None:
                            candidate_set = h_candidates
                        else:
                            candidate_set &= h_candidates
                    
                    if not candidate_set:
                        counts[neighbors_c] += 1
                        continue
                    
                    # Remove the word we're trying to remove
                    candidate_set.discard(c_remove)
                    
                    if not candidate_set:
                        counts[neighbors_c] += 1
                        continue
                    
                    # Score candidates: prefer those that maximize overlap with existing coverage
                    # counts currently reflects state AFTER removing c_remove
                    # We want to maximize the "redundant" coverage (words already covered by others)
                    
                    cand_list = list(candidate_set)
                    
                    # If too many candidates, sample
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    best_cand = None
                    best_score = -1
                    
                    code_set = set(code)
                    
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        # Count how many of cand's neighbors are already covered (counts > 0)
                        redundant = np.sum(counts[cand_neigh] > 0)
                        # Also penalize if cand is in the current code (to avoid duplicates)
                        if cand in code_set:
                            redundant -= 100
                        if redundant > best_score:
                            best_score = redundant
                            best_cand = cand
                    
                    if best_cand is not None:
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break
                    else:
                        counts[neighbors_c] += 1
                else:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair
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

    def multi_perturb(code, deadline, num_perturb=3):
        """Try multiple perturbations and keep the best."""
        best = list(code)
        best_size = len(code)
        
        for _ in range(num_perturb):
            if time.time() >= deadline:
                break
            k = random.randint(1, min(4, len(code) - 1))
            perturbed = perturb_repair(code, k, deadline - 5)
            if len(perturbed) < best_size:
                best = perturbed
                best_size = len(perturbed)
        
        return best

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 30:
        # 1. Greedy
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        # 2. Local Search
        if time.time() < deadline - 10:
            code = local_search(code, deadline - 5)
            code = reduce_code(code)
        
        # 3. Swap reduction
        if time.time() < deadline - 8:
            code = swap_reduce(code, deadline - 5)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # 4. Perturbation Strategy (if we have time and code is not perfect)
        # Try to escape local minima by perturbing the current best or current code
        if time.time() < deadline - 20 and best_code is not None:
            # Try multiple perturbations
            perturbed = multi_perturb(best_code, deadline - 10, num_perturb=2)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 8:
                perturbed = local_search(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
                perturbed = swap_reduce(perturbed, deadline - 3)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
    
    # Final polish: try to improve the best code found
    if best_code is not None and time.time() < deadline - 5:
        polished = local_search(best_code, deadline - 3)
        polished = reduce_code(polished)
        polished = swap_reduce(polished, deadline - 2)
        polished = reduce_code(polished)
        polished = exhaustive_reduce(polished, deadline - 1)
        polished = reduce_code(polished)
        if len(polished) < best_size:
            best_code = polished
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
