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
                    
                    # Try to find a single codeword that covers all holes
                    # Vectorized: for each candidate, check if it covers all holes
                    # candidate covers hole h if h is in neighbors_arr[candidate]
                    # This is equivalent to: for each hole, the candidate must be in the "pre-image"
                    # But we can check: candidate covers all holes iff 
                    # np.all(np.isin(neighbors_arr[candidate], holes)) ... no, that's not right.
                    # Actually: candidate covers hole h iff h is in neighbors_arr[candidate]
                    # So we need: for all h in holes, h is in neighbors_arr[candidate]
                    # Equivalently: holes is a subset of neighbors_arr[candidate]
                    
                    # Vectorized approach: 
                    # For each candidate c, check if all holes are covered
                    # holes_mask = np.zeros(total_words, dtype=bool)
                    # holes_mask[holes] = True
                    # covered_by_c = neighbors_arr[c]  # indices of words covered by c
                    # all_covered = np.all(holes_mask[covered_by_c])  # wrong direction
                    
                    # Better: for each hole h, the set of candidates that cover h is
                    # the set of c such that h is in neighbors_arr[c]
                    # This is the "reverse neighbor" - preimage of h under the neighbor map.
                    # We can precompute this, but for now use a simpler approach.
                    
                    # Simple approach: for each candidate in the code (excluding c1),
                    # check if it covers all holes
                    for j in range(len(code)):
                        if j == i:
                            continue
                        c2 = code[j]
                        neighbors_c2 = neighbors_arr[c2, :]
                        # Check if c2 covers all holes
                        # holes are covered by c2 if all holes are in neighbors_c2
                        # i.e., for each h in holes, there exists k such that neighbors_c2[k] == h
                        # Equivalent: np.all(np.isin(holes, neighbors_c2))
                        if len(holes) <= 5:  # Only do this for small hole sets
                            if np.all(np.isin(holes, neighbors_c2)):
                                # Swap c1 with c2
                                code[i] = c2
                                code[j] = c1
                                counts[neighbors_c1] += 1
                                counts[neighbors_c2] -= 1
                                improved = True
                                break
                    if improved:
                        break
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
                    
                    if len(holes) > 8:
                        # Too many holes to efficiently find a replacement
                        counts[neighbors_c] += 1
                        continue
                    
                    # Find candidates covering all holes using vectorized approach
                    # A candidate c covers all holes if holes is a subset of neighbors_arr[c]
                    # Equivalently: for each h in holes, c must be in the "reverse neighbors" of h
                    # Precompute reverse neighbors: rev_neighbors[h] = set of c such that h in neighbors_arr[c]
                    # But we can do it on the fly for small hole sets
                    
                    # Vectorized: for each candidate, count how many holes it covers
                    # holes_mask = np.zeros(total_words, dtype=bool)
                    # holes_mask[holes] = True
                    # For each candidate c, covered_holes = np.sum(holes_mask[neighbors_arr[c]])
                    # We want covered_holes == len(holes)
                    
                    holes_mask = np.zeros(total_words, dtype=np.bool_)
                    holes_mask[holes] = True
                    
                    # For all candidates, count how many of the holes they cover
                    # covered_counts[c] = number of holes covered by candidate c
                    covered_counts = np.sum(holes_mask[neighbors_arr], axis=1)
                    
                    # Candidates that cover all holes
                    full_cover = covered_counts == len(holes)
                    full_cover[c_remove] = False  # Exclude the word we're removing
                    
                    if not np.any(full_cover):
                        counts[neighbors_c] += 1
                        continue
                    
                    candidate_indices = np.where(full_cover)[0]
                    
                    # Score candidates: prefer those that maximize overlap with existing coverage
                    # counts currently reflects state AFTER removing c_remove
                    # We want to minimize the "new" coverage (words that were uncovered)
                    # and maximize the "redundant" coverage (words already covered by others)
                    
                    # For each candidate, compute:
                    # - num_redundant: number of words in neighbors_arr[cand] with counts > 0
                    # - We want to MAXIMIZE num_redundant to minimize wasted coverage
                    
                    cand_list = candidate_indices.tolist()
                    
                    # If too many candidates, sample
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    best_cand = None
                    best_score = -1
                    
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        # Count how many of cand's neighbors are already covered (counts > 0)
                        redundant = np.sum(counts[cand_neigh] > 0)
                        # Also penalize if cand is in the current code (to avoid duplicates)
                        if cand in code:
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
        polished = local_search(best_code, deadline - 2)
        polished = reduce_code(polished)
        polished = swap_reduce(polished, deadline - 1)
        if len(polished) < best_size:
            best_code = polished
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
