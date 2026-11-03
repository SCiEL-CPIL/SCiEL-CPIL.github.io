import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
    # Precompute neighbors
    # neighbors_arr[i] contains the indices of all words within Hamming distance 'radius' from i
    # For radius 1, this is the word itself and all words differing in exactly one position.
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        idxs = [i]
        for pos in range(n):
            original_val = (i // (q ** (n - 1 - pos))) % q
            for new_val in range(q):
                if new_val == original_val:
                    continue
                j = i + (new_val - original_val) * (q ** (n - 1 - pos))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
    
    # Build inverse neighbors: for each word, which codewords cover it?
    # Actually, for local search we need to know if removing a codeword creates holes.
    # We can maintain a count array.
    
    def idx_to_list(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return list(reversed(t))
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any() and time.time() < deadline - 10:
            # Calculate gain: how many currently uncovered words does each candidate cover?
            # We can speed this up by only considering candidates that cover at least one uncovered word.
            # But computing gains for all is vectorized and fast enough for 2187 size.
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            
            # Get all candidates with max gain
            best_candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to diversify
            chosen = random.choice(best_candidates.tolist())
            
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
            
        return code
    
    def reduce_code(code):
        """Remove any codeword that is not essential for coverage."""
        if not code:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            # Shuffle to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                c = code[i]
                neighbors_c = neighbors_arr[c, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    # Essential, keep it
                    counts[neighbors_c] += 1
                    new_code.append(c)
                else:
                    # Redundant, remove it
                    changed = True
            
            code = new_code
        return code
    
    def local_search_1opt(code, dl):
        """Try to replace a codeword with another to potentially allow further reductions, or just optimize.
        Here we focus on: if we can't remove, can we swap to a 'better' position?
        A 'better' position might be one that overlaps more with others (increasing counts) 
        or simply one that allows a subsequent removal.
        However, the most direct improvement is often just finding a codeword that can be removed.
        The reduce_code function handles removals.
        Local search 1-opt for covering codes often involves:
        1. Try removing c. If valid, do it. (Handled by reduce_code)
        2. If not valid, try swapping c with c' such that the new set is valid.
           This doesn't change size, but might enable a future removal.
           Or, try swapping c with c' where c' is not in code, and the new set is valid.
           This also doesn't change size.
           
        Actually, a more effective 1-opt is:
        For each c in code:
          Remove c.
          If valid, keep removed.
          If invalid, find holes H.
          Find candidates C' that cover all H.
          For each c' in C':
             Swap c with c'.
             If the new code is valid (it is by construction of covering H), 
             does it allow a removal?
             This is complex.
             
        Let's stick to a simpler heuristic:
        Try to swap c with c' (where c' is not in code) if it increases the minimum count in the covered set,
        making the code "denser" and potentially easier to reduce later.
        """
        code = list(code)
        if not code:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= dl:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Find candidates that cover all holes
                    # Intersection of inverse neighbors of holes
                    # We don't have precomputed inverse neighbors for speed, so we check candidates
                    # A candidate c' is valid if neighbors_arr[c'] covers all holes.
                    # i.e., set(holes).issubset(set(neighbors_arr[c']))
                    
                    # To speed up, we can filter candidates:
                    # Any valid c' must be in the union of neighbors of the holes? No, c' must cover the holes,
                    # so c' must be a neighbor of each hole.
                    # So c' must be in intersection of inv_neighbors[h] for h in holes.
                    
                    # Let's just iterate through all possible c' and check? 2187 checks.
                    # Or use the precomputed neighbors_arr.
                    # c' is valid if for all h in holes, h is in neighbors_arr[c'].
                    # This is equivalent to: holes is a subset of neighbors_arr[c'].
                    
                    # Optimization: pick the hole with the fewest neighbors to start the candidate pool.
                    # Since all words have the same number of neighbors in a regular graph, this doesn't help directly.
                    # But we can check candidates from the neighbors of the first hole.
                    
                    first_hole = holes[0]
                    # Candidates must be neighbors of first_hole
                    cand_pool = neighbors_arr[first_hole]
                    
                    valid_candidates = []
                    for cand in cand_pool:
                        # Check if cand covers all other holes
                        cand_neigh = neighbors_arr[cand]
                        # Check if holes are subset of cand_neigh
                        # Vectorized check: 
                        # np.all(np.isin(holes, cand_neigh)) is slow due to isin.
                        # Better: create a mask?
                        # Or: np.sum(np.isin(cand_neigh, holes)) == len(holes)
                        
                        # Even better:
                        # We know holes is small (usually 1 or 2).
                        # If len(holes) == 1: any neighbor of hole is valid.
                        # If len(holes) > 1: check intersection.
                        
                        if len(holes) == 1:
                            valid_candidates.append(cand)
                        else:
                            # Check if all holes are in cand_neigh
                            # np.all(np.isin(holes, cand_neigh))
                            # Let's use a precomputed set for holes? No, holes changes.
                            # Use: np.intersect1d(holes, cand_neigh).size == len(holes)
                            if np.intersect1d(holes, cand_neigh).size == len(holes):
                                valid_candidates.append(cand)
                    
                    # Exclude the original c_remove and existing code?
                    # We can swap with any c', even if it's already in code (then it's a duplicate, which is bad).
                    # We should avoid duplicates.
                    code_set = set(code)
                    code_set.discard(c_remove)
                    
                    best_cand = None
                    best_score = -1
                    
                    for cand in valid_candidates:
                        if cand in code_set:
                            continue
                        
                        # Score: prefer candidates that have high overlap with existing code (high counts)
                        # This makes the code "denser".
                        cand_neigh = neighbors_arr[cand, :]
                        score = np.sum(counts[cand_neigh])
                        
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    
                    if best_cand is not None:
                        # Perform swap
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break # Restart loop
                    else:
                        # No valid swap found, restore
                        counts[neighbors_c] += 1
                else:
                    # Safe to remove
                    code.pop(i)
                    improved = True
                    break # Restart loop
        
        return code

    def local_search_2opt(code, dl):
        """Try swapping two codewords c1, c2 with c1', c2' such that size decreases or stays same but allows reduction.
        Actually, a simpler 2-opt is:
        Try removing c1. If invalid, try removing c2. If invalid, try swapping c1 with c1' and c2 with c2'?
        Too complex.
        
        Let's implement a "Swap" move that replaces one codeword with another, 
        but we already did that in 1-opt.
        
        A true 2-opt for minimization:
        Try to remove c1 and c2 simultaneously.
        If valid, remove both.
        If invalid, try to replace c1 with c1' and c2 with c2' such that the new set is valid and allows removal of something else?
        
        Given time constraints, let's focus on a stronger 1-opt and more restarts.
        The 1-opt above tries to swap to a "denser" position.
        
        Let's add a "Two-Opt Removal" check:
        For each pair (i, j), try removing code[i] and code[j].
        If valid, remove both.
        """
        code = list(code)
        if len(code) < 2:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            n_code = len(code)
            # Sample pairs to avoid O(N^2) full scan if N is large, but N is ~180, so N^2 is ~32k, which is fine.
            # We can shuffle indices.
            indices = list(range(n_code))
            random.shuffle(indices)
            
            for i_idx in range(n_code):
                if time.time() >= dl:
                    break
                i = indices[i_idx]
                if i >= len(code): continue # code might have changed? No, we only break on success.
                c1 = code[i]
                n1 = neighbors_arr[c1, :]
                
                for j_idx in range(i_idx + 1, n_code):
                    j = indices[j_idx]
                    if j >= len(code): continue
                    c2 = code[j]
                    n2 = neighbors_arr[c2, :]
                    
                    # Try removing both
                    counts[n1] -= 1
                    counts[n2] -= 1
                    
                    if not np.any(counts == 0):
                        # Valid removal of two!
                        # Remove from end to preserve indices? Or just pop.
                        # We are in a loop, so let's just mark for removal.
                        # Easiest: pop j then i (assuming j > i)
                        code.pop(j)
                        code.pop(i)
                        improved = True
                        break # Restart outer loop
                    else:
                        # Restore
                        counts[n2] += 1
                        counts[n1] += 1
                        
                if improved:
                    break
        
        return code

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 5:
        # Generate initial code
        code = greedy_construct(seed)
        
        # Reduce
        code = reduce_code(code)
        
        # Local Search 1-opt
        if time.time() < deadline - 10:
            code = local_search_1opt(code, deadline - 5)
            code = reduce_code(code)
            
        # Local Search 2-opt (pair removal)
        if time.time() < deadline - 10:
            code = local_search_2opt(code, deadline - 5)
            code = reduce_code(code)
            
        # Iterate local search until no improvement or time up
        prev_size = len(code)
        while time.time() < deadline - 10:
            code = local_search_1opt(code, deadline - 5)
            code = reduce_code(code)
            code = local_search_2opt(code, deadline - 5)
            code = reduce_code(code)
            
            if len(code) >= prev_size:
                break
            prev_size = len(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
        # Perturbation
        if time.time() < deadline - 20 and best_code is not None:
            k = random.randint(2, 5)
            if len(best_code) > k:
                perturbed = list(best_code)
                remove_indices = random.sample(range(len(perturbed)), k)
                for idx in sorted(remove_indices, reverse=True):
                    perturbed.pop(idx)
                
                # Repair
                counts = np.zeros(total_words, dtype=np.int16)
                for c in perturbed:
                    counts[neighbors_arr[c, :]] += 1
                uncovered = counts == 0
                
                while uncovered.any() and time.time() < deadline - 10:
                    gains = np.sum(uncovered[neighbors_arr], axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0:
                        break
                    best_candidates = np.where(gains == max_gain)[0]
                    chosen = random.choice(best_candidates.tolist())
                    perturbed.append(chosen)
                    counts[neighbors_arr[chosen, :]] += 1
                    uncovered = counts == 0
                
                # Polish
                perturbed = reduce_code(perturbed)
                if time.time() < deadline - 10:
                    perturbed = local_search_1opt(perturbed, deadline - 5)
                    perturbed = reduce_code(perturbed)
                    perturbed = local_search_2opt(perturbed, deadline - 5)
                    perturbed = reduce_code(perturbed)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed
        
        seed += 1
    
    result = []
    if best_code:
        for idx in best_code:
            result.append(idx_to_list(idx))
    else:
        # Fallback
        for idx in range(total_words):
            result.append(idx_to_list(idx))
            
    return result
