import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Specific optimization for the task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        cover_count = 1 + n * (q - 1)  # 15
        
        # Precompute neighbors
        # Index i represents the word with base-3 digits d_0...d_6
        # i = d_0 * 3^6 + ... + d_6
        # Neighbor: change d_k to d_k +/- 1 (mod 3? No, just other 2 values)
        
        neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
        
        # Powers of 3
        p3 = [1]
        for _ in range(n):
            p3.append(p3[-1] * 3)
            
        # p3[k] is 3^k. The digit at position pos (0 is most significant) has weight p3[n-1-pos]
        # Let's use standard indexing: idx = sum(d_i * 3^i) where i=0 is least significant.
        # But the problem statement usually implies d_0 is first.
        # Let's stick to: idx = d_0 * 3^(n-1) + ... + d_{n-1}
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % 3
                idx //= 3
            return digits

        def digits_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * 3 + d
            return idx

        for i in range(total_words):
            digits = idx_to_digits(i)
            idxs = [i]
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(3):
                    if new_val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    j = digits_to_idx(new_digits)
                    idxs.append(j)
            for k, j in enumerate(idxs):
                neighbors_arr[i, k] = j

        start_time = time.time()
        deadline = start_time + 115  # Safety margin for 120s limit
        
        best_code = None
        best_size = float('inf')
        
        def greedy_construct(random_seed):
            if random_seed is not None:
                random.seed(random_seed)
            
            uncovered = np.ones(total_words, dtype=np.bool_)
            code = []
            
            while uncovered.any():
                # gains[c] = number of uncovered words covered by codeword c
                # neighbors_arr[c] are the indices of words covered by c
                gains = np.sum(uncovered[neighbors_arr], axis=1)
                
                # Only consider candidates that cover at least one uncovered word
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
            if not code:
                return code
            
            # Convert to set for faster removal? No, list is fine for small sizes.
            # Use counts array
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
                        # Cannot remove
                        counts[neighbors_arr[c, :]] += 1
                        new_code.append(c)
                    else:
                        # Can remove
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
            no_improve_count = 0
            max_no_improve = 5
            
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(len(code)))
                random.shuffle(indices)
                
                # 1-for-0 and 1-for-1 moves
                for i in indices:
                    if time.time() >= deadline:
                        break
                    
                    c_remove = code[i]
                    neighbors_c = neighbors_arr[c_remove, :]
                    counts[neighbors_c] -= 1
                    
                    if np.any(counts == 0):
                        # Need replacement
                        holes = np.where(counts == 0)[0]
                        
                        # Find candidates that cover all holes
                        # A candidate c covers a hole h if h is in neighbors_arr[c]
                        # Since the graph is regular and symmetric, we can check if h is in neighbors_arr[c]
                        # But we need to intersect the neighbor sets of all holes.
                        
                        # Optimization: If there are many holes, it's unlikely a single codeword covers them all.
                        # Max holes covered by one codeword is 15. If len(holes) > 15, no single codeword can cover all.
                        if len(holes) > 15:
                            counts[neighbors_c] += 1
                            continue
                        
                        # Find intersection of neighbors of all holes
                        # neighbors_arr[holes] is (len(holes), 15)
                        # We want indices c such that for all h in holes, h is in neighbors_arr[c]
                        # This is equivalent to: c is in neighbors_arr[h] for all h in holes.
                        
                        candidate_set = set(neighbors_arr[holes[0]].tolist())
                        for h in holes[1:]:
                            if not candidate_set:
                                break
                            candidate_set &= set(neighbors_arr[h].tolist())
                        
                        candidates = list(candidate_set)
                        if not candidates:
                            counts[neighbors_c] += 1
                            continue
                        
                        # Score candidates: how many words do they cover (sum of counts in their neighborhood)
                        # We want to pick one that covers many "high count" words to potentially allow future reductions
                        # Or simply one that covers the most "currently covered" words to be robust.
                        # Actually, the gain of adding c is sum(counts[neighbors_arr[c]])
                        
                        cand_indices = np.array(candidates, dtype=np.int32)
                        # If too many candidates, sample
                        if len(cand_indices) > 50:
                            cand_indices = np.array(random.sample(list(cand_indices), 50), dtype=np.int32)
                        
                        cand_neighbors = neighbors_arr[cand_indices, :]
                        scores = np.sum(counts[cand_neighbors], axis=1)
                        
                        best_idx = np.argmax(scores)
                        best_cand = cand_indices[best_idx]
                        
                        # Check if best_cand is better than c_remove?
                        # c_remove covers neighbors_c. best_cand covers neighbors_arr[best_cand]
                        # If best_cand == c_remove, we just restore.
                        if best_cand != c_remove:
                            code[i] = best_cand
                            counts[neighbors_arr[best_cand, :]] += 1
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            counts[neighbors_c] += 1
                    else:
                        # Can remove
                        code.pop(i)
                        improved = True
                        no_improve_count = 0
                        break
                
                # 2-for-1 move
                if not improved and len(code) >= 2 and time.time() < deadline:
                    # Try a few random pairs
                    pairs_tried = 0
                    max_pairs = 10
                    while pairs_tried < max_pairs and time.time() < deadline:
                        i, j = random.sample(range(len(code)), 2)
                        ci, cj = code[i], code[j]
                        
                        ni = neighbors_arr[ci, :]
                        nj = neighbors_arr[cj, :]
                        
                        counts[ni] -= 1
                        counts[nj] -= 1
                        
                        if np.any(counts == 0):
                            holes = np.where(counts == 0)[0]
                            if len(holes) > 15:
                                counts[ni] += 1
                                counts[nj] += 1
                                pairs_tried += 1
                                continue
                            
                            candidate_set = set(neighbors_arr[holes[0]].tolist())
                            for h in holes[1:]:
                                if not candidate_set:
                                    break
                                candidate_set &= set(neighbors_arr[h].tolist())
                            
                            # Exclude ci and cj
                            candidate_set.discard(ci)
                            candidate_set.discard(cj)
                            
                            if not candidate_set:
                                counts[ni] += 1
                                counts[nj] += 1
                                pairs_tried += 1
                                continue
                            
                            cand_indices = np.array(list(candidate_set), dtype=np.int32)
                            if len(cand_indices) > 50:
                                cand_indices = np.array(random.sample(list(cand_indices), 50), dtype=np.int32)
                            
                            cand_neighbors = neighbors_arr[cand_indices, :]
                            scores = np.sum(counts[cand_neighbors], axis=1)
                            
                            best_idx = np.argmax(scores)
                            best_cand = cand_indices[best_idx]
                            
                            # Remove i and j, add best_cand
                            # Be careful with indices if i > j
                            remove_indices = sorted([i, j], reverse=True)
                            for idx in remove_indices:
                                code.pop(idx)
                            code.append(best_cand)
                            counts[neighbors_arr[best_cand, :]] += 1
                            
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            # Both can be removed (2-for-0)
                            remove_indices = sorted([i, j], reverse=True)
                            for idx in remove_indices:
                                code.pop(idx)
                            improved = True
                            no_improve_count = 0
                            break
                        
                        pairs_tried += 1
                    else:
                        # If loop finished without break, restore counts if we didn't improve
                        # This logic is tricky inside while loop. 
                        # Better structure:
                        pass
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break
                        
            return code

        def perturb_repair(code, k, deadline):
            code = list(code)
            if len(code) <= k:
                return code
            
            # Remove k random codewords
            remove_indices = random.sample(range(len(code)), k)
            for idx in sorted(remove_indices, reverse=True):
                code.pop(idx)
            
            # Repair with greedy
            counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                counts[neighbors_arr[c, :]] += 1
            
            uncovered = counts == 0
            
            while uncovered.any() and time.time() < deadline:
                gains = np.sum(uncovered[neighbors_arr], axis=1)
                valid = gains > 0
                if not valid.any():
                    break
                max_gain = np.max(gains[valid])
                best_candidates = np.where((gains == max_gain) & valid)[0]
                chosen = random.choice(best_candidates.tolist())
                code.append(chosen)
                counts[neighbors_arr[chosen, :]] += 1
                uncovered = counts == 0
            
            return code

        seed = 0
        consecutive_no_improve = 0
        
        while time.time() < deadline:
            # 1. Greedy
            code = greedy_construct(seed)
            code = reduce_code(code)
            
            # 2. Local Search
            if time.time() < deadline - 5:
                code = local_search(code, deadline - 2)
                code = reduce_code(code)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            
            # 3. Perturbation
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
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1
            
            seed += 1
            
            # Stop if we've been stuck for a while and time is running out
            if consecutive_no_improve > 20 and time.time() > deadline - 30:
                break
        
        # Final polish
        if best_code is not None and time.time() < deadline - 3:
            polished = local_search(best_code, deadline - 2)
            polished = reduce_code(polished)
            if len(polished) < len(best_code):
                best_code = polished
        
        result = []
        for idx in best_code:
            digits = idx_to_digits(idx)
            result.append(digits)
        
        return result

    # Fallback for other parameters (not used in this specific task but good for robustness)
    # Simple greedy
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
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
    
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
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
        if not code:
            return code
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
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
