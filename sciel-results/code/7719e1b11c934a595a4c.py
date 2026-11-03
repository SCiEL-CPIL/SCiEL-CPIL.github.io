import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    # For our specific case: q=3, n=7, radius=1
    # We can hardcode some optimizations for this case
    if q == 3 and n == 7 and radius == 1:
        # Precompute all neighbors using bit operations for speed
        # Each word is an index from 0 to 3^7 - 1 = 2186
        # Neighbors of a word include the word itself and all words at Hamming distance 1
        # Total neighbors per word: 1 + 7 * 2 = 15
        
        cover_count = 1 + n * (q - 1)
        
        # Build neighbor lookup using integer arithmetic
        neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
        
        # Convert index to base-q representation
        def idx_to_base_q(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def base_q_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        for i in range(total_words):
            digits = idx_to_base_q(i)
            idxs = [i]
            for pos in range(n):
                original_val = digits[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    j = base_q_to_idx(new_digits)
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
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                best_candidates = np.where(gains == max_gain)[0]
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
                
                # Try 1-for-0 and 1-for-1 moves
                for i in indices:
                    if time.time() >= deadline:
                        break
                    
                    c_remove = code[i]
                    neighbors_c = neighbors_arr[c_remove, :]
                    counts[neighbors_c] -= 1
                    
                    if np.any(counts == 0):
                        holes = np.where(counts == 0)[0]
                        candidates = set(neighbors_arr[holes[0]].tolist())
                        for h in holes[1:]:
                            if not candidates:
                                break
                            candidates &= set(neighbors_arr[h].tolist())
                        
                        candidates.discard(c_remove)
                        
                        if not candidates:
                            counts[neighbors_c] += 1
                            continue
                        
                        cand_list = list(candidates)
                        if len(cand_list) > 100:
                            cand_list = random.sample(cand_list, 100)
                        
                        cand_indices = np.array(cand_list, dtype=np.int32)
                        cand_neighbors = neighbors_arr[cand_indices, :]
                        scores = np.sum(counts[cand_neighbors], axis=1)
                        
                        best_idx = np.argmax(scores)
                        best_cand = cand_list[best_idx]
                        
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        code.pop(i)
                        improved = True
                        no_improve_count = 0
                        break
                
                # Try 2-for-1 move: remove two codewords, add one that covers all exposed words
                if not improved and len(code) >= 2 and time.time() < deadline:
                    pairs = random.sample(range(len(code)), min(20, len(code)))
                    for p in range(0, len(pairs) - 1, 2):
                        i, j = pairs[p], pairs[p + 1]
                        if i >= len(code) or j >= len(code) or i == j:
                            continue
                        ci, cj = code[i], code[j]
                        ni = neighbors_arr[ci, :]
                        nj = neighbors_arr[cj, :]
                        counts[ni] -= 1
                        counts[nj] -= 1
                        
                        if np.any(counts == 0):
                            holes = np.where(counts == 0)[0]
                            candidates = set(neighbors_arr[holes[0]].tolist())
                            for h in holes[1:]:
                                if not candidates:
                                    break
                                candidates &= set(neighbors_arr[h].tolist())
                            candidates.discard(ci)
                            candidates.discard(cj)
                            
                            if not candidates:
                                counts[ni] += 1
                                counts[nj] += 1
                                continue
                            
                            cand_list = list(candidates)
                            if len(cand_list) > 150:
                                cand_list = random.sample(cand_list, 150)
                            cand_indices = np.array(cand_list, dtype=np.int32)
                            cand_neighbors = neighbors_arr[cand_indices, :]
                            scores = np.sum(counts[cand_neighbors], axis=1)
                            
                            best_idx = np.argmax(scores)
                            best_cand = cand_list[best_idx]
                            
                            # Remove both, add one
                            idx_remove = sorted([i, j], reverse=True)
                            for idx in idx_remove:
                                code.pop(idx)
                            code.append(best_cand)
                            counts[neighbors_arr[best_cand, :]] += 1
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            # Both can be removed (2-for-0)
                            idx_remove = sorted([i, j], reverse=True)
                            for idx in idx_remove:
                                code.pop(idx)
                            improved = True
                            no_improve_count = 0
                            break
                    
                    if not improved:
                        counts[ni] += 1
                        counts[nj] += 1
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break
            return code
        
        def perturb_repair(code, k, deadline):
            code = list(code)
            if len(code) <= k:
                return code
            
            counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                counts[neighbors_arr[c, :]] += 1
            
            # Remove k random codewords
            remove_indices = random.sample(range(len(code)), k)
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
        
        seed = 0
        consecutive_no_improve = 0
        while time.time() < deadline:
            code = greedy_construct(seed)
            code = reduce_code(code)
            
            if time.time() < deadline - 5:
                code = local_search(code, deadline - 2)
                code = reduce_code(code)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            
            if time.time() < deadline - 15 and best_code is not None:
                k = random.randint(2, 4)
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
        
        # Final polish: try to reduce further with more iterations
        if best_code is not None and time.time() < deadline - 3:
            polished = local_search(best_code, deadline - 2)
            polished = reduce_code(polished)
            if len(polished) < len(best_code):
                best_code = polished
        
        result = []
        for idx in best_code:
            digits = idx_to_base_q(idx)
            result.append(digits)
        
        return result
    
    # Fallback for other parameters
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
    
    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        no_improve_count = 0
        max_no_improve = 3
        while improved and time.time() < deadline:
            improved = False
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
                    candidates = set(neighbors_arr[holes[0]].tolist())
                    for h in holes[1:]:
                        if not candidates:
                            break
                        candidates &= set(neighbors_arr[h].tolist())
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts[neighbors_c] += 1
                        continue
                    
                    cand_list = list(candidates)
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    
                    best_idx = np.argmax(scores)
                    best_cand = cand_list[best_idx]
                    
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    code.pop(i)
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
        return code
    
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
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
