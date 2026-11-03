import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for 120s limit
    
    total_words = q ** n
    
    # Precompute neighbors as bitsets
    # Word i is represented by its index in base-q expansion
    # neighbor_mask[i] is a Python int where bit j is 1 if word j is within distance 1 of word i
    
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

    neighbor_masks = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 1 << i
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
                idxs.append(j)
        neighbor_masks[i] = mask
        neighbor_indices[i] = idxs

    # Precompute reverse cover: for each word w, list of codewords that cover w
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        mask = neighbor_masks[i]
        m = mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            reverse_cover[j].append(i)
            m ^= lsb

    # Precompute adjacency matrix for O(1) swap checks
    # adj[i][j] = 1 if i is in N(j) (i.e., j covers i)
    adj = [bytearray(total_words) for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbor_indices[i]:
            adj[i][j] = 1

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = (1 << total_words) - 1
        code = []
        active = list(range(total_words))
        random.shuffle(active)
        
        while uncovered:
            best_gain = -1
            best_candidate = -1
            best_count = 0
            
            for i in active:
                cov = neighbor_masks[i] & uncovered
                if cov == 0:
                    continue
                gain = cov.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = i
                    best_count = 1
                elif gain == best_gain:
                    best_count += 1
                    if random.random() < 1.0 / best_count:
                        best_candidate = i
            
            if best_gain == -1:
                break
                
            chosen = best_candidate
            code.append(chosen)
            uncovered &= ~neighbor_masks[chosen]
            
            # Remove candidates that are now fully covered
            new_active = []
            for i in active:
                if i == chosen:
                    continue
                if (neighbor_masks[i] & uncovered) != 0:
                    new_active.append(i)
            active = new_active
            
        return code

    def reduce_code(code):
        code = list(code)
        if not code:
            return code
            
        # Calculate coverage mask for all words
        # We will iteratively remove redundant words
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant
                # Union of neighbors of all other words
                others_mask = 0
                for x in code:
                    if x != c:
                        others_mask |= neighbor_masks[x]
                
                if (neighbor_masks[c] & ~others_mask) == 0:
                    # c is redundant
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    def local_search_optimized(code, deadline):
        code = list(code)
        if not code:
            return code
            
        # Use count-based approach for efficiency
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Tentatively remove c_remove
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                # Check if any count is 0 in the neighborhood of c_remove
                min_count = np.min(counts[neighbor_indices[c_remove]])
                
                if min_count > 0:
                    # Redundant, keep removed
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore counts
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1
                    
                    # Find critical words (those with count 1 in N(c_remove))
                    critical_words = [w for w in neighbor_indices[c_remove] if counts[w] == 1]
                    
                    if not critical_words:
                        continue

                    # Find a replacement c_new that covers all critical_words
                    candidates = neighbor_indices[critical_words[0]]
                    random.shuffle(candidates)
                    
                    best_new = -1
                    for c_new in candidates:
                        if c_new == c_remove:
                            continue
                        
                        # O(1) check using precomputed adjacency matrix
                        valid = True
                        for cw in critical_words[1:]:
                            if not adj[cw][c_new]:
                                valid = False
                                break
                        
                        if valid:
                            best_new = c_new
                            break
                    
                    if best_new != -1:
                        # Perform swap
                        code[i] = best_new
                        
                        # Update counts: -c_remove, +best_new
                        for w in neighbor_indices[c_remove]:
                            counts[w] -= 1
                        for w in neighbor_indices[best_new]:
                            counts[w] += 1
                        
                        improved = True
                        break
                        
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        # Remove k random words
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
            
        # Repair greedily
        covered_mask = 0
        for c in code:
            covered_mask |= neighbor_masks[c]
            
        uncovered = (1 << total_words) - 1
        uncovered &= ~covered_mask
        
        while uncovered and time.time() < deadline:
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                gain = bin(uncovered & neighbor_masks[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            chosen = best_candidates[random.randint(0, len(best_candidates) - 1)]
            code.append(chosen)
            uncovered &= ~neighbor_masks[chosen]
            
        return code

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # 1. Greedy
        code = greedy_construct(seed)
        
        # 2. Reduce
        code = reduce_code(code)
        
        # 3. Local Search
        if time.time() < deadline - 5:
            code = local_search_optimized(code, deadline - 2)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # 4. Perturb and Repair
        if time.time() < deadline - 10 and best_code:
            # Perturb best code
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 5)
            
            # Reduce and Local Search perturbed
            perturbed = reduce_code(perturbed)
            if time.time() < deadline - 5:
                perturbed = local_search_optimized(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
                
        seed += 1
        
    if best_code is None:
        best_code = greedy_construct(0)
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
