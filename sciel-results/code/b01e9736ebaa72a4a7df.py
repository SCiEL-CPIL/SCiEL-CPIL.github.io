import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    cover_size = 1 + n * (q - 1)
    
    # 1. Precompute neighbors and reverse covers
    # neighbors[i]: list of indices covered by codeword i
    # reverse_cover[w]: bitset of codewords that cover word w
    # cover_bitsets[i]: bitset of words covered by codeword i
    
    neighbors = []
    for i in range(total_words):
        w = []
        idx = i
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                j = 0
                for val in new_w:
                    j = j * q + val
                idxs.append(j)
        neighbors.append(idxs)
    
    # Convert neighbors to numpy arrays for fast count updates
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbors]
    
    # Precompute cover bitsets (codeword -> words)
    cover_bitsets = []
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        cover_bitsets.append(bs)
        
    # Precompute reverse cover bitsets (word -> codewords)
    # reverse_cover_bitsets[w] is a bitset where bit i is 1 if codeword i covers word w
    reverse_cover_bitsets = [0] * total_words
    for i in range(total_words):
        for j in neighbors[i]:
            reverse_cover_bitsets[j] |= (1 << i)
            
    all_words_mask = (1 << total_words) - 1
    all_codewords_mask = (1 << total_words) - 1
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered_mask = all_words_mask
        code = []
        code_bitset = 0
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            # Find codeword with max gain
            # Optimization: Only check codewords that cover at least one uncovered word?
            # Hard to filter without iterating. Just iterate all.
            for i in range(total_words):
                gain = bin(uncovered_mask & cover_bitsets[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if best_gain == 0:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            code_bitset |= (1 << chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
            
        return code, code_bitset

    def reduce_code(code, code_bitset):
        code = list(code)
        if not code:
            return code, code_bitset
            
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
            
        changed = True
        while changed:
            changed = False
            new_code = []
            new_code_bitset = 0
            for c in code:
                counts[neighbor_arrays[c]] -= 1
                if np.any(counts[neighbor_arrays[c]] == 0):
                    # Cannot remove
                    counts[neighbor_arrays[c]] += 1
                    new_code.append(c)
                    new_code_bitset |= (1 << c)
                else:
                    # Can remove
                    changed = True
            code = new_code
            code_bitset = new_code_bitset
        return code, code_bitset

    def local_search(code, code_bitset, deadline):
        code = list(code)
        if not code:
            return code, code_bitset
            
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                    
                c_remove = code[i]
                nbrs = neighbor_arrays[c_remove]
                counts[nbrs] -= 1
                
                # Check if any word becomes uncovered
                uncovered_words = nbrs[counts[nbrs] == 0]
                
                if len(uncovered_words) == 0:
                    # Can remove
                    code.pop(i)
                    code_bitset &= ~(1 << c_remove)
                    improved = True
                    break
                else:
                    # Restore counts for now
                    counts[nbrs] += 1
                    
                    # Try to find a swap: a codeword c' not in code that covers all uncovered_words
                    # Intersection of reverse_cover for each uncovered word
                    candidate_mask = all_codewords_mask
                    for w in uncovered_words:
                        candidate_mask &= reverse_cover_bitsets[w]
                    
                    # c' must not be in current code
                    candidate_mask &= ~code_bitset
                    
                    if candidate_mask:
                        # Pick a random candidate from the mask
                        # Convert mask to list of indices
                        # Efficient way: iterate bits? Or just random sampling from range?
                        # Since mask is sparse, finding bits is better.
                        # But for simplicity and speed in Python, we can just iterate 0..total_words
                        # 2187 is small.
                        candidates = []
                        m = candidate_mask
                        idx = 0
                        while m:
                            if m & 1:
                                candidates.append(idx)
                            m >>= 1
                            idx += 1
                            
                        if candidates:
                            best_w = random.choice(candidates)
                            # Perform swap
                            counts[nbrs] -= 1 # Remove c_remove
                            counts[neighbor_arrays[best_w]] += 1 # Add best_w
                            code[i] = best_w
                            code_bitset &= ~(1 << c_remove)
                            code_bitset |= (1 << best_w)
                            improved = True
                            break
                            
        return code, code_bitset

    def repair_greedy(code, code_bitset, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
            
        while time.time() < deadline:
            uncovered_mask = 0
            # Find uncovered words
            # np.where(counts == 0)
            unc_idx = np.where(counts == 0)[0]
            if len(unc_idx) == 0:
                break
                
            # Build uncovered mask
            for w in unc_idx:
                uncovered_mask |= (1 << w)
                
            # Find best codeword to add
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                # Gain is number of uncovered words covered by i
                # We can use bitset intersection
                gain = bin(uncovered_mask & cover_bitsets[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if best_gain == 0:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            code_bitset |= (1 << chosen)
            counts[neighbor_arrays[chosen]] += 1
            
        return code, code_bitset

    def perturb_repair(code, code_bitset, k, deadline):
        code = list(code)
        code_bitset = code_bitset
        if len(code) <= k:
            return code, code_bitset
            
        remove_indices = random.sample(range(len(code)), k)
        # Remove from code and update bitset/counts?
        # Easier: just create new code list and let repair handle counts.
        # But repair needs initial counts.
        # Let's just remove from list and rebuild counts in repair.
        for idx in sorted(remove_indices, reverse=True):
            c = code.pop(idx)
            code_bitset &= ~(1 << c)
            
        return repair_greedy(code, code_bitset, deadline)

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_code_bitset = 0
    best_size = float('inf')
    
    # Initial construction
    code, code_bitset = greedy_construct(0)
    code, code_bitset = reduce_code(code, code_bitset)
    code, code_bitset = local_search(code, code_bitset, deadline - 10)
    
    best_code = code
    best_code_bitset = code_bitset
    best_size = len(code)
    
    seed = 1
    while time.time() < deadline - 5:
        strategy = seed % 3
        
        if strategy == 0:
            k = random.randint(1, 2)
            perturbed_code, perturbed_bitset = perturb_repair(best_code, best_code_bitset, k, deadline - 5)
        elif strategy == 1:
            k = random.randint(3, 5)
            perturbed_code, perturbed_bitset = perturb_repair(best_code, best_code_bitset, k, deadline - 5)
        else:
            # Swap-based perturbation: remove 2, add 2 random, then repair
            perturbed_code = list(best_code)
            perturbed_bitset = best_code_bitset
            if len(perturbed_code) >= 2:
                remove_idx = random.sample(range(len(perturbed_code)), 2)
                for idx in sorted(remove_idx, reverse=True):
                    c = perturbed_code.pop(idx)
                    perturbed_bitset &= ~(1 << c)
                # Add 2 random codewords
                for _ in range(2):
                    new_c = random.randint(0, total_words - 1)
                    if not (perturbed_bitset & (1 << new_c)):
                        perturbed_code.append(new_c)
                        perturbed_bitset |= (1 << new_c)
            perturbed_code, perturbed_bitset = repair_greedy(perturbed_code, perturbed_bitset, deadline - 5)
            
        # Local search on perturbed code
        perturbed_code, perturbed_bitset = local_search(perturbed_code, perturbed_bitset, deadline - 5)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
            best_code_bitset = perturbed_bitset
            
        seed += 1
        
        # Fresh restarts
        if seed % 8 == 0 and time.time() < deadline - 20:
            fresh_code, fresh_bitset = greedy_construct(seed)
            fresh_code, fresh_bitset = reduce_code(fresh_code, fresh_bitset)
            fresh_code, fresh_bitset = local_search(fresh_code, fresh_bitset, deadline - 10)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code
                best_code_bitset = fresh_bitset

    result = []
    for idx in best_code:
        w = []
        idx_val = idx
        for _ in range(n):
            w.append(idx_val % q)
            idx_val //= q
        w.reverse()
        result.append(w)
    
    return result
