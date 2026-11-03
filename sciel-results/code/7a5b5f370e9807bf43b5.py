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

    # Precompute neighbor indices and masks
    neighbor_indices = [[] for _ in range(total_words)]
    masks = [0] * total_words
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        mask_val = 1 << i
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
                mask_val |= (1 << j)
        neighbor_indices[i] = idxs
        masks[i] = mask_val
        
    # Build adjacency matrix (still useful for some ops, but masks are primary now)
    # We can skip full adj matrix if we use masks for everything, 
    # but greedy uses vectorized ops. Let's stick to masks for speed.
    
    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        # Use a bitmask for uncovered words
        # Bit i is 1 if word i is uncovered
        uncovered_mask = (1 << total_words) - 1
        code = []
        
        while uncovered_mask != 0:
            # Find candidate with max gain
            # Gain of c is popcount(masks[c] & uncovered_mask)
            max_gain = 0
            best_candidates = []
            
            # Iterate all candidates
            for c in range(total_words):
                gain = (masks[c] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [c]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(c)
            
            if max_gain == 0:
                break # Should not happen if space is coverable
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~masks[chosen]
            
        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbor_indices[c]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbor_indices[c]] -= 1
                
                if (counts == 0).any():
                    counts[neighbor_indices[c]] += 1
                    new_code.append(c)
                else:
                    changed = True
            
            code = new_code
        return code

    def local_search_improve(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbor_indices[c]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                counts[neighbor_indices[c_remove]] -= 1
                
                if (counts == 0).any():
                    counts[neighbor_indices[c_remove]] += 1
                    
                    # Find critical words (count == 1 after removal)
                    critical_indices = np.where(counts == 1)[0]
                    if len(critical_indices) == 0:
                        continue
                        
                    # Build critical mask
                    critical_mask = 0
                    for w in critical_indices:
                        critical_mask |= (1 << w)
                    
                    # Find a codeword w such that masks[w] covers all critical words
                    # i.e., critical_mask is a subset of masks[w]
                    # critical_mask & ~masks[w] == 0
                    best_w = -1
                    for w in range(total_words):
                        if w == c_remove:
                            continue
                        if (critical_mask & ~masks[w]) == 0:
                            best_w = w
                            break
                    
                    if best_w != -1:
                        code[i] = best_w
                        counts[neighbor_indices[c_remove]] -= 1
                        counts[neighbor_indices[best_w]] += 1
                        improved = True
                        break
                
                else:
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def two_one_exchange(code, counts, deadline):
        """Try 2-1 exchange: remove two codewords, add one that covers their union."""
        code = list(code)
        counts = counts.copy()
        improved = False
        
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            for j in range(i+1, len(code)):
                if time.time() >= deadline:
                    break
                
                c1 = code[i]
                c2 = code[j]
                
                counts[neighbor_indices[c1]] -= 1
                counts[neighbor_indices[c2]] -= 1
                
                if (counts == 0).any():
                    counts[neighbor_indices[c1]] += 1
                    counts[neighbor_indices[c2]] += 1
                    continue
                
                # Find critical words
                critical_indices = np.where(counts == 0)[0]
                if len(critical_indices) == 0:
                    counts[neighbor_indices[c1]] += 1
                    counts[neighbor_indices[c2]] += 1
                    continue
                
                critical_mask = 0
                for w in critical_indices:
                    critical_mask |= (1 << w)
                
                # Find a single codeword that covers all critical words
                best_w = -1
                for w in range(total_words):
                    if w == c1 or w == c2:
                        continue
                    if (critical_mask & ~masks[w]) == 0:
                        best_w = w
                        break
                
                if best_w != -1:
                    # Perform the exchange
                    counts[neighbor_indices[best_w]] += 1
                    new_code = []
                    for k in range(len(code)):
                        if k != i and k != j:
                            new_code.append(code[k])
                    new_code.append(best_w)
                    code = new_code
                    improved = True
                    break
                else:
                    counts[neighbor_indices[c1]] += 1
                    counts[neighbor_indices[c2]] += 1
        
        return code, counts, improved

    def perturb_and_repair(code, counts, deadline):
        """Perturb by removing a random codeword, then repair."""
        code = list(code)
        counts = counts.copy()
        
        if len(code) == 0:
            return code, counts, False
        
        # Remove a random codeword
        idx = random.randint(0, len(code)-1)
        c_remove = code[idx]
        counts[neighbor_indices[c_remove]] -= 1
        code.pop(idx)
        
        # Repair: add codewords until all covered
        # Use bitmask for uncovered to speed up repair
        # Reconstruct uncovered mask from counts
        uncovered_indices = np.where(counts == 0)[0]
        if len(uncovered_indices) == 0:
            return code, counts, True
            
        uncovered_mask = 0
        for w in uncovered_indices:
            uncovered_mask |= (1 << w)
            
        while uncovered_mask != 0 and time.time() < deadline:
            # Find best candidate for uncovered_mask
            max_gain = 0
            best_candidates = []
            for c in range(total_words):
                gain = (masks[c] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [c]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(c)
            
            if max_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            counts[neighbor_indices[chosen]] += 1
            uncovered_mask &= ~masks[chosen]
            
        return code, counts, True

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or (best_code is None):
            if time.time() < deadline - 10:
                # Try local search
                code = local_search_improve(code, deadline - 5)
                code = reduce_code_fast(code)
                
                # Try 2-1 exchanges
                counts = np.zeros(total_words, dtype=np.int16)
                for c in code:
                    counts[neighbor_indices[c]] += 1
                
                for _ in range(3):
                    if time.time() >= deadline - 5:
                        break
                    code, counts, improved = two_one_exchange(code, counts, deadline - 5)
                    if not improved:
                        break
                
                # Try perturb and repair a few times
                for _ in range(2):
                    if time.time() >= deadline - 5:
                        break
                    counts = np.zeros(total_words, dtype=np.int16)
                    for c in code:
                        counts[neighbor_indices[c]] += 1
                    code, counts, _ = perturb_and_repair(code, counts, deadline - 5)
                    code = reduce_code_fast(code)
                    code = local_search_improve(code, deadline - 5)
                    code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
