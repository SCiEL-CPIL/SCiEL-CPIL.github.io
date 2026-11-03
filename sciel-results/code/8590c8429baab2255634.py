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

    neighbors_mask = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 0
        idxs = [i]
        mask |= (1 << i)
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
        neighbors_mask[i] = mask
        neighbor_indices[i] = idxs

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for i in candidates:
                if (neighbors_mask[i] & uncovered) == 0:
                    continue
                gain = (neighbors_mask[i] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain == -1:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered &= ~neighbors_mask[chosen]
            
        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                for w in neighbor_indices[c]:
                    counts[w] -= 1
                
                if np.any(counts == 0):
                    for w in neighbor_indices[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    changed = True
            
            code = new_code
        return code

    def local_search_improve(code, deadline):
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for w in neighbor_indices[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Reduction Pass
            new_code = []
            for c in code:
                for w in neighbor_indices[c]:
                    counts[w] -= 1
                if np.any(counts == 0):
                    for w in neighbor_indices[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    improved = True
            code = new_code
            
            if time.time() >= deadline:
                break

            # 2. Swap Pass
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Tentatively remove c_remove
                for w in neighbor_indices[c_remove]:
                    counts[w] -= 1
                
                critical_words = []
                for w in neighbor_indices[c_remove]:
                    if counts[w] == 0:
                        critical_words.append(w)
                
                if not critical_words:
                    # c_remove was redundant
                    improved = True
                    break
                
                critical_mask = 0
                for cw in critical_words:
                    critical_mask |= (1 << cw)
                
                # Build mask of single-covered words
                single_covered_mask = 0
                for w in range(total_words):
                    if counts[w] == 1:
                        single_covered_mask |= (1 << w)
                
                best_w = -1
                best_score = -1
                
                for w in range(total_words):
                    if w == c_remove:
                        continue
                    
                    if (neighbors_mask[w] & critical_mask) != critical_mask:
                        continue
                    
                    score = (neighbors_mask[w] & single_covered_mask).bit_count()
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w != -1:
                    code[i] = best_w
                    
                    for w in neighbor_indices[best_w]:
                        counts[w] += 1
                    
                    improved = True
                    break
                else:
                    for w in neighbor_indices[c_remove]:
                        counts[w] += 1

        return code

    start_time = time.time()
    deadline = start_time + 145
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 10:
                code = local_search_improve(code, deadline - 5)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
        
    if best_code is None:
        return []

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
