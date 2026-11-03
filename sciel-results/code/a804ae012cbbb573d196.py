import time
import random
import numpy as np
from itertools import product

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

    # Precompute neighbor indices
    neighbor_indices = [[] for _ in range(total_words)]
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
        neighbor_indices[i] = idxs
    
    # Build adjacency matrix: adj[i][j] = 1 if j is in neighbor_indices[i]
    adj = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in neighbor_indices[i]:
            adj[i][j] = 1

    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            gains = (adj * uncovered).sum(axis=1)
            max_gain = gains.max()
            
            if max_gain == 0:
                break
            
            # Find all candidates with max gain
            best_candidates = np.where(gains == max_gain)[0]
            
            # Pick randomly from best candidates
            chosen = int(random.choice(best_candidates))
            code.append(chosen)
            uncovered[neighbor_indices[chosen]] = 0
            
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
                    
                    critical_words = [w for w in neighbor_indices[c_remove] if counts[w] == 1]
                    
                    if not critical_words:
                        continue

                    best_w = -1
                    for w in range(total_words):
                        if w == c_remove:
                            continue
                        covers_all = all(adj[w][cw] == 1 for cw in critical_words)
                        if covers_all:
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

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or (best_code is None):
            if time.time() < deadline - 5:
                code = local_search_improve(code, deadline - 2)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
