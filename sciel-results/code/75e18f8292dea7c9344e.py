"""
Ternary covering code K_3(8,1) construction.

Strategy:
1. Start with the direct product of the ternary tetracode (9 words, length 4, perfect
   covering radius 1) with all words on the remaining 4 coordinates: 9 * 3^4 = 729 words.
2. Greedy removal: iteratively remove redundant codewords (those whose removal
   preserves coverage).
3. Local search: try swaps (remove one, add another) to reduce size while maintaining
   coverage. Focus on codewords with few unique neighbors.
"""

import time
import numpy as np
from itertools import product
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    total_words = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors as a flat array
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    pow3 = [3 ** (n - 1 - i) for i in range(n)]

    for i in range(total_words):
        temp = i
        word = [0] * n
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q

        neighbors[i, 0] = i
        nb_idx = 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                delta = (new_val - orig_val) * pow3[pos]
                neighbors[i, nb_idx] = i + delta
                nb_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        return w

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        uncovered = np.ones(total_words, dtype=bool)
        selected_indices = []
        
        while time.time() < deadline - 20.0:
            counts = uncovered[neighbors].sum(axis=1)
            best_idx = np.argmax(counts)
            best_count = counts[best_idx]
            
            if best_count == 0:
                break
                
            selected_indices.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
            if not uncovered.any():
                break
                
        return selected_indices

    def local_search(initial_indices):
        if not initial_indices:
            return []
            
        code_set = set(initial_indices)
        
        # Coverage counts
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_set:
            counts[neighbors[idx]] += 1
            
        improved = True
        no_improvement_rounds = 0
        max_no_improvement = 100
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to remove redundant codewords
            removable = []
            for c in code_set:
                if np.min(counts[neighbors[c]]) > 1:
                    removable.append(c)
            
            if removable:
                c_to_remove = random.choice(removable) if len(removable) > 1 else removable[0]
                code_set.remove(c_to_remove)
                counts[neighbors[c_to_remove]] -= 1
                improved = True
                no_improvement_rounds = 0
                continue
            
            # 2. If no removals, try 1-swap
            best_c = None
            best_unique_count = float('inf')
            
            for c in code_set:
                uc = np.sum(counts[neighbors[c]] == 1)
                if uc < best_unique_count:
                    best_unique_count = uc
                    best_c = c
                    if uc == 1:
                        break
            
            if best_c is None:
                break
                
            if best_unique_count > 3:
                no_improvement_rounds += 1
                if no_improvement_rounds > max_no_improvement:
                    break
                continue
                
            crit_neighbors = neighbors[best_c][counts[neighbors[best_c]] == 1]
            
            if len(crit_neighbors) == 1:
                w = crit_neighbors[0]
                candidates = neighbors[w]
            elif len(crit_neighbors) == 2:
                w1, w2 = crit_neighbors
                set1 = set(neighbors[w1])
                candidates = [x for x in neighbors[w2] if x in set1]
            elif len(crit_neighbors) == 3:
                w1, w2, w3 = crit_neighbors
                set1 = set(neighbors[w1])
                set2 = set(neighbors[w2])
                candidates = [x for x in neighbors[w3] if x in set1 and x in set2]
            else:
                candidates = []
                
            valid_candidates = [c_new for c_new in candidates if c_new not in code_set]
            
            if valid_candidates:
                c_new = random.choice(valid_candidates)
                
                code_set.remove(best_c)
                code_set.add(c_new)
                
                counts[neighbors[best_c]] -= 1
                counts[neighbors[c_new]] += 1
                
                improved = True
                no_improvement_rounds = 0
                continue
            
            no_improvement_rounds += 1
            if no_improvement_rounds > max_no_improvement:
                break
                
        return list(code_set)

    # Main Strategy:
    # 1. Start with the tetracode product code
    # 2. Run local search
    # 3. Try a few random restarts of greedy + local search if time permits
    
    best_code = []
    
    # First: use the tetracode product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]
    initial_indices = [word_to_idx(w) for w in initial_words] if False else [tuple_to_idx(tuple(w)) for w in initial_words]
    
    # We need a helper to convert word to index
    def tuple_to_idx(word_tuple):
        idx = 0
        for i, val in enumerate(word_tuple):
            idx = idx * q + val
        return idx
    
    initial_indices = [tuple_to_idx(tuple(w)) for w in initial_words]
    
    refined = local_search(initial_indices)
    best_code = refined
    
    # Check time
    if time.time() < deadline - 30.0:
        # Try a few random restarts
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10.0:
                break
            init = greedy_construct(seed=seed)
            refined = local_search(init)
            if len(refined) < len(best_code):
                best_code = refined
    
    return [idx_to_word(idx) for idx in best_code]
