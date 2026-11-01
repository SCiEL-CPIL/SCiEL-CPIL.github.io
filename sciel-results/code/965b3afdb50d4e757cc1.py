import itertools
import numpy as np
import random

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Strategy 1: Integer Linear Programming for small instances
    # This is an exact method, distinct from the greedy heuristic.
    if num_words <= 1500:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import lil_matrix, csr_matrix
            
            # Build A: A[i, j] = 1 if d(word_i, word_j) <= radius
            A = lil_matrix((num_words, num_words), dtype=int)
            
            # To speed up distance calculation, we can precompute?
            # For small N, direct calculation is fine.
            for i, w in enumerate(all_words):
                for j, w2 in enumerate(all_words):
                    dist = sum(1 for k in range(n) if w[k] != w2[k])
                    if dist <= radius:
                        A[i, j] = 1
                        
            A = A.tocsr()
            
            c = np.ones(num_words)
            # Constraint: sum_{j} A[i, j] x_j >= 1 for all i
            constraints = LinearConstraint(A, 1, np.inf)
            bounds = Bounds(0, 1)
            integrality = np.ones(num_words)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.success:
                x = res.x
                codewords = []
                for i in range(num_words):
                    if x[i] > 0.5:
                        codewords.append(list(all_words[i]))
                return codewords
        except Exception:
            pass # Fall back to heuristic

    # Strategy 2: Fast Greedy with Bitmasks
    # This is an optimized version of greedy set cover, using bit operations.
    
    # 1. Compute coverage masks
    # mask[i] is an integer where bit j is 1 if word i covers word j
    masks = [0] * num_words
    for i, w in enumerate(all_words):
        mask = 0
        for j, w2 in enumerate(all_words):
            dist = sum(1 for k in range(n) if w[k] != w2[k])
            if dist <= radius:
                mask |= (1 << j)
        masks[i] = mask
        
    # 2. Greedy Selection
    uncovered = (1 << num_words) - 1
    codewords = []
    
    while uncovered:
        best_i = -1
        best_count = -1
        
        for i in range(num_words):
            count = (masks[i] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_i = i
                
        if best_count == 0:
            break
            
        codewords.append(list(all_words[best_i]))
        uncovered &= ~masks[best_i]
        
    return codewords
