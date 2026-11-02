"""Program for kissing number in dimension 11 using greedy augmentation of small-norm integer vectors."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct():
    # Generate candidate vectors with small norms
    # Norm squared 2: (1, 1, 0, ...)
    # Norm squared 3: (1, 1, 1, 0, ...)
    # Norm squared 4: (2, 0, ...), (1, 1, 1, 1, ...)
    # Norm squared 5: (2, 1, 0, ...), (1, 1, 1, 1, 1, ...)
    
    candidates = []
    
    # Helper to add permutations of a pattern
    def add_pattern(pattern):
        # pattern is a tuple of non-zero values, length L
        # We need to place them in D dimensions
        # And assign signs? No, pattern includes signs?
        # Let's generate base patterns with positive signs, then apply sign flips?
        # Simpler: generate all combinations of indices and signs.
        
        pass

    # Direct generation for small norms
    # Norm sq 2
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates.append(np.array(v, dtype=np.int64))
                    
    # Norm sq 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(np.array(v, dtype=np.int64))

    # Norm sq 4
    # Type 1: (2, 0, ...)
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            candidates.append(np.array(v, dtype=np.int64))
            
    # Type 2: (1, 1, 1, 1, 0, ...)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = [0]*D
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    candidates.append(np.array(v, dtype=np.int64))

    # Sort by norm squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    sorted_indices = np.argsort(norms_sq)
    sorted_candidates = candidates[sorted_indices]
    
    selected = []
    max_norm_sq = 0
    
    for v in sorted_candidates:
        v_norm_sq = np.sum(v * v)
        new_max_norm_sq = max(max_norm_sq, v_norm_sq)
        
        compatible = True
        for u in selected:
            diff = v - u
            dist_sq = np.sum(diff * diff)
            if dist_sq < new_max_norm_sq:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
            max_norm_sq = new_max_norm_sq
    
    if not selected:
        # Fallback to D_11 if something went wrong
        base_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    base_vectors.append(v)
        return np.array(base_vectors, dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
