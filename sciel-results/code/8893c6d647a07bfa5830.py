import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def generate_vectors(weight):
        vecs = []
        for positions in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    def greedy_mis(vecs, threshold):
        N = len(vecs)
        if N == 0:
            return []
        
        # Compute dot products
        dots = vecs @ vecs.T
        # Conflict if dot > threshold
        conflicts = dots > threshold
        np.fill_diagonal(conflicts, False)
        
        alive = np.ones(N, dtype=bool)
        selected = []
        
        # Optimization: Precompute neighbor lists?
        # For N=1320, N^2 matrix is fine.
        
        while alive.any():
            if time.time() > deadline:
                break
                
            # Find an alive index
            # Using np.where is O(N)
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Pick the first one (greedy)
            idx = alive_indices[0]
            selected.append(idx)
            
            # Mark idx and its neighbors as dead
            # conflicts[idx] is a boolean array
            to_remove = conflicts[idx].copy()
            to_remove[idx] = True
            alive[to_remove] = False
            
        return selected

    # Try Weight-3
    v3 = generate_vectors(3)
    # Threshold for weight 3 (norm sqrt(3)):
    # d^2 >= 3 => 3+3-2dot >= 3 => 2dot <= 3 => dot <= 1.5 => dot <= 1
    selected_3 = greedy_mis(v3, 1)
    k3 = len(selected_3)
    
    # Try Weight-2
    v2 = generate_vectors(2)
    # Threshold for weight 2 (norm sqrt(2)):
    # d^2 >= 2 => 2+2-2dot >= 2 => 2dot <= 2 => dot <= 1
    selected_2 = greedy_mis(v2, 1)
    k2 = len(selected_2)
    
    # We know all weight-2 vectors are compatible, so k2 should be 220
    # But greedy might miss some if not perfect? No, if all compatible, greedy picks all.
    
    if k3 > k2:
        return v3[selected_3]
    else:
        return v2[selected_2]
