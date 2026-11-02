import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_vectors = []
    
    def solve_for_weight(k):
        # Generate candidates
        candidates = []
        indices_combos = list(itertools.combinations(range(D), k))
        for combo in indices_combos:
            for signs in itertools.product([1, -1], repeat=k):
                v = [0] * D
                for idx, s in zip(combo, signs):
                    v[idx] = s
                candidates.append(v)
        
        if not candidates:
            return []
            
        V = np.array(candidates, dtype=np.int8)
        N = len(V)
        
        # Compute dot products
        # V is N x D
        # Dot = V @ V.T
        # This is N x N
        if N > 8000:
             # Too big for dense matrix? 
             # Use optimized greedy loop
             return fast_greedy_solve(candidates, k)
        
        dots = V @ V.T
        threshold = k / 2.0
        # Conflict if dot > threshold
        # Note: diagonal is k, which is > k/2. We ignore self.
        conflict_mask = dots > threshold
        
        # Greedy MIS using conflict mask
        # We want to pick a set of indices such that no two are connected in conflict_mask
        # Heuristic: pick node with min degree in remaining graph
        
        remaining = np.ones(N, dtype=bool)
        selected_indices = []
        
        while np.any(remaining):
            # Count conflicts for each remaining node
            # conflict_mask[:, remaining] sum
            degrees = np.sum(conflict_mask[:, remaining], axis=1)
            # Only consider remaining nodes
            # Avoid inf to prevent conversion errors
            max_deg = np.max(degrees[remaining]) + 1 if np.any(remaining) else 0
            degrees[~remaining] = max_deg
            
            # Pick min degree
            idx = np.argmin(degrees)
            if not remaining[idx]:
                break
            selected_indices.append(idx)
            # Remove idx and its neighbors
            remaining[idx] = False
            neighbors = np.where(conflict_mask[idx, :])[0]
            remaining[neighbors] = False
            
        return [candidates[i] for i in selected_indices]

    def fast_greedy_solve(candidates, k):
        candidates = np.array(candidates, dtype=np.int8)
        N = len(candidates)
        threshold = k / 2.0
        selected = []
        remaining_indices = np.arange(N)
        
        # Shuffle
        np.random.shuffle(remaining_indices)
        
        for idx in remaining_indices:
            if time.time() > deadline - 2.0:
                break
            v = candidates[idx]
            if not selected:
                selected.append(v)
                continue
            
            # Vectorized dot product check
            sel_arr = np.array(selected)
            dots = sel_arr @ v
            if np.any(dots > threshold):
                continue
            selected.append(v)
            
        return [v.tolist() for v in selected]

    for k in range(2, D + 1):
        if time.time() > deadline - 5.0:
            break
        v = solve_for_weight(k)
        if len(v) > len(best_vectors):
            best_vectors = v

    return np.array(best_vectors, dtype=np.int64)
