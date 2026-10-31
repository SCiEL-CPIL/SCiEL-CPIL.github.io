import numpy as np
import itertools

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    # Max norm squared in the set is 4 (from Type 2 and 3)
    # We want min pairwise distance >= max norm.
    # If we select a subset, the max norm of the subset might be smaller.
    # However, the condition is "minimum pairwise distance is at least the maximum vector norm".
    # This implies we should check the condition on the SELECTED set.
    # But the greedy algorithm usually works on a fixed conflict graph.
    # If we assume max norm is 4 (norm=2), then dist_sq >= 4.
    # If the selected set only has norm 2 vectors, max norm is 2, dist_sq >= 2 is required.
    # Using dist_sq >= 4 is a stricter constraint, so any set satisfying it is valid.
    # It might be suboptimal if we could use norm 2 vectors with dist_sq=2.
    # But Type 1 vectors have norm sqrt(2).
    # If we select only Type 1, max norm is sqrt(2). Dist_sq >= 2.
    # My conflict graph uses dist_sq < 4 as conflict.
    # This means I am forbidding pairs with dist_sq 2 or 3.
    # This is safe but might miss large sets of Type 1 vectors.
    
    # To maximize k, I should probably separate the pools or handle the norm constraint dynamically.
    # However, the current program uses a single pool.
    # Let's stick to the stricter constraint (dist_sq >= 4) to ensure validity, 
    # but this might limit us to Type 2/3 heavy sets.
    # Actually, if I use dist_sq >= 4, I can still select Type 1 vectors if they are orthogonal.
    # Orthogonal Type 1 vectors have dist_sq = 2 + 2 = 4.
    # So orthogonal Type 1 vectors are allowed.
    # Non-orthogonal Type 1 vectors (dot=1) have dist_sq = 2. Forbidden.
    # So this effectively finds an independent set in the graph of non-orthogonal Type 1s 
    # and conflicting Type 2/3s.
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Greedy MIS
    best_selected = []
    best_size = 0
    np.random.seed(42)
    
    for attempt in range(200):
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            neigh_mask = conflict[chosen_global]
            
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    result = candidates[best_selected]
    return result
