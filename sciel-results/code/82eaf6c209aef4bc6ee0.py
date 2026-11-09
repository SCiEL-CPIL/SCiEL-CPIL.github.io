import numpy as np
from itertools import product

D = 11

def construct():
    candidates = []
    for entry_tuple in product([-1, 0, 1], repeat=D):
        v = np.array(entry_tuple, dtype=np.int64)
        if np.count_nonzero(v) >= 2:
            candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    norms = np.linalg.norm(candidates, axis=1)
    
    # Get unique norms and indices for each norm
    unique_norms = np.unique(norms)
    norm_groups = {}
    for u_norm in unique_norms:
        # Use a tolerance for float comparison
        mask = np.isclose(norms, u_norm)
        norm_groups[u_norm] = np.where(mask)[0]
    
    best_selected = []
    best_count = 0
    n_restarts = 50
    
    for r in range(n_restarts):
        selected_idx = []
        
        # Process norms in ascending order
        for u_norm in sorted(unique_norms):
            group_indices = norm_groups[u_norm].copy()
            np.random.shuffle(group_indices)
            
            for i in group_indices:
                v = candidates[i]
                norm_v = norms[i]
                
                if not selected_idx:
                    selected_idx.append(i)
                    continue
                
                sel_vecs = candidates[selected_idx]
                sel_norms = norms[selected_idx]
                
                diffs = sel_vecs - v
                dists = np.linalg.norm(diffs, axis=1)
                max_norms = np.maximum(sel_norms, norm_v)
                
                if np.all(dists >= max_norms - 1e-9):
                    selected_idx.append(i)
        
        if len(selected_idx) > best_count:
            best_count = len(selected_idx)
            best_selected = [candidates[i] for i in selected_idx]
            
    if not best_selected:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return np.array(best_selected, dtype=np.int64)
