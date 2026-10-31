import numpy as np
import time
from itertools import combinations

D = 11

def construct():
    start_time = time.time()
    deadline = 135  # Safety margin before 150s kill

    # We look for integer vectors with small norm.
    # For equal-norm vectors with norm squared R2, the condition ||v_i - v_j|| >= R (max norm)
    # becomes 2*R2 - 2*v_i.v_j >= R2 => v_i.v_j <= R2/2.
    
    # Let's try R2 = 3. Vectors are of form (±1, ±1, ±1, 0, ...).
    # Dot product condition: v_i.v_j <= 1 (since 3/2 = 1.5, integer dot <= 1).
    # Two such vectors share at most 1 index with same sign, or 2 indices with opposite signs (dot = -1 or -2... wait).
    # v1 = (1,1,1,0..), v2 = (1,1,-1,0..). dot = 1+1-1 = 1. OK.
    # v1 = (1,1,1,0..), v2 = (1,-1,1,0..). dot = 1-1+1 = 1. OK.
    # v1 = (1,1,1,0..), v2 = (1,1,1,0..). dot = 3. NOT OK (same vector).
    # v1 = (1,1,1,0..), v2 = (0,0,0,...) different support.
    
    # Let's generate all candidates with R2 = 3.
    candidates = []
    for combo in combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[combo[0]] = s1
                    v[combo[1]] = s2
                    v[combo[2]] = s3
                    candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Pre-compute compatibility matrix
    # compat[i, j] = True if candidates[i] . candidates[j] <= 1
    G = candidates @ candidates.T
    # We need G[i, j] <= 1 for i != j
    # Note: G[i, i] = 3, which is > 1, but we only care about i != j
    compat = (G <= 1).astype(np.int8)
    np.fill_diagonal(compat, 0)  # Don't pick same vector twice
    
    # Greedy algorithm with random restarts
    best_set = []
    best_size = 0
    
    # We'll do random greedy: pick a random candidate, then repeatedly pick a random compatible candidate
    # Actually, a better heuristic: random order, then greedily pick if compatible with all already picked.
    
    num_trials = 0
    while time.time() < deadline - 5:
        order = np.random.permutation(n_cand)
        selected_indices = []
        selected_mask = np.zeros(n_cand, dtype=bool)
        
        for idx in order:
            if time.time() > deadline - 3:
                break
            
            # Check if idx is compatible with all selected
            # We can maintain a running "compat with all selected" mask
            # But simpler: just check against selected_indices
            if not selected_indices:
                selected_indices.append(idx)
                selected_mask[idx] = True
            else:
                # Check G[idx, selected_indices] <= 1
                # Since G is precomputed, we can do a quick check
                # Using numpy: np.all(G[idx, selected_indices] <= 1)
                # But selected_indices grows, so this gets slower.
                
                # Alternative: maintain a bitmask of "still compatible"
                # Let's use a different approach:
                # active_mask: which candidates are still compatible with ALL selected so far
                # Initially active_mask = all True
                # When we pick idx, new_active = active_mask & compat[idx, :]
                
                # Let's restructure:
                pass
        
        # Let's use the active_mask approach for speed
        active_mask = np.ones(n_cand, dtype=bool)
        selected_indices = []
        
        for idx in order:
            if time.time() > deadline - 3:
                break
            
            if not active_mask[idx]:
                continue
            
            selected_indices.append(idx)
            active_mask[idx] = False  # Remove from active (already selected)
            # Update active_mask: intersect with compat[idx, :]
            # Only keep candidates that are compatible with idx
            # But we also need to ensure they're compatible with all previously selected
            # active_mask already represents "compatible with all previously selected"
            # So we just AND with compat[idx, :]
            active_mask &= (compat[idx, :] == 1)
            
            # Optional: if no active candidates left, break early
            # if not active_mask.any():
            #     break
        
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_set = list(selected_indices)
            num_trials += 1
    
    if best_size == 0:
        # Fallback
        return np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return candidates[best_set]
