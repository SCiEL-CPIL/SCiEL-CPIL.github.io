import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit
    
    # Generate candidate pool: all vectors with norm^2 = 4 (three non-zero entries, each ±1)
    # There are C(11,3) * 2^3 = 165 * 8 = 1320 such vectors
    # Also include norm^2 = 2 vectors (two non-zero entries, each ±1): C(11,2) * 4 = 55 * 4 = 220
    # Total pool: 1540 vectors
    
    candidates = []
    
    # Norm-2 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    # Norm-4 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
    
    n_cand = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # Precompute pairwise dot products for efficiency
    # dot[i,j] = candidates[i] . candidates[j]
    dot = candidates @ candidates.T
    
    # The constraint: for any two distinct points u, v in our set,
    # ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2*u.v
    # So we need: ||u||^2 + ||v||^2 - 2*u.v >= max(||u||^2, ||v||^2)
    # WLOG assume ||u||^2 >= ||v||^2, then: ||u||^2 + ||v||^2 - 2*u.v >= ||u||^2
    # => ||v||^2 - 2*u.v >= 0
    # => 2*u.v <= ||v||^2
    
    # For norm-2 vectors: ||v||^2 = 2, so 2*u.v <= 2, i.e., u.v <= 1
    # For norm-4 vectors: ||v||^2 = 4, so 2*u.v <= 4, i.e., u.v <= 2
    
    # More generally: for u with norm^2 = nu and v with norm^2 = nv:
    # If nu >= nv: need 2*u.v <= nv
    # If nv >= nu: need 2*u.v <= nu
    # So: 2*u.v <= min(nu, nv)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compatibility matrix: compat[i,j] = True if candidates[i] and candidates[j] can both be in the set
    # We need to check the constraint for each pair
    # For efficiency, build a bitmask or boolean matrix
    
    # Two vectors i and j are compatible if 2*dot[i,j] <= min(norms_sq[i], norms_sq[j])
    # Also they must not be the same vector (distinct points)
    
    # Build compatibility as a boolean matrix (n_cand x n_cand)
    # compat[i,j] = True means i and j can coexist
    min_norm = np.minimum(norms_sq[:, None], norms_sq[None, :])
    # Constraint: 2*dot[i,j] <= min_norm[i,j]
    # Note: for i=j, dot[i,i] = norms_sq[i], and 2*norms_sq[i] <= norms_sq[i] is false unless norms_sq[i]=0
    # So i != j is automatically enforced since 2*x <= x implies x <= 0, but norms_sq >= 2
    
    compat = (2 * dot <= min_norm)
    
    # Now we want to find a large clique in the compatibility graph
    # Greedy approach with random restarts
    
    best_size = 0
    best_set = []
    
    # Sort candidates by degree (number of compatible partners) descending for greedy
    degrees = np.sum(compat, axis=1)
    
    # Do multiple random restarts with different orderings
    n_restarts = 200
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # First try: sort by degree descending
            order = np.argsort(-degrees)
        else:
            # Random perturbation of degree-based order
            order = np.arange(n_cand)
            # Mix in some randomness
            if rng.random() < 0.5:
                order = order[rng.permutation(n_cand)]
            else:
                # Partially randomize
                mask = rng.random(n_cand) < 0.3
                order = np.concatenate([order[mask][rng.permutation(mask.sum())], order[~mask]])
        
        selected = []
        selected_mask = np.zeros(n_cand, dtype=bool)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            v = idx
            # Check if v is compatible with all selected
            # selected_mask has True for selected indices
            # We need compat[v, selected_mask] to all be True
            if not selected_mask.any():
                compatible = True
            else:
                compatible = np.all(compat[v, selected_mask])
            
            if compatible:
                selected.append(v)
                selected_mask[v] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            # If we've found something significantly better, we can continue searching
            # but we might also want to try local improvement
            if best_size > 300:
                # Try a few more restarts but with less randomness
                pass
    
    # Local improvement: try to swap out poorly-connected points for better ones
    # Or try adding points that weren't selected
    if best_size > 0 and time.time() < deadline:
        # Convert best_set to a set for fast lookup
        best_set_arr = np.array(best_set)
        best_mask = np.zeros(n_cand, dtype=bool)
        best_mask[best_set_arr] = True
        
        # Try to add any unselected point that's compatible with all selected
        unselected = np.where(~best_mask)[0]
        for u in unselected:
            if time.time() > deadline:
                break
            if np.all(compat[u, best_mask]):
                best_set.append(u)
                best_mask[u] = True
                best_size += 1
    
    if best_size == 0:
        # Fallback to norm-2 vectors only
        result = candidates[:220]
    else:
        result = candidates[best_set]
    
    # Final validation: verify the constraint
    # For all pairs, check ||u-v||^2 >= max(||u||^2, ||v||^2)
    n = result.shape[0]
    result_norms_sq = np.sum(result**2, axis=1)
    
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = np.sum((result[i] - result[j])**2)
            max_norm = max(result_norms_sq[i], result_norms_sq[j])
            if dist_sq < max_norm:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to norm-2 vectors which are guaranteed valid
        result = candidates[:220]
    
    return result
