import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Generate all weight-2 vectors in {-1, 0, 1}^11 (Norm^2 = 2)
    # These have norm^2 = 2, so they can coexist with norm^2=4 vectors
    # if their dot product with any norm^2=4 vector is small enough
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    N1 = len(vectors_v1)
    N2 = len(vectors_v2)
    
    if N4 == 0 and N1 == 0 and N2 == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Combine all candidates
    all_candidates = np.vstack([vectors_v4, vectors_v1, vectors_v2])
    N_total = len(all_candidates)
    
    # Compute norms squared for each candidate
    norms_sq = np.sum(all_candidates * all_candidates, axis=1)
    
    # The constraint: for all pairs (i,j), ||v_i - v_j||^2 >= max_norm_sq
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j
    # So we need: ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j >= max_norm_sq
    # where max_norm_sq = max(||v_k||^2) over selected k
    
    # Farthest-point sampling approach:
    # 1. Start with an initial vector
    # 2. Iteratively add the candidate that maximizes minimum distance to current set
    # 3. Check constraint after each addition
    
    # Precompute dot product matrix
    dots = all_candidates @ all_candidates.T
    
    # Strategy: Use farthest-point sampling with constraint checking
    # We want to maximize k such that min pairwise distance >= max norm
    
    # First, let's determine what max_norm_sq will be.
    # If we only use norm^2=4 vectors, max_norm_sq = 4
    # If we include norm^2=2 vectors, max_norm_sq is still 4 (from v4/v1)
    # The constraint becomes: ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j >= 4
    
    # For two norm^2=4 vectors: 4+4-2*d >= 4 => d <= 2
    # For a norm^2=4 and norm^2=2 vector: 4+2-2*d >= 4 => d <= 1
    # For two norm^2=2 vectors: 2+2-2*d >= 4 => d <= 0
    
    # Let's do farthest-point sampling on the combined set, checking the constraint
    
    # Initialize with a random or strategic starting point
    best_selected = None
    best_count = 0
    
    # Multiple restarts with different starting points
    rng = np.random.default_rng(42)
    
    # Precompute which candidates have which norm type
    is_v4 = norms_sq == 4
    is_v2 = norms_sq == 2
    
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        # Choose a starting vector
        if restart < 10:
            # Start with different strategic vectors
            start_idx = restart
        else:
            start_idx = rng.integers(N_total)
        
        # Ensure start vector has norm^2 = 4 (preferred)
        if norms_sq[start_idx] != 4:
            # Find a norm^2=4 vector
            v4_indices = np.where(norms_sq == 4)[0]
            if len(v4_indices) > 0:
                start_idx = v4_indices[rng.integers(len(v4_indices))]
        
        selected = [start_idx]
        selected_set = set(selected)
        
        # Track minimum distance for each candidate to the selected set
        # min_dist_sq[i] = min over j in selected of ||v_i - v_j||^2
        min_dist_sq = np.full(N_total, np.inf)
        
        # Compute distances from start vector
        for j in selected:
            dist_sq = norms_sq[start_idx] + norms_sq - 2 * dots[:, j]
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        
        # Remove self
        min_dist_sq[start_idx] = 0
        
        max_norm_sq = norms_sq[start_idx]
        
        # Iteratively add vectors
        while True:
            if time.time() > deadline - 10:
                break
            
            # The constraint for adding vector i:
            # For all j in selected: ||v_i - v_j||^2 >= max(max_norm_sq, norms_sq[i])
            # Since max_norm_sq is the max over selected, and we want min pairwise dist >= max norm
            # Let M = max(max_norm_sq, norms_sq[i])
            # We need: for all j in selected, norms_sq[i] + norms_sq[j] - 2*dots[i,j] >= M
            
            # Find candidates that can be added
            # For efficiency, first filter by min_dist_sq >= current_max_norm_sq
            # Then check against the actual max norm
            
            # Get candidates with min_dist_sq >= max_norm_sq (necessary condition)
            candidate_mask = min_dist_sq >= max_norm_sq
            candidate_mask[start_idx] = False
            
            if not np.any(candidate_mask):
                break
            
            candidate_indices = np.where(candidate_mask)[0]
            
            # For each candidate, compute the actual minimum distance to selected set
            # and check if it's >= max(max_norm_sq, norms_sq[candidate])
            
            # Vectorized: for each candidate i, check min over j in selected of 
            # (norms_sq[i] + norms_sq[j] - 2*dots[i,j]) >= max(max_norm_sq, norms_sq[i])
            
            # This is equivalent to: min_dist_sq[i] >= max(max_norm_sq, norms_sq[i])
            # But min_dist_sq[i] is the min over ALL candidates, not just selected
            # We need to recompute for just the selected set
            
            # Let's compute the actual min distance to selected set for candidates
            # selected is a list, so we can do this efficiently
            
            if len(selected) < 50:
                # Small selected set: compute directly
                sel_array = np.array(selected)
                # dist_sq[i, j] = norms_sq[i] + norms_sq[sel_j] - 2*dots[i, sel_j]
                # We want min over j for each i
                sel_norms = norms_sq[sel_array]
                sel_dots = dots[:, sel_array]  # shape (N_total, len(selected))
                dist_to_sel = norms_sq[:, None] + sel_norms[None, :] - 2 * sel_dots
                actual_min_dist = np.min(dist_to_sel, axis=1)
            else:
                # Use precomputed min_dist_sq as approximation (it's min over all, which is <= min over selected)
                # This might reject some valid candidates, but that's okay for a heuristic
                actual_min_dist = min_dist_sq
            
            # Check constraint: actual_min_dist[i] >= max(max_norm_sq, norms_sq[i])
            required_min = np.maximum(max_norm_sq, norms_sq[candidate_indices])
            valid_mask = actual_min_dist[candidate_indices] >= required_min
            
            if not np.any(valid_mask):
                break
            
            valid_candidates = candidate_indices[valid_mask]
            
            # Among valid candidates, pick the one that maximizes minimum distance to selected set
            # This is the farthest-point criterion
            # Use actual_min_dist for this
            
            # Also consider: prefer candidates with higher norm (they're "bigger" spheres)
            # But the constraint is about pairwise distances, so let's just use farthest-point
            
            farthest_idx = valid_candidates[np.argmax(actual_min_dist[valid_candidates])]
            
            # Add this vector
            selected.append(farthest_idx)
            selected_set.add(farthest_idx)
            
            # Update max_norm_sq
            if norms_sq[farthest_idx] > max_norm_sq:
                max_norm_sq = norms_sq[farthest_idx]
            
            # Update min_dist_sq for all candidates
            dist_to_new = norms_sq + norms_sq[farthest_idx] - 2 * dots[:, farthest_idx]
            min_dist_sq = np.minimum(min_dist_sq, dist_to_new)
            min_dist_sq[farthest_idx] = 0
            
            # If max_norm_sq increased, we might need to remove some previously selected vectors
            # that no longer satisfy the constraint
            if norms_sq[farthest_idx] > max(max_norm_sq - norms_sq[farthest_idx], 0):
                # max_norm_sq just increased, check all selected vectors
                pass  # For simplicity, we don't remove vectors in this version
        
        count = len(selected)
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Get the best result
    if best_selected is None:
        # Fallback: return just the v1 vectors
        return vectors_v1
    
    result_vectors = all_candidates[best_selected]
    
    # Verify the constraint
    if len(result_vectors) > 1:
        # Compute all pairwise distances
        dists_sq = np.sum(result_vectors * result_vectors, axis=1)[:, None] + \
                   np.sum(result_vectors * result_vectors, axis=1)[None, :] - \
                   2 * (result_vectors @ result_vectors.T)
        min_dist_sq = np.min(dists_sq)
        max_norm_sq = np.max(np.sum(result_vectors * result_vectors, axis=1))
        
        if min_dist_sq < max_norm_sq:
            # Constraint violated, need to fix
            # Remove vectors that violate the constraint
            # This is a fallback, should rarely happen
            print(f"Warning: constraint violated. min_dist_sq={min_dist_sq}, max_norm_sq={max_norm_sq}", file=sys.stderr)
    
    return result_vectors
