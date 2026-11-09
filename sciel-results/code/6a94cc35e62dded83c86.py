# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin
    
    # We need points such that for all pairs i,j: ||p_i - p_j||^2 >= max_k ||p_k||^2
    # This means: 2*||p_i||^2 + 2*||p_j||^2 - 2*p_i·p_j >= max_norm_sq
    # If all points have the same norm R, then: 2R^2 - 2*p_i·p_j >= R^2 => R^2 >= 2*p_i·p_j
    # So p_i·p_j <= R^2/2 for all i != j
    
    # Strategy: use points on a sphere of radius R with integer coordinates.
    # We want to maximize the number of such points with pairwise dot products <= R^2/2.
    
    # Let's try different norms R and find compatible sets of integer vectors.
    
    best_points = None
    best_k = 2
    best_set = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    # Generate candidate vectors with various norms
    # Start with small norms and build up
    
    # For a given norm squared R2, generate all integer vectors with that norm
    # and check compatibility (dot product <= R2/2)
    
    max_R2_to_try = 200  # limit search space
    candidates_by_norm = {}
    
    # Pre-generate candidate vectors for various norms
    for R2 in range(1, max_R2_to_try + 1):
        if time.time() > deadline - 30:
            break
        vecs = _generate_vectors_with_norm(d, R2)
        if vecs:
            candidates_by_norm[R2] = vecs
    
    # For each norm, try to build a large compatible set
    for R2, vecs in candidates_by_norm.items():
        if time.time() > deadline - 60:
            break
        
        # We need pairwise dot products <= R2/2
        # Greedy approach: sort vectors by some heuristic, add if compatible with all existing
        # Also try multiple random orderings
        
        # First, compute dot products efficiently
        n_vecs = len(vecs)
        if n_vecs == 0:
            continue
        
        vecs_arr = np.array(vecs, dtype=np.int64)
        
        # Compute pairwise dot products
        # dot[i,j] = vecs[i] · vecs[j]
        # We need dot[i,j] <= R2/2 for i != j
        
        # Use greedy with multiple restarts
        for attempt in range(20):
            if time.time() > deadline - 60:
                break
            
            if attempt == 0:
                order = np.arange(n_vecs)
            else:
                order = np.random.permutation(n_vecs)
            
            selected = []
            selected_indices = []
            
            for idx in order:
                v = vecs_arr[idx]
                compatible = True
                for sel_idx in selected_indices:
                    if np.dot(v, vecs_arr[sel_idx]) > R2 // 2:
                        compatible = False
                        break
                if compatible:
                    selected.append(v)
                    selected_indices.append(idx)
                
                if len(selected) > best_k:
                    best_k = len(selected)
                    best_set = selected[:]
            
            if len(selected) > best_k:
                best_k = len(selected)
                best_set = selected[:]
    
    # Also try a different approach: use the A1 lattice or other structured constructions
    # In dimension 11, we can try to use vectors from a known construction
    
    # Try adding more points to the best set found
    if best_k >= 2:
        best_arr = np.array(best_set, dtype=np.int64)
        R2_best = np.max(np.sum(best_arr ** 2, axis=1))
        
        # Try to add more vectors that are compatible
        for R2_add in range(1, max_R2_to_try + 1):
            if time.time() > deadline - 30:
                break
            
            # For adding vectors of norm R2_add, we need:
            # ||p_new - p_old||^2 >= R2_best
            # ||p_new||^2 + ||p_old||^2 - 2*p_new·p_old >= R2_best
            # R2_add + R2_old - 2*p_new·p_old >= R2_best
            # p_new·p_old <= (R2_add + R2_old - R2_best) / 2
            
            # This gets complicated with mixed norms. Let's stick to same norm.
            pass
    
    # Final validation and return
    if best_set and len(best_set) >= 2:
        result = np.array(best_set, dtype=np.int64)
        result = np.around(result).astype(np.int64)
        
        # Validate
        k = len(result)
        if k >= 2 and result.shape[1] == d:
            # Check no zero vector
            norms_sq = np.sum(result ** 2, axis=1)
            if np.all(norms_sq > 0):
                max_norm_sq = np.max(norms_sq)
                
                # Check min pairwise distance
                min_dist_sq = np.inf
                for i in range(k):
                    for j in range(i + 1, k):
                        diff = result[i] - result[j]
                        dist_sq = np.sum(diff ** 2)
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                
                if min_dist_sq >= max_norm_sq:
                    return result
    
    # Fallback: return the trivial 2-point solution
    return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)


def _generate_vectors_with_norm(d, R2):
    """Generate all integer vectors in R^d with squared norm R2."""
    if R2 < d:  # minimum norm squared for non-zero integer vector in d dims is d (all ±1)
        # Actually, minimum is 1 (one coordinate is ±1, rest 0)
        # But let's be more careful
        pass
    
    vecs = []
    
    # Recursive generation
    def _gen(coords, remaining_dim, remaining_norm_sq):
        if remaining_dim == 0:
            if remaining_norm_sq == 0:
                vecs.append(tuple(coords))
            return
        
        max_coord = int(np.sqrt(remaining_norm_sq))
        for val in range(-max_coord, max_coord + 1):
            new_remaining = remaining_norm_sq - val * val
            if new_remaining < 0:
                continue
            if new_remaining > remaining_dim * max_coord * max_coord:
                continue
            coords.append(val)
            _gen(coords, remaining_dim - 1, new_remaining)
            coords.pop()
    
    _gen([], d, R2)
    return vecs


# EVOLVE-BLOCK-END
