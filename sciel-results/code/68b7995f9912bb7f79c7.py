import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate pool
    # Coords in {-1, 0, 1} and some with 2
    # To keep it manageable, we focus on small norms
    candidates = []
    # Add all vectors with coords in {-1, 0, 1}
    for c in itertools.product([-1, 0, 1], repeat=D):
        v = np.array(c, dtype=np.int64)
        if np.any(v != 0):
            candidates.append(v)
            
    # Add some vectors with a single 2
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    candidates = np.array(candidates)
    
    # Pre-compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Start with Type D vectors (norm sq 2)
    # Find indices of candidates with norm sq 2
    type_d_indices = np.where(norms_sq == 2)[0]
    
    # Initialize with a subset or all?
    # All Type D are mutually compatible.
    current_set_indices = type_d_indices.copy()
    current_set_vectors = candidates[current_set_indices]
    
    best_set = current_set_indices.copy()
    best_size = len(best_set)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    # Pre-compute pairwise distances for the initial set?
    # It's large (220x220).
    
    while time.time() < deadline:
        # Pick a random move
        if np.random.rand() < 0.5:
            # Add a random candidate
            idx = np.random.randint(len(candidates))
            if idx in current_set_indices:
                continue
            
            v_new = candidates[idx]
            norm_new = norms_sq[idx]
            
            # Check compatibility
            # New max norm
            current_max_norm = np.max(np.sum(current_set_vectors * current_set_vectors, axis=1))
            new_max_norm = max(current_max_norm, norm_new)
            
            # Check distances
            # dist_sq = ||v_new - v_i||^2
            # We need dist_sq >= new_max_norm for all i
            diffs = current_set_vectors - v_new
            dists_sq = np.sum(diffs * diffs, axis=1)
            min_dist_sq = np.min(dists_sq)
            
            if min_dist_sq >= new_max_norm:
                # Accept
                current_set_indices = np.append(current_set_indices, idx)
                current_set_vectors = candidates[current_set_indices]
                if len(current_set_indices) > best_size:
                    best_size = len(current_set_indices)
                    best_set = current_set_indices.copy()
            else:
                # Reject or accept with probability
                penalty = new_max_norm - min_dist_sq
                if np.random.rand() < np.exp(-penalty / T):
                    current_set_indices = np.append(current_set_indices, idx)
                    current_set_vectors = candidates[current_set_indices]
                    if len(current_set_indices) > best_size:
                        best_size = len(current_set_indices)
                        best_set = current_set_indices.copy()
        else:
            # Remove a random vector and try to replace
            if len(current_set_indices) == 0:
                continue
            remove_idx_pos = np.random.randint(len(current_set_indices))
            remove_idx_val = current_set_indices[remove_idx_pos]
            
            # Remove it
            current_set_indices = np.delete(current_set_indices, remove_idx_pos)
            current_set_vectors = candidates[current_set_indices]
            
            # Try to add a different one
            for _ in range(10): # Try a few replacements
                add_idx = np.random.randint(len(candidates))
                if add_idx in current_set_indices:
                    continue
                
                v_new = candidates[add_idx]
                norm_new = norms_sq[add_idx]
                
                if len(current_set_vectors) > 0:
                    current_max_norm = np.max(np.sum(current_set_vectors * current_set_vectors, axis=1))
                else:
                    current_max_norm = 0
                    
                new_max_norm = max(current_max_norm, norm_new)
                
                if len(current_set_vectors) > 0:
                    diffs = current_set_vectors - v_new
                    dists_sq = np.sum(diffs * diffs, axis=1)
                    min_dist_sq = np.min(dists_sq)
                else:
                    min_dist_sq = float('inf')
                    
                if min_dist_sq >= new_max_norm:
                    current_set_indices = np.append(current_set_indices, add_idx)
                    current_set_vectors = candidates[current_set_indices]
                    if len(current_set_indices) > best_size:
                        best_size = len(current_set_indices)
                        best_set = current_set_indices.copy()
                    break # Successful replacement, stop trying others for this move
        
        T *= cooling
        if T < 0.1:
            break

    return candidates[best_set]

# EVOLVE-BLOCK-START
# The code above is the implementation
# EVOLVE-BLOCK-END
