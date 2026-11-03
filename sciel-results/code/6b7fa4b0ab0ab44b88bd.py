"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin

    # 1. Generate Candidate Vectors
    # We include weights 1 through 4 to allow for higher norms if beneficial.
    candidates = []
    norms_sq = []
    
    for w in range(1, 5):
        for comb in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, s in zip(comb, signs):
                    v[idx] = s
                candidates.append(v)
                norms_sq.append(w)
    
    candidates = np.array(candidates, dtype=np.int8)
    norms_sq = np.array(norms_sq, dtype=np.int8)
    N = len(candidates)
    
    best_vectors = None
    best_size = 0
    best_K = 0

    # We test different maximum norm squared values K
    # K=2 corresponds to the seed (weight 2 vectors)
    # K=3 allows weight 3 vectors
    # K=4 allows weight 4 vectors
    for K in [2, 3, 4, 5]:
        if time.time() > deadline - 15:
            break
            
        # Filter candidates with norm squared <= K
        mask = norms_sq <= K
        idxs = np.where(mask)[0]
        C = len(idxs)
        if C == 0:
            continue
            
        v_sub = candidates[idxs]
        norms_sub = norms_sq[idxs]
        
        # Compute conflict matrix
        # dist_sq[i, j] < K
        # dist_sq = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j
        dot = v_sub @ v_sub.T
        dist_sq = norms_sub[:, None] + norms_sub[None, :] - 2 * dot
        
        # Conflict if dist_sq < K
        # Note: dist_sq is integer.
        conflict_matrix = dist_sq < K
        # No self conflict
        np.fill_diagonal(conflict_matrix, False)
        
        # Local Search to find Max Independent Set
        # State: selected (boolean array of size C)
        selected = np.zeros(C, dtype=bool)
        
        # Greedy Initialization
        # Sort by degree (number of conflicts) to pick "safe" nodes first?
        # Or random.
        # Let's try random with multiple restarts if time permits.
        
        current_size = 0
        best_local_size = 0
        best_local_selected = np.zeros(C, dtype=bool)
        
        # We run a few iterations of local search
        # Since C can be large, we limit the search time per K
        search_deadline = time.time() + 20 # 20s per K
        
        while time.time() < search_deadline:
            # Greedy Phase
            selected[:] = False
            current_size = 0
            order = np.random.permutation(C)
            
            for i in order:
                if time.time() > search_deadline:
                    break
                if not selected[i]:
                    # Check if i conflicts with any selected node
                    # conflict_matrix[i] is a boolean vector
                    # selected & conflict_matrix[i] gives conflicts
                    if not np.any(selected & conflict_matrix[i]):
                        selected[i] = True
                        current_size += 1
            
            if current_size > best_local_size:
                best_local_size = current_size
                best_local_selected = selected.copy()
            
            # Local Search Phase (1-for-1 and 1-for-2 swaps)
            improved = True
            # Limit inner loop iterations to prevent hanging on a single restart
            inner_iter = 0
            while improved and time.time() < search_deadline and inner_iter < 500:
                improved = False
                inner_iter += 1
                
                # Update conflict counts: number of selected nodes conflicting with i
                # conflict_counts[i] = sum(selected & conflict_matrix[i])
                # This is a matrix-vector multiplication
                conflict_counts = selected @ conflict_matrix.astype(np.int8)
                
                # 1. Try to add nodes with 0 conflicts
                zero_conf = np.where(conflict_counts == 0)[0]
                zero_conf = zero_conf[~selected[zero_conf]]
                
                if len(zero_conf) > 0:
                    # We can add these nodes, but they might conflict with each other.
                    # We need to find an independent set among zero_conf.
                    # Greedy: sort by degree in the subgraph induced by zero_conf
                    sub_conf = conflict_matrix[np.ix_(zero_conf, zero_conf)]
                    degs = np.sum(sub_conf, axis=1)
                    order_z = zero_conf[np.argsort(degs)]
                    
                    added_any = False
                    for i in order_z:
                        if not selected[i]:
                            # Re-check against current selected (which might have changed)
                            if not np.any(selected & conflict_matrix[i]):
                                selected[i] = True
                                current_size += 1
                                added_any = True
                                improved = True
                    if added_any:
                        continue # Restart conflict count calculation
                    
                # 2. Try 1-for-1 Swaps (Size Preserving)
                # Find a selected node u and an unselected node i such that
                # i conflicts with exactly one selected node, which is u.
                # Then we can remove u and add i.
                
                # Heuristic: Pick a random selected node u
                sel_idx = np.where(selected)[0]
                if len(sel_idx) > 0:
                    u = sel_idx[np.random.randint(len(sel_idx))]
                    
                    # Find unselected i such that conflict_counts[i] == 1 and i conflicts with u
                    # conflict_counts is from before any changes in this iteration, 
                    # but we haven't changed selected in this branch yet.
                    # We need i to be unselected.
                    unselected = np.where(~selected)[0]
                    
                    # Filter unselected by conflict_counts == 1
                    candidates_swap = unselected[conflict_counts[unselected] == 1]
                    
                    if len(candidates_swap) > 0:
                        # Check which of these conflict with u
                        # conflict_matrix[candidates_swap, u]
                        valid_swaps = candidates_swap[conflict_matrix[candidates_swap, u]]
                        
                        if len(valid_swaps) > 0:
                            # Pick one randomly
                            i = valid_swaps[np.random.randint(len(valid_swaps))]
                            # Perform swap
                            selected[u] = False
                            selected[i] = True
                            # Size remains same
                            improved = True
                            continue

                # 3. If no improvement, try a kick (remove 1 or 2 nodes)
                if not improved:
                    if current_size > 0:
                        # Remove 1 random node
                        u = sel_idx[np.random.randint(len(sel_idx))]
                        selected[u] = False
                        current_size -= 1
                        improved = True
                    else:
                        break

        # End while time < search_deadline

        if best_local_size > best_size:
            best_size = best_local_size
            best_vectors = v_sub[best_local_selected]
            best_K = K

    # If no vectors found (should not happen), return seed-like result
    if best_vectors is None:
        # Fallback to seed logic
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return best_vectors.astype(np.int64)
# EVOLVE-BLOCK-END
