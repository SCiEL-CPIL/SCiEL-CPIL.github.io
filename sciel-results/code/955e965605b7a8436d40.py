import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    # Condition for compatibility: distance^2 >= max_norm^2 = 4
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # We need 8 - 2*v_i.v_j >= 4  =>  4 >= 2*v_i.v_j  =>  v_i.v_j <= 2
    # So conflict if v_i.v_j > 2.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Greedy construction with min-degree heuristic
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        # Use int64 to prevent overflow/casting errors
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            # Find active nodes
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            # Get degrees of active nodes
            active_deg = deg[active_indices]
            
            # Find min degree
            min_deg = np.min(active_deg)
            
            # Find all active nodes with min degree
            candidates_for_pick = active_indices[active_deg == min_deg]
            
            # Pick one randomly
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors
            neighbors_mask = Conflict[idx].astype(bool)
            
            # Nodes to remove: idx and neighbors
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            # Get list of removed indices for efficient degree update
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees:
            # For each remaining active node j, deg[j] -= number of removed neighbors
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                
                # Update deg for active nodes
                deg[active] -= removal_counts[active]
                
            # Update active mask
            active[removed_mask] = False
            
        return selected

    # Run with several seeds
    best_size = 0
    best_set = []
    
    # Helper for local search: try to improve a set
    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffled order to break symmetry
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find candidates j not in selected_set that conflict ONLY with i
                # and do NOT conflict with any other element in selected_set
                
                # Efficient check:
                # Let S = selected_set.
                # For j not in S, let C_j = set of elements in S that conflict with j.
                # We want |C_j| == 1 and that element is i.
                
                # Compute conflicts of i with all N
                conflicts_with_i = Conflict[i].astype(bool)
                
                # For each j not in S:
                # Check if j conflicts with any k in S \ {i}
                # This is expensive.
                
                # Alternative:
                # We know j must conflict with i. So j is in neighbors of i.
                # Let neighbors_of_i = np.where(conflicts_with_i)[0]
                # Filter neighbors_of_i to those not in S.
                # For each candidate j in that list:
                #   Check if j conflicts with any k in S \ {i}.
                #   If not, j is a "valid replacement candidate" for i (single conflict).
                
                neighbors_of_i = np.where(conflicts_with_i)[0]
                # Exclude those already in S
                valid_js = [j for j in neighbors_of_i if j not in selected_set]
                
                # Now find pairs (j1, j2) in valid_js such that:
                # 1. j1 and j2 are compatible (Conflict[j1, j2] == 0)
                # 2. j1 does not conflict with any k in S \ {i}
                # 3. j2 does not conflict with any k in S \ {i}
                
                # Precompute "bad" sets for S \ {i}?
                # Let's just check on the fly for small lists.
                # If valid_js is large, this is slow.
                
                # Optimization:
                # A vector j is "compatible with S\{i}" if Conflict[j, S\{i}].sum() == 0.
                
                compatible_with_rest = []
                for j in valid_js:
                    # Check conflicts with S \ {i}
                    # S is a set. We can iterate.
                    # To speed up, if len(S) is large, this is bad.
                    # However, greedy sets are usually smallish relative to N.
                    conflict_count = 0
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict[j, k]:
                            conflict_count += 1
                            break
                    if conflict_count == 0:
                        compatible_with_rest.append(j)
                
                # Now look for a pair in compatible_with_rest that are compatible with each other
                found_pair = False
                for a in range(len(compatible_with_rest)):
                    if found_pair:
                        break
                    j1 = compatible_with_rest[a]
                    for b in range(a + 1, len(compatible_with_rest)):
                        j2 = compatible_with_rest[b]
                        if not Conflict[j1, j2]:
                            # Swap i with j1, j2
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                            
        return list(selected_set)

    # Try multiple greedy runs
    for seed in range(50):
        if time.time() > deadline:
            break
        res = run_greedy(seed)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
        # Try local search
        if len(res) > 0:
            improved_set = local_search(res)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set

    # Convert indices back to vectors
    if best_size > 0:
        return candidates[best_set].astype(np.int64)
    else:
        # Fallback to a trivial valid set if something went wrong
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:5])
    # Verify
    norms = np.sum(res**2, axis=1)
    max_norm = np.max(norms)
    dists = np.sum((res[:, np.newaxis, :] - res[np.newaxis, :, :])**2, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    print(f"Max Norm: {max_norm}, Min Dist: {min_dist}, Valid: {min_dist >= max_norm}")
    print(f"Count: {len(res)}")
