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
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    
    # Try multiple random tie-breakings
    # The greedy algorithm is:
    # 1. active = all True
    # 2. deg = sum(Conflict, axis=1)
    # 3. While active:
    #    a. Find active nodes with min deg
    #    b. Pick one (randomly if tie, or first)
    #    c. Add to set
    #    d. Remove it and its neighbors
    #    e. Update degrees
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int32)
        selected = []
        
        while active.any():
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
            # neighbors are where Conflict[idx] == 1 and active
            neighbors_mask = Conflict[idx].astype(bool)
            
            # Nodes to remove: idx and neighbors
            # We need to update degrees of remaining active nodes
            
            # Identify nodes to remove
            # removed_mask is True for nodes that will become inactive
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            # Get list of removed indices for efficient degree update
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees:
            # For each remaining active node j, deg[j] -= number of removed neighbors
            # removed_neighbors_count = Conflict[removed_indices, :].sum(axis=0)
            # We only care about active nodes
            if len(removed_indices) > 0:
                # Vectorized update
                # conflict_counts = Conflict[removed_indices, :].sum(axis=0)
                # This creates an array of size N
                # We can do:
                # deg[active] -= Conflict[removed_indices, :].sum(axis=0)[active]
                
                # To save memory/time, we can just do:
                # The sum is over the rows in removed_indices
                # Let's compute the sum for all columns
                removal_counts = Conflict[removed_indices, :].sum(axis=0)
                
                # Update deg for active nodes
                deg[active] -= removal_counts[active]
                
            # Update active mask
            active[removed_mask] = False
            
            # Optional: zero out deg for inactive to keep clean, but not strictly necessary
            # deg[~active] = 0
            
        return selected

    # Run with several seeds
    best_size = 0
    for s in range(50):
        if time.time() > deadline:
            break
        res = run_greedy(s)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            
    # If we still have time, try more seeds or a different heuristic
    # For now, 50 seeds should be fast.
    
    if not best_set:
        # Fallback to simple greedy or all type 1
        best_set = list(range(22)) # Type 1 vectors are indices 0-21
        
    return candidates[best_set].astype(np.int64)
