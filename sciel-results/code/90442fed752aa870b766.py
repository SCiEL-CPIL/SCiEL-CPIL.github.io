"""Combined program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict masks
    # dots = vectors_v4 @ vectors_v4.T is O(N^3) or O(N^2 * D)
    # N=5280, D=11. 5280^2 * 11 ~ 300M ops.
    # Numpy matmul is fast.
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to Python integer bitmasks
    conflict_masks = [0] * N4
    # This loop is O(N^2 / word_size) effectively if we optimize, but N=5280 is small enough for Python loops with bit ops?
    # 5280 * 5280 = 27M iterations.
    # Each iteration: check if conflict, set bit.
    # This might take a few seconds in Python.
    # We can vectorize this?
    # Or just use the numpy array to build masks.
    
    for i in range(N4):
        mask = 0
        # Get indices of conflicts for i
        # neighbors = np.where(conflicts[i])[0]
        # for j in neighbors: mask |= (1 << j)
        # This is slow.
        
        # Alternative:
        # Use numpy to get bits?
        # No, just use the boolean array.
        # 5280 * 5280 is 27 million.
        # Python loop 27M times might take 10-20s.
        # We have 145s. It's acceptable.
        
        # Optimization:
        # We can iterate over set bits in conflicts[i]?
        # conflicts[i] is a numpy bool array.
        # np.nonzero is fast.
        nbrs = np.nonzero(conflicts[i])[0]
        for j in nbrs:
            mask |= (1 << j)
        conflict_masks[i] = mask
        
    full_mask = (1 << N4) - 1
    
    rng = np.random.default_rng(42)
    
    def greedy_init():
        in_set_mask = 0
        available = full_mask
        # Min-degree greedy is hard with bitmasks without precomputed degrees.
        # Use random greedy with some heuristic?
        # Just pick random available node?
        # Or pick node with min degree in available?
        # Let's use a simple random greedy first, then local search improves it.
        # Or use the numpy greedy from before?
        # Let's use a fast numpy greedy for init.
        
        # Re-use logic from before but return mask
        in_set_bool = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = np.array([conflicts[i].sum() for i in range(N4)], dtype=np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set_bool[chosen] = True
            nbrs = np.nonzero(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        # Convert to mask
        mask = 0
        indices = np.where(in_set_bool)[0]
        for i in indices:
            mask |= (1 << i)
        return mask

    def local_search(mask):
        current_mask = mask
        current_size = current_mask.bit_count()
        best_mask = current_mask
        best_size = current_size
        
        # Extract indices of current set for random removal
        def get_indices(m):
            inds = []
            while m:
                lsb = m & -m
                inds.append(lsb.bit_length() - 1)
                m ^= lsb
            return inds
        
        # Precompute neighbors list for faster access if needed?
        # We have conflict_masks.
        
        while time.time() < deadline - 5:
            # Get current indices
            inds = get_indices(current_mask)
            if not inds:
                break
                
            # Pick random node to remove
            v = inds[rng.integers(len(inds))]
            
            # Remove v
            new_mask = current_mask & ~(1 << v)
            
            # Rebuild blocked
            blocked = 0
            mm = new_mask
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                blocked |= conflict_masks[idx]
                mm ^= lsb
                
            # Greedy fill
            free = full_mask & ~(new_mask | blocked)
            while free:
                # Pick a node from free
                # To avoid bias, we can pick random?
                # For speed, pick lowest.
                lsb = free & -free
                idx = lsb.bit_length() - 1
                new_mask |= (1 << idx)
                blocked |= conflict_masks[idx]
                free = full_mask & ~(new_mask | blocked)
                
            new_size = new_mask.bit_count()
            
            if new_size > current_size:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # No improvement. 
                # To escape local optima, we can accept worse moves?
                # Or just restart?
                # Let's try a few random restarts if stuck.
                # For now, just continue. The random removal changes the landscape.
                pass
                
        return best_mask, best_size

    # Main Loop
    best_global_mask = 0
    best_global_size = 0
    
    # Initial run
    init_mask = greedy_init()
    best_mask, best_size = local_search(init_mask)
    best_global_mask = best_mask
    best_global_size = best_size
    
    # Multiple restarts
    while time.time() < deadline - 10:
        # Perturb best or random init?
        # Random init is safer to explore different basins.
        init_mask = greedy_init()
        
        # Or perturb best?
        # Let's mix: 50% random, 50% perturb best.
        if rng.random() < 0.5 and best_global_size > 0:
            # Perturb best
            inds = []
            m = best_global_mask
            while m:
                lsb = m & -m
                inds.append(lsb.bit_length() - 1)
                m ^= lsb
            if len(inds) > 5:
                to_remove = rng.choice(inds, size=3, replace=False)
                mask = best_global_mask
                for v in to_remove:
                    mask &= ~(1 << v)
                init_mask = mask
            else:
                init_mask = greedy_init()
        else:
            init_mask = greedy_init()
            
        best_mask, best_size = local_search(init_mask)
        if best_size > best_global_size:
            best_global_size = best_size
            best_global_mask = best_mask

    # Final result
    # Extract indices from best_global_mask
    final_indices = []
    m = best_global_mask
    while m:
        lsb = m & -m
        final_indices.append(lsb.bit_length() - 1)
        m ^= lsb
        
    result_v4 = vectors_v4[final_indices]
    
    # Add W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
