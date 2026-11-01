import numpy as np
import itertools
import time

D = 11

def _greedy_mis_size(n, dot_matrix):
    """
    Greedy MIS size for a graph given by dot_matrix.
    Conflict if dot > 2.
    """
    if n == 0:
        return 0
    
    conflicts = dot_matrix > 2
    np.fill_diagonal(conflicts, False)
    
    alive = np.ones(n, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    count = 0
    
    # Use a simple priority queue logic or just loop
    # For small n, simple loop is fine
    remaining_indices = np.arange(n)
    
    while len(remaining_indices) > 0:
        if time.time() > 145: # Hard safety
            break
        # Find min degree among alive
        # degrees is updated for all, but we only care about alive
        # Actually, let's just recompute or maintain.
        # For simplicity and speed in small graphs, just pick from remaining
        min_deg = np.min(degrees[remaining_indices])
        cands = remaining_indices[degrees[remaining_indices] == min_deg]
        chosen = cands[0] # Deterministic for reproducibility in selection phase
        
        count += 1
        
        # Remove chosen and neighbors
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update remaining_indices
        remaining_indices = np.setdiff1d(remaining_indices, to_remove)
        
        # Update degrees for remaining
        # degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        if len(remaining_indices) > 0:
             degrees[remaining_indices] -= conflicts[remaining_indices][:, to_remove].sum(axis=1)
             
    return count

def _generate_block_vectors(block_coords):
    """
    Generate optimal vectors for a specific set of coordinates.
    Returns list of vectors (as full D-dim arrays).
    """
    k = len(block_coords)
    vectors = []
    
    # 1. Weight-1 magnitude 2 vectors
    for pos in block_coords:
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = 2 * sign
            vectors.append(v)
            
    # 2. Weight-4 vectors
    if k >= 4:
        # Generate all weight-4 vectors in this block
        local_candidates = []
        local_indices = []
        for positions in itertools.combinations(range(k), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(k, dtype=np.int64)
                for pos, s in zip(positions, signs):
                    v[pos] = s
                local_candidates.append(v)
                
        if local_candidates:
            local_candidates = np.array(local_candidates, dtype=np.int64)
            N = len(local_candidates)
            dots = local_candidates @ local_candidates.T
            
            # Run greedy MIS to get selected indices
            # We need the actual indices, not just the count.
            # Let's reuse the logic but return indices.
            
            conflicts = dots > 2
            np.fill_diagonal(conflicts, False)
            
            alive = np.ones(N, dtype=bool)
            degrees = conflicts.sum(axis=1).astype(np.int32)
            selected = []
            remaining_indices = np.arange(N)
            
            while len(remaining_indices) > 0:
                if time.time() > 145:
                    break
                min_deg = np.min(degrees[remaining_indices])
                cands = remaining_indices[degrees[remaining_indices] == min_deg]
                chosen = cands[0]
                selected.append(chosen)
                
                neighbors = np.where(conflicts[chosen])[0]
                to_remove = np.unique(np.concatenate(([chosen], neighbors)))
                remaining_indices = np.setdiff1d(remaining_indices, to_remove)
                
                if len(remaining_indices) > 0:
                    degrees[remaining_indices] -= conflicts[remaining_indices][:, to_remove].sum(axis=1)
                    
            # Map local indices to global vectors
            for idx in selected:
                v_local = local_candidates[idx]
                v_global = np.zeros(D, dtype=np.int64)
                for i, pos in enumerate(block_coords):
                    v_global[pos] = v_local[i]
                vectors.append(v_global)
                
    return vectors

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # List of partitions of 11
    # We can generate them or hardcode the likely best ones
    # Let's hardcode a few good candidates to save time
    candidates = [
        [11],
        [7, 4],
        [6, 5],
        [5, 4, 2],
        [4, 4, 3],
        [8, 3],
        [9, 2],
        [10, 1],
        [5, 3, 3],
        [4, 3, 4], # same as 4,4,3
        [6, 4, 1],
        [7, 2, 2],
    ]
    
    best_score = -1
    best_partition = None
    best_vectors = []
    
    # Estimate scores for each partition
    # We need to compute MIS size for each block size.
    # Since block sizes repeat, we can cache.
    mis_cache = {}
    
    def get_mis_size(k):
        if k in mis_cache:
            return mis_cache[k]
        if k < 4:
            mis_cache[k] = 0
            return 0
            
        # Generate local candidates for estimation
        local_candidates = []
        for positions in itertools.combinations(range(k), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(k, dtype=np.int64)
                for pos, s in zip(positions, signs):
                    v[pos] = s
                local_candidates.append(v)
                
        local_candidates = np.array(local_candidates, dtype=np.int64)
        dots = local_candidates @ local_candidates.T
        size = _greedy_mis_size(len(local_candidates), dots)
        mis_cache[k] = size
        return size
        
    for part in candidates:
        if time.time() > deadline - 10:
            break
        score = 0
        valid = True
        if sum(part) != 11:
            continue
        for k in part:
            # f(k) = 2k + MIS(k)
            score += 2 * k + get_mis_size(k)
            
        if score > best_score:
            best_score = score
            best_partition = part
            
    if best_partition is None:
        best_partition = [11]
        
    # Generate vectors for the best partition
    coords = list(range(D))
    current_idx = 0
    for k in best_partition:
        if time.time() > deadline - 5:
            break
        block_coords = coords[current_idx : current_idx + k]
        v_block = _generate_block_vectors(block_coords)
        best_vectors.extend(v_block)
        current_idx += k
        
    if not best_vectors:
        # Fallback to simple weight-1
        for pos in range(D):
            for sign in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[pos] = 2 * sign
                best_vectors.append(v)
                
    return np.array(best_vectors, dtype=np.int64)
