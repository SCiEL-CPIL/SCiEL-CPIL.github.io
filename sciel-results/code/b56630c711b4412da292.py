import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precomputation
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Convert to numpy array for faster access
    # all_subsets_np[i] is the i-th t-subset
    all_subsets_np = np.array(all_t_subsets, dtype=np.int8)
    
    # Precompute for each point x, the indices of t-subsets containing x
    # and the other t-1 elements of those subsets
    subset_indices_by_point = {}
    other_elements_by_point = {}
    
    for x in all_points:
        idxs = []
        others = []
        for i, s in enumerate(all_t_subsets):
            if x in s:
                idxs.append(i)
                others.append(tuple(y for y in s if y != x))
        subset_indices_by_point[x] = np.array(idxs, dtype=np.int32)
        other_elements_by_point[x] = np.array(others, dtype=np.int8)
        
    # Covered array
    covered = np.zeros(n_t_subsets, dtype=bool)
    blocks = []
    
    # Greedy Construction
    while not covered.all():
        if time.time() > deadline:
            break
            
        # Find first uncovered subset
        # This can be slow if done naively. 
        # We can maintain a list of uncovered indices or just find the first False.
        # Since we update covered, we can just use np.where(~covered)[0][0]
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0:
            break
        first_uncovered_idx = uncovered_idx[0]
        S = all_t_subsets[first_uncovered_idx]
        
        B = list(S)
        
        # Build the block
        while len(B) < k:
            if time.time() > deadline:
                # If we run out of time, we might return a partial block or just stop
                # But we need a valid block. Let's just pick random points to complete it.
                # Actually, better to just break and hope we have enough blocks.
                # But if we break here, B is not a full block.
                # Let's complete it with arbitrary points.
                for x in all_points:
                    if x not in B:
                        B.append(x)
                    if len(B) == k:
                        break
                break
            
            best_x = -1
            max_gain = -1
            B_set = set(B)
            B_mask = np.zeros(v, dtype=bool)
            for b in B:
                B_mask[b] = True
                
            for x in all_points:
                if x in B_set:
                    continue
                
                idxs = subset_indices_by_point[x]
                others = other_elements_by_point[x]
                
                # Check if others are in B
                # others shape: (N, t-1)
                # B_mask[others] shape: (N, t-1)
                in_B = B_mask[others]
                valid = in_B.all(axis=1)
                
                # Count uncovered valid subsets
                gain = (~covered[idxs] & valid).sum()
                
                if gain > max_gain:
                    max_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Should not happen if k <= v
                break
            B.append(best_x)
        
        # Sort B for consistency
        B.sort()
        blocks.append(B)
        
        # Update covered array
        # For each t-subset in B, mark as covered
        for s in itertools.combinations(B, t):
            idx = subset_to_idx[s]
            covered[idx] = True
            
    # Local Search: Remove redundant blocks
    # This part is tricky to vectorize easily without a coverage count matrix.
    # Given time constraints, we might skip complex local search if greedy is fast enough.
    # But let's try a simple redundancy check.
    
    # We need to know for each t-subset, how many blocks cover it.
    # coverage_count[i] = number of blocks containing t-subset i
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    
    for B in blocks:
        for s in itertools.combinations(B, t):
            idx = subset_to_idx[s]
            coverage_count[idx] += 1
            
    # A block is redundant if for all t-subsets s in B, coverage_count[s] > 1
    # We can try to remove blocks one by one.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            B = blocks[i]
            redundant = True
            for s in itertools.combinations(B, t):
                idx = subset_to_idx[s]
                if coverage_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                # Update coverage_count
                for s in itertools.combinations(B, t):
                    idx = subset_to_idx[s]
                    coverage_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break # Restart loop as indices changed
    
    return blocks
