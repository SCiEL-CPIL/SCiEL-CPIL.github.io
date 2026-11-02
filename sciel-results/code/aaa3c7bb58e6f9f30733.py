"""Greedy + Local Search (1-for-2 Swap) for kissing number in dimension 11."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    D = 11

    # Generate candidate vectors: weight-4 (norm^2=4)
    # We focus on weight 4 as they provide a good balance of density and distance.
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
            
    # Add weight 2 vectors? 
    # Norm^2 = 2. Max norm in set would be 4 (if mixed) or 2 (if pure).
    # If we mix, max norm is 2 (sqrt(4)=2? No, norm is sqrt(sum squares)).
    # Weight 4 vector: norm = 2.
    # Weight 2 vector: norm = sqrt(2) ~ 1.41.
    # If max norm is 2, min dist must be >= 2.
    # dist^2 >= 4.
    # v_i . v_j <= (||v_i||^2 + ||v_j||^2 - 4)/2
    # If both weight 4: v_i . v_j <= (4+4-4)/2 = 2.
    # If w4 and w2: v_i . v_j <= (4+2-4)/2 = 1.
    # If both w2: v_i . v_j <= (2+2-4)/2 = 0.
    # This mixed approach is complex. Let's stick to weight 4 only for simplicity and speed, 
    # or weight 2 only?
    # Kissing number in 11 is likely achieved by a mix or specific lattice.
    # However, weight 4 vectors are more "spread out".
    # Let's try Weight 4 only first. If it's too small, we can add Weight 2.
    # Actually, let's include Weight 2 as well, but handle the threshold correctly.
    
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => 2*v_i.v_j <= ||v_i||^2 + ||v_j||^2 - max_norm_sq
    # => v_i.v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    # We use integer arithmetic to avoid float precision issues if possible, 
    # but float is fine for comparison if careful.
    # Let's use: 2*dot <= norms_sq[i] + norms_sq[j] - max_norm_sq
    # conflict if 2*dot > norms_sq[i] + norms_sq[j] - max_norm_sq
    
    # Vectorized calculation of the condition for conflict
    # conflict[i,j] is True if 2*dot[i,j] > norms_sq[i] + norms_sq[j] - max_norm_sq
    rhs = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq
    conflict = (2 * dot > rhs).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Degree of each vertex in the conflict graph
    degree = conflict.sum(axis=1).astype(np.int32)

    # Greedy Initialization
    # We want to select a large independent set.
    # Strategy: Pick vertex with min degree, remove it and its neighbors.
    
    remaining = np.ones(n, dtype=bool)
    selected = []
    
    # Pre-compute neighbors for faster removal? 
    # conflict[i] is a boolean mask of neighbors.
    
    while time.time() < deadline - 10.0:
        if not remaining.any():
            break
            
        # Compute degree within remaining subgraph
        # deg[i] = sum(conflict[i, remaining])
        # This is O(N^2) per step. N=5500. 5500^2 = 30M ops. 
        # Doing this 100 times is 3B ops. Might be slow in Python loop.
        # Optimization: Maintain a count of remaining neighbors.
        
        # Let's use a faster heuristic: 
        # Just pick a random vertex from remaining with low global degree?
        # Or precompute degrees and update?
        
        # For now, let's do a simple greedy pass with a time limit.
        # To speed up, we can sample a subset of remaining vertices to find min degree.
        
        # Actually, N=5500 is small enough that `conflict[remaining, :].sum(axis=0)` is fast in C.
        # Let's try it.
        
        # Get degrees of remaining vertices
        # conflict is uint8. 
        # deg_remaining = conflict[remaining].sum(axis=1)
        
        # We want min degree.
        # idx = np.argmin(deg_remaining)
        
        # But we need to map back to global index.
        rem_indices = np.where(remaining)[0]
        if len(rem_indices) == 0:
            break
            
        # Calculate degrees for remaining
        # This creates a view, so it's fast.
        degs = conflict[rem_indices].sum(axis=1)
        
        min_deg_idx_local = np.argmin(degs)
        min_deg_val = degs[min_deg_idx_local]
        
        # Tie-breaking: pick first one with min degree
        # Or pick random among min degree to diversify?
        # Let's pick the one with min degree.
        
        new_idx = rem_indices[min_deg_idx_local]
        selected.append(new_idx)
        
        # Remove new_idx and its neighbors from remaining
        remaining[new_idx] = False
        neighbors = conflict[new_idx]
        remaining[neighbors.astype(bool)] = False
        
        # If we have selected many, the remaining set shrinks.
        # If time is running out, break.
        
    # Convert to boolean array
    x = np.zeros(n, dtype=np.uint8)
    for s in selected:
        x[s] = 1
        
    # Local Search: 1-for-2 Swap
    # Try to replace one selected vertex with two unselected ones that are compatible.
    
    # Precompute compatibility with current set?
    # A vertex v is compatible with set S if conflict[v, S].sum() == 0.
    
    sel_arr = np.array(selected)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Iterate through selected vertices
        # To make it faster, we can shuffle the order
        order = np.random.permutation(len(sel_arr))
        
        for i in order:
            if time.time() > deadline - 5.0:
                break
                
            v1 = sel_arr[i]
            
            # Remove v1 from selection temporarily
            # The set is S \ {v1}
            # We need to find two vertices u, w not in S such that:
            # 1. u, w compatible with S \ {v1}
            # 2. u, w compatible with each other
            
            # Candidates: vertices not in S (x==0)
            # And compatible with S \ {v1}
            
            # Let's compute conflicts with S \ {v1}
            # S_no_v1 = sel_arr with v1 removed
            
            # Efficient way:
            # conflict_with_S = conflict[:, sel_arr].sum(axis=1)
            # But we need to exclude v1.
            # conflict_with_S_no_v1 = conflict_with_S - conflict[:, v1]
            
            # Compute conflict_with_S for all vertices
            # This is O(N * |S|). |S| ~ 100-200. N=5500. 1M ops. Fast.
            
            c_with_S = conflict[:, sel_arr].sum(axis=1).astype(np.int32)
            c_with_v1 = conflict[:, v1].astype(np.int32)
            c_with_S_no_v1 = c_with_S - c_with_v1
            
            # Candidates are those with x==0 and c_with_S_no_v1 == 0
            # Note: v1 itself has x==1, so it's excluded by x==0.
            # But neighbors of v1 in S have c_with_S_no_v1 >= 0.
            # We need c_with_S_no_v1 == 0.
            
            cand_mask = (x == 0) & (c_with_S_no_v1 == 0)
            cand_indices = np.where(cand_mask)[0]
            
            if len(cand_indices) < 2:
                continue
                
            # Now we need to find a pair (u, w) from cand_indices such that conflict[u, w] == 0
            # This is finding an edge in the "compatibility graph" of candidates.
            # The compatibility graph is the complement of the conflict graph restricted to candidates.
            
            # We can check pairs. If len(cand) is small, it's fast.
            # If len(cand) is large, we can just pick the first two that are compatible.
            
            found_swap = False
            # To avoid O(K^2) if K is large, we can just check a few random pairs or the first few.
            # Let's check the first few candidates against each other.
            
            # Optimization: If cand_indices is large, the probability of finding a compatible pair is high.
            # We can just check cand_indices[0] against cand_indices[1:]
            
            u = cand_indices[0]
            # Check if u is compatible with any other candidate
            # conflict[u, cand_indices[1:]] == 0
            compat_with_u = conflict[u, cand_indices[1:]] == 0
            if compat_with_u.any():
                w = cand_indices[1:][np.argmax(compat_with_u)]
                
                # Perform swap
                # Remove v1, add u, w
                x[v1] = 0
                x[u] = 1
                x[w] = 1
                
                # Update sel_arr
                # Remove v1, add u, w
                sel_arr = np.delete(sel_arr, i)
                sel_arr = np.append(sel_arr, [u, w])
                
                improved = True
                found_swap = True
                break # Restart loop with new set
                
            # If no swap found with u, maybe try another u?
            # For speed, we only try the first candidate u.
            
    # Final Greedy Fill
    # Add any remaining compatible vertices
    sel_arr = np.where(x == 1)[0]
    
    if time.time() < deadline - 2.0:
        while True:
            if time.time() > deadline - 2.0:
                break
            c_with_S = conflict[:, sel_arr].sum(axis=1).astype(np.int32)
            addable = np.where((x == 0) & (c_with_S == 0))[0]
            if len(addable) == 0:
                break
            # Pick one with min degree
            degs = degree[addable]
            min_idx = addable[np.argmin(degs)]
            x[min_idx] = 1
            sel_arr = np.append(sel_arr, min_idx)

    # Extract result
    sel_indices = np.where(x == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)
