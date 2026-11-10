import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_indep for all masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    # We iterate masks in increasing order of popcount or just integer order?
    # Integer order works if we use the recurrence:
    # lsb = mask & -mask
    # v = lsb.bit_length() - 1
    # rest = mask ^ lsb
    # is_clique[mask] = is_clique[rest] and (adj_mask[v] & rest) == rest
    # is_indep[mask] = is_indep[rest] and (adj_mask[v] & rest) == 0
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest]:
            if (adj_mask[v] & rest) == rest:
                is_clique[mask] = True
            else:
                is_clique[mask] = False
        else:
            is_clique[mask] = False
            
        if is_indep[rest]:
            if (adj_mask[v] & rest) == 0:
                is_indep[mask] = True
            else:
                is_indep[mask] = False
        else:
            is_indep[mask] = False
            
    # Precompute valid_1: masks that are clique or independent
    valid_1 = [False] * (1 << n)
    for mask in range(1 << n):
        valid_1[mask] = is_clique[mask] or is_indep[mask]
        
    # DP for k parts
    # valid_k[S] = True if S can be partitioned into k non-empty valid parts
    # We store witnesses to reconstruct the partition
    
    # valid_0 is only True for mask 0
    # valid_1 is valid_1
    
    # We want the smallest k such that valid_k[full_mask] is True
    
    # To save memory and time, we only keep the previous layer
    # But we need to reconstruct the partition for the final k.
    # We can store witnesses for the current best k found.
    # Since k is small (likely 2 or 3), we can just store the witness for the specific path.
    
    # Let's do a BFS/DP layer by layer.
    # prev_valid is a list of bools for k-1
    # curr_valid is a list of bools for k
    # witness is a 2D list or we just store the witness for the full mask at the final step?
    # Actually, we need the full partition.
    # Let's store `witness[S][k]`? That's $2^N \times N$ which is fine.
    # Or simpler: just find the k, then backtrack.
    
    # Optimization: If k=1, check full_mask.
    if valid_1[full_mask]:
        return [list(range(n))]
        
    # Check k=2
    # We need to find a submask T of full_mask such that valid_1[T] and valid_1[full_mask ^ T]
    # To reconstruct, we just need T.
    
    # General approach:
    # prev_valid = valid_1 (for k=1)
    # for k in range(2, n+1):
    #   curr_valid = [False] * (1<<n)
    #   witness_curr = [-1] * (1<<n)
    #   for S in range(1<<n):
    #      if not prev_valid[S]: continue? No, we need to build S from subsets.
    #      # Iterate subsets T of S
    #      # If valid_1[T] and prev_valid[S^T], then curr_valid[S] = True, witness_curr[S] = T
    #      # We can break early if we find one.
    #      # To optimize, iterate T as submasks of S.
    #      sub = S
    #      while sub:
    #         if valid_1[sub] and prev_valid[S ^ sub]:
    #             curr_valid[S] = True
    #             witness_curr[S] = sub
    #             break
    #         sub = (sub - 1) & S
    #             
    #   if curr_valid[full_mask]:
    #       # Reconstruct
    #       parts = []
    #       S = full_mask
    #       # We need witnesses for all layers.
    #       # So we should store witness for all k?
    #       # Or we can just store the path for full_mask backwards?
    #       # The witness for full_mask at layer k is T_k.
    #       # Then we need witness for full_mask ^ T_k at layer k-1.
    #       # So we need the witness arrays for all layers up to k.
    #       # Since k is small, we can store a list of witness arrays.
    #       
    #       # Let's restructure:
    #       # Store `all_witnesses` as a list of lists.
    #       # all_witnesses[k-1][S] = T such that S is partitioned into k parts, and T is the k-th part.
    #       
    #   prev_valid = curr_valid

    # Let's implement this.
    
    prev_valid = valid_1
    witnesses = [valid_1] # witnesses[0] is not really a witness, just the base. 
    # Actually, for k=1, the part is the mask itself.
    
    for k in range(2, n + 1):
        curr_valid = [False] * (1 << n)
        curr_witness = [-1] * (1 << n)
        
        # Iterate all masks S
        # We only care about S that can be formed.
        # But we must check all S to build up to full_mask.
        for S in range(1, 1 << n):
            # If S is not valid in prev layer, it doesn't mean S^T can't be valid?
            # No, the recurrence is: S is valid in k if there exists T subset S such that T is valid in 1 and S^T is valid in k-1.
            # So we iterate submasks T of S.
            
            # Optimization: If prev_valid[S] is False, it doesn't help directly, 
            # but S^T could be valid.
            
            sub = S
            found = False
            while sub:
                if valid_1[sub] and prev_valid[S ^ sub]:
                    curr_valid[S] = True
                    curr_witness[S] = sub
                    found = True
                    break
                sub = (sub - 1) & S
                
        witnesses.append(curr_witness)
        
        if curr_valid[full_mask]:
            # Reconstruct partition
            parts = []
            S = full_mask
            # Go back from k to 1
            for i in range(k - 1, 0, -1):
                T = witnesses[i][S]
                # T is the part
                part_vertices = [j for j in range(n) if (T >> j) & 1]
                parts.append(part_vertices)
                S = S ^ T
            # Last part is S (which is valid_1)
            part_vertices = [j for j in range(n) if (S >> j) & 1]
            parts.append(part_vertices)
            
            # Reverse parts to have consistent order? Not required.
            # Ensure types are lists of ints
            return parts
            
        prev_valid = curr_valid
        
    # Fallback: should not happen for n=14
    return greedy_partition()

def greedy_partition():
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(P['n']):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
