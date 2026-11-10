import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_indep for all masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest
        # If rest is clique, and v is connected to all in rest, then mask is clique
        if is_clique[rest]:
            # Check if v is connected to all in rest
            is_clique[mask] = True
            # Iterate bits in rest
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if not adj[v][u]:
                    is_clique[mask] = False
                    break
                r ^= lsb_r
        else:
            is_clique[mask] = False
            
        # Check independent set: v must NOT be connected to any in rest
        # If rest is independent, and v is not connected to any in rest, then mask is independent
        if is_indep[rest]:
            # Check if v is not connected to any in rest
            is_indep[mask] = True
            # Iterate bits in rest
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if adj[v][u]:
                    is_indep[mask] = False
                    break
                r ^= lsb_r
        else:
            is_indep[mask] = False

    if time.time() > deadline:
        # Fallback to greedy if precomputation is too slow (unlikely)
        return greedy_partition()

    # Precompute cochromatic_2 and witness_2
    cochromatic_2 = [False] * (1 << n)
    witness_2 = [-1] * (1 << n)
    
    cochromatic_2[0] = True
    witness_2[0] = 0 # Empty partition of empty set
    
    for mask in range(1, 1 << n):
        # Iterate over submasks s of mask
        s = mask
        while s:
            if (is_clique[s] and is_indep[mask ^ s]) or (is_indep[s] and is_clique[mask ^ s]):
                cochromatic_2[mask] = True
                witness_2[mask] = s
                break
            s = (s - 1) & mask
            if time.time() > deadline:
                # Fallback
                return greedy_partition()
                
    # Search for k=1
    if is_clique[full_mask] or is_indep[full_mask]:
        return [list(range(n))]
        
    # Search for k=2
    s = full_mask
    while s:
        if (is_clique[s] and is_indep[full_mask ^ s]) or (is_indep[s] and is_clique[full_mask ^ s]):
            part1 = [i for i in range(n) if (s >> i) & 1]
            part2 = [i for i in range(n) if (full_mask ^ s) >> i & 1]
            return [part1, part2]
        s = (s - 1) & full_mask
        if time.time() > deadline:
            return greedy_partition()
            
    # Search for k=3
    s1 = full_mask
    while s1:
        if (is_clique[s1] or is_indep[s1]) and cochromatic_2[full_mask ^ s1]:
            rem = full_mask ^ s1
            s2 = witness_2[rem]
            s3 = rem ^ s2
            part1 = [i for i in range(n) if (s1 >> i) & 1]
            part2 = [i for i in range(n) if (s2 >> i) & 1]
            part3 = [i for i in range(n) if (s3 >> i) & 1]
            return [part1, part2, part3]
        s1 = (s1 - 1) & full_mask
        if time.time() > deadline:
            return greedy_partition()
            
    # If k=3 not found, try k=4 or fallback
    # For now, fallback to greedy
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
