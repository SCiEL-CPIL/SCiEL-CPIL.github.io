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
            
    # Precompute valid_1: non-empty masks that are clique or independent
    valid_1 = [False] * (1 << n)
    valid_1[0] = False  # Empty set is not a valid part
    for mask in range(1, 1 << n):
        valid_1[mask] = is_clique[mask] or is_indep[mask]
        
    # Check k=1
    if valid_1[full_mask]:
        return [list(range(n))]
        
    # Check k=2
    # Find a submask T of full_mask such that valid_1[T] and valid_1[full_mask ^ T]
    sub = full_mask
    found_k2 = False
    witness_k2 = -1
    while sub:
        if valid_1[sub] and valid_1[full_mask ^ sub]:
            witness_k2 = sub
            found_k2 = True
            break
        sub = (sub - 1) & full_mask
        
    if found_k2:
        T = witness_k2
        S = full_mask ^ T
        part1 = [j for j in range(n) if (T >> j) & 1]
        part2 = [j for j in range(n) if (S >> j) & 1]
        return [part1, part2]
        
    # Check k=3
    # Find T, U, W such that T ∪ U ∪ W = full_mask, all non-empty, all valid_1
    # Equivalently: find T subset full_mask with valid_1[T], and U subset (full_mask ^ T) with valid_1[U] and valid_1[full_mask ^ T ^ U]
    
    # Iterate over all T that are valid_1
    for T in range(1, full_mask + 1):
        if not valid_1[T]:
            continue
        if time.time() > deadline:
            break
        rest = full_mask ^ T
        # Iterate over U subset rest
        sub = rest
        while sub:
            if valid_1[sub] and valid_1[rest ^ sub]:
                U = sub
                W = rest ^ sub
                part1 = [j for j in range(n) if (T >> j) & 1]
                part2 = [j for j in range(n) if (U >> j) & 1]
                part3 = [j for j in range(n) if (W >> j) & 1]
                return [part1, part2, part3]
            sub = (sub - 1) & rest
            
    # Check k=4
    # Find T, U, V, W such that T ∪ U ∪ V ∪ W = full_mask, all non-empty, all valid_1
    # Equivalently: find T subset full_mask with valid_1[T], then find U, V, W partitioning rest into 3 valid parts
    
    for T in range(1, full_mask + 1):
        if not valid_1[T]:
            continue
        if time.time() > deadline:
            break
        rest = full_mask ^ T
        # Find U, V, W partitioning rest into 3 valid parts
        sub = rest
        while sub:
            if not valid_1[sub]:
                sub = (sub - 1) & rest
                continue
            U = sub
            rest2 = rest ^ U
            # Find V, W partitioning rest2 into 2 valid parts
            sub2 = rest2
            while sub2:
                if valid_1[sub2] and valid_1[rest2 ^ sub2]:
                    V = sub2
                    W = rest2 ^ V
                    part1 = [j for j in range(n) if (T >> j) & 1]
                    part2 = [j for j in range(n) if (U >> j) & 1]
                    part3 = [j for j in range(n) if (V >> j) & 1]
                    part4 = [j for j in range(n) if (W >> j) & 1]
                    return [part1, part2, part3, part4]
                sub2 = (sub2 - 1) & rest2
            sub = (sub - 1) & rest
            
    # Fallback: greedy partition
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts
