import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    total_masks = 1 << n
    
    # Precompute popcount for all masks
    # This allows O(1) degree calculation: deg(v, M) = popcount[adj[v] & M]
    popcount = [0] * total_masks
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    best_count = -1
    best_mask = 0
    
    # Iterate through all possible subsets
    # We can iterate by size to potentially find good solutions early, 
    # but given the small N, full iteration is fine.
    # To be slightly more efficient, we can skip masks with popcount < minv.
    
    for mask in range(total_masks):
        if popcount[mask] < minv:
            continue
            
        # Calculate degrees for vertices in the mask
        degs = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Degree of v in induced subgraph is number of neighbors of v also in mask
            d = popcount[adj[v] & mask]
            degs.add(d)
            m ^= lsb
            
        c = len(degs)
        if c > best_count:
            best_count = c
            best_mask = mask
            
            # Theoretical max distinct degrees is the size of the subset
            # If we achieve this for the largest possible size, we can't do better?
            # No, a smaller subset can't have more distinct degrees than its size.
            # If best_count == n (17), we can break immediately.
            if best_count == n:
                break
                
    # Extract vertices from best_mask
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb
        
    return best_vertices
