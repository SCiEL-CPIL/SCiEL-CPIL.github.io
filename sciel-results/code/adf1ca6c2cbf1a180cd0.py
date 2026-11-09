import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115
    
    full_mask = (1 << n) - 1
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(masks, u, v):
        # Check if adding edge (u, v) creates a K3,3
        # u, v are not connected yet
        
        # Case 1: u, v on same side of K3,3
        # Need w such that |N(u) ∩ N(v) ∩ N(w)| >= 3
        common_uv = masks[u] & masks[v]
        if common_uv:
            for w in range(n):
                if w != u and w != v:
                    if (common_uv & masks[w]).bit_count() >= 3:
                        return False
        
        # Case 2: u, v on opposite sides of K3,3
        # u in A, v in B.
        # A = {u, a2, a3}, B = {v, b2, b3}
        # a2, a3 in N(v) \ N(u)
        # b2, b3 in N(u) \ N(v)
        # K2,2 between {a2, a3} and {b2, b3}
        
        # A_cands: neighbors of v that are NOT neighbors of u
        A_cands = masks[v] & (full_mask ^ masks[u])
        # B_cands: neighbors of u that are NOT neighbors of v
        B_cands = masks[u] & (full_mask ^ masks[v])
        
        if A_cands and B_cands:
            # Get list of vertices in A_cands
            a_list = []
            temp = A_cands
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                a_list.append(idx)
                temp ^= lsb # temp -= lsb also works
                
            # Check all pairs in a_list
            for i in range(len(a_list)):
                for j in range(i+1, len(a_list)):
                    a2 = a_list[i]
                    a3 = a_list[j]
                    # Common neighbors of a2 and a3 that are in B_cands
                    common_ab = masks[a2] & masks[a3] & B_cands
                    if common_ab.bit_count() >= 2:
                        return False
        
        return True

    best_count = 0
    best_masks = None
    
    # Run multiple trials with different random orderings
    trials = 1000
    for t in range(trials):
        if time.time() > deadline:
            break
            
        masks = [0] * n
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add(masks, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        
        count = sum(m.bit_count() for m in masks) // 2
        if count > best_count:
            best_count = count
            best_masks = masks[:]
            
    # Convert best_masks to edge list
    if best_masks is None:
        # Fallback
        return []
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_masks[i] & (1 << j):
                edges.append([i, j])
                
    return edges
