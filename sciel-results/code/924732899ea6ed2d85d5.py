import time
import random

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin
    
    def generate_paley(n):
        squares = set((x*x) % n for x in range(n))
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                colors.append(1 if d in squares else 0)
        return colors

    paley_29_colors = generate_paley(29)
    
    # Build adjacency bitmasks for Paley(29)
    n_base = 29
    adj1_base = [0] * n_base
    idx = 0
    for i in range(n_base):
        for j in range(i+1, n_base):
            if paley_29_colors[idx] == 1:
                adj1_base[i] |= (1 << j)
                adj1_base[j] |= (1 << i)
            idx += 1

    # Fast checker for K2,2,2 in a graph defined by adjacency bitmasks
    def has_k222(adj, n):
        # adj[i] is a bitmask of neighbors of i
        # We look for disjoint pairs (u,v), (x,y), (z,w) 
        # such that u,v non-edge, x,y non-edge, z,w non-edge
        # and all cross edges are present.
        
        # Optimization: 
        # Iterate u < v where (u,v) is a non-edge.
        # S1 = N(u) & N(v)
        # Iterate x < y in S1 where (x,y) is a non-edge.
        # S2 = N(x) & N(y) & S1
        # Check if S2 contains a non-edge (z,w).
        
        for u in range(n):
            # Find non-neighbors of u
            # Mask of all vertices
            all_mask = (1 << n) - 1
            # Non-neighbors of u, excluding u itself
            non_neigh_u = all_mask & ~(adj[u] | (1 << u))
            
            # Iterate v in non_neigh_u where v > u to avoid duplicates? 
            # Actually, the pair (u,v) is unordered. 
            # We can just iterate v from u+1 to n-1.
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                
                # (u, v) is a non-edge
                S1 = adj[u] & adj[v]
                
                # Get list of vertices in S1
                # To iterate pairs (x, y) in S1, we need them to be non-edges.
                # We can iterate x in S1, then y in S1 where y > x.
                # But we need to check if (x,y) is a non-edge.
                
                # Efficient iteration over bits in S1
                # We can create a list of vertices in S1
                # But doing this inside the loop is slow.
                # Let's try to iterate x, y directly from S1.
                
                # Extract indices
                s_list = []
                temp = S1
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    s_list.append(idx_bit)
                    temp ^= lsb
                
                m = len(s_list)
                if m < 4:
                    continue
                
                for i in range(m):
                    x = s_list[i]
                    for j in range(i + 1, m):
                        y = s_list[j]
                        
                        # Check if (x, y) is a non-edge
                        if (adj[x] >> y) & 1:
                            continue
                        
                        # (x, y) is a non-edge in G, and x, y are in N(u) & N(v).
                        # Now we need a non-edge (z, w) in N(x) & N(y) & S1.
                        S2 = adj[x] & adj[y] & S1
                        
                        # Check if S2 contains a non-edge
                        # Iterate z in S2, w in S2, z < w
                        # Check if (z, w) is a non-edge.
                        
                        # To be faster, we can check if S2 is a clique.
                        # If S2 is NOT a clique, it contains a non-edge.
                        
                        # Check if S2 is a clique:
                        # For each z in S2, adj[z] & S2 should contain all other vertices in S2.
                        # i.e., adj[z] & S2 == S2 ^ (1 << z) ? No, S2 doesn't contain z.
                        # adj[z] & S2 should be S2.
                        
                        # Wait, if S2 is a clique, then for all z in S2, 
                        # every w in S2 \ {z} is a neighbor of z.
                        # So adj[z] & S2 should equal S2.
                        
                        # Let's verify this logic.
                        # If S2 = {1, 2, 3} and it's a clique.
                        # adj[1] has bits 1, 2, 3 (and others).
                        # adj[1] & S2 = {1, 2, 3} & {1, 2, 3} = {1, 2, 3}.
                        # Correct.
                        
                        # If it's not a clique, say (1,2) is non-edge.
                        # adj[1] & S2 will not have bit 2.
                        # So adj[1] & S2 != S2.
                        
                        # This check is O(|S2|) bit operations.
                        
                        temp_s2 = S2
                        is_clique = True
                        while temp_s2:
                            lsb = temp_s2 & -temp_s2
                            z = lsb.bit_length() - 1
                            if (adj[z] & S2) != S2:
                                is_clique = False
                                break
                            temp_s2 ^= lsb
                        
                        if not is_clique:
                            return True
        return False

    def is_valid_30(mask):
        # mask is 29 bits, bit i is color of edge (29, i)
        # Build adj1 and adj0 for n=30
        
        adj1 = [0] * 30
        adj0 = [0] * 30
        
        # Copy base adjacencies
        for i in range(29):
            adj1[i] = adj1_base[i]
            # adj0 is complement of adj1 within 0..28
            # But we need to be careful with the new vertex 29.
            
        # Add edges for vertex 29
        for i in range(29):
            if (mask >> i) & 1:
                adj1[29] |= (1 << i)
                adj1[i] |= (1 << 29)
            else:
                adj0[29] |= (1 << i)
                adj0[i] |= (1 << 29)
                
        # Complete adj0 for base vertices
        # adj0[i] should have bits for j where color is 0.
        # We already set bits for j=29.
        # We need bits for j in 0..28.
        # adj0[i] = ~(adj1[i]) & ((1<<29)-1) for the base part?
        # Let's just build adj0 from scratch for safety and speed (it's small).
        
        adj0 = [0] * 30
        # Base part
        for i in range(29):
            for j in range(i+1, 29):
                if (adj1_base[i] >> j) & 1:
                    # Color 1
                    pass
                else:
                    # Color 0
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
        # New vertex edges
        for i in range(29):
            if (mask >> i) & 1:
                pass # Color 1
            else:
                adj0[29] |= (1 << i)
                adj0[i] |= (1 << 29)
        
        if has_k222(adj1, 30):
            return False
        if has_k222(adj0, 30):
            return False
        return True

    # Try to find a 30-vertex coloring
    # Strategy: 
    # 1. Try specific structured masks (e.g., neighborhood of a vertex, non-neighborhood).
    # 2. Random restarts.
    
    # Precompute some candidate masks
    candidates = []
    
    # All 0
    candidates.append(0)
    # All 1
    candidates.append((1 << 29) - 1)
    
    # Neighborhoods of vertices in Paley(29)
    for i in range(29):
        candidates.append(adj1_base[i])
        # Non-neighborhood (excluding self)
        non_neigh = ((1 << 29) - 1) & ~(adj1_base[i] | (1 << i))
        candidates.append(non_neigh)
        
    # Random masks
    random.seed(42)
    for _ in range(1000):
        candidates.append(random.getrandbits(29))
        
    # Deduplicate
    candidates = list(set(candidates))
    
    found_30 = False
    valid_mask = None
    
    for mask in candidates:
        if time.time() > deadline:
            break
        if is_valid_30(mask):
            found_30 = True
            valid_mask = mask
            break
            
    if found_30:
        # Construct the full color list for n=30
        colors = []
        for i in range(30):
            for j in range(i + 1, 30):
                if i == 29:
                    colors.append((valid_mask >> j) & 1)
                elif j == 29:
                    colors.append((valid_mask >> i) & 1)
                else:
                    # Get from base
                    # Index in base: sum_{k=0}^{i-1} (28-k) + (j-i-1)
                    # Simpler: just use the generated paley_29_colors and map indices
                    # But we already have adj1_base.
                    if (adj1_base[i] >> j) & 1:
                        colors.append(1)
                    else:
                        colors.append(0)
        return {'n': 30, 'colors': colors}
        
    # If 30 not found, return 29
    return {'n': 29, 'colors': paley_29_colors}
