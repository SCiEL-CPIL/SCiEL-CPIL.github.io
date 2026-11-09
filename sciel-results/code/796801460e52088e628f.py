import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Construct Paley graph for n=29
    # p=29 is a prime where p % 4 == 1.
    # Quadratic residues mod 29.
    p29 = 29
    Q29 = {x * x % p29 for x in range(1, p29)}
    
    # Build adjacency bitsets for Paley 29
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge.
    # We only need to check for K2,2,2 in one color class, then the other.
    # Actually, we check both.
    
    def build_adj(n, residues):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    adj_29_0 = build_adj(p29, Q29)
    # Complement adjacency
    adj_29_1 = [0] * p29
    for i in range(p29):
        # All bits set for vertices 0..n-1, excluding self
        mask = (1 << p29) - 1
        mask ^= (1 << i) # Remove self
        adj_29_1[i] = adj_29_0[i] ^ mask

    def has_k222_bitset(adj, n):
        """
        Check if graph with adjacency bitsets 'adj' contains K2,2,2.
        """
        # Iterate over all pairs (u, v)
        # For K2,2,2, we need two disjoint pairs (a,b) and (c,d) and (e,f)
        # such that all edges between them exist.
        # A faster way: 
        # A K2,2,2 exists iff there exist vertices a,b,c,d,e,f such that
        # a,b are connected to c,d,e,f? No.
        # Definition: Partition V'={v1..v6} into 3 pairs P1,P2,P3.
        # All edges between Pi and Pj (i!=j) exist.
        
        # Efficient check:
        # Pick a pair of vertices (u, v). 
        # Let N(u) and N(v) be their neighborhoods.
        # If we find another pair (x, y) such that x,y are in N(u) AND N(v),
        # and we find a third pair (z, w) such that z,w are in N(u) AND N(v),
        # and x,y are not connected to each other? No, they just need to be in the pair.
        # Wait, the condition is:
        # Pairs (u,v), (x,y), (z,w).
        # Edges: u-x, u-y, v-x, v-y (u,v connected to x,y)
        #         u-z, u-w, v-z, v-w (u,v connected to z,w)
        #         x-z, x-w, y-z, y-w (x,y connected to z,w)
        # Note: No edges required within pairs (u,v), (x,y), (z,w).
        # But u,v MUST be connected? No, in K2,2,2, the pairs are independent sets?
        # No, K_{2,2,2} is the complete tripartite graph K_{2,2,2}.
        # The parts are independent sets. So u and v are NOT connected.
        # x and y are NOT connected.
        # z and w are NOT connected.
        
        # So, we look for 3 disjoint pairs (u,v), (x,y), (z,w) such that:
        # 1. u,v not connected
        # 2. x,y not connected
        # 3. z,w not connected
        # 4. All cross edges exist.
        
        # Algorithm:
        # Iterate over all pairs (u, v) that are NOT connected.
        # Let C = N(u) & N(v). (Vertices connected to both u and v)
        # We need to find two disjoint pairs (x,y) and (z,w) from C
        # such that:
        # - x,y not connected
        # - z,w not connected
        # - x,z,x,w,y,z,y,w all connected.
        
        # This is still complex. Let's use a simpler sufficient condition or just brute force with bitsets.
        # With n<=31, bitsets are fast.
        
        # Precompute non-edges?
        # Let's iterate over 6-subsets? C(31,6) = 736281.
        # For each subset, check 15 partitions.
        # 736281 * 15 * (bit ops) might be okay in 100s?
        # 11 million checks. Each check is a few bit ANDs and comparisons.
        # In Python, this might be too slow.
        
        # Optimized Check:
        # Iterate u < v.
        # If adj[u] & (1<<v): continue (u,v must be non-edge)
        # C = adj[u] & adj[v]
        # Remove u, v from C? They are not in C anyway if graph is simple.
        # We need to pick 4 distinct vertices from C: x,y,z,w.
        # Such that (x,y) non-edge, (z,w) non-edge, and all cross edges.
        
        # Let's iterate x < y from C.
        # If adj[x] & (1<<y): continue (x,y must be non-edge)
        # Let D = C & ~((1<<x) | (1<<y))
        # D must contain z, w such that:
        # z,w non-edge.
        # z,w connected to x and y.
        # So z,w must be in N(x) & N(y) & D.
        # Let E = D & adj[x] & adj[y]
        # Now we just need to find a non-edge (z,w) in E.
        
        # So:
        # 1. Loop u < v where (u,v) is non-edge.
        # 2. C = N(u) & N(v)
        # 3. Loop x < y in C where (x,y) is non-edge.
        # 4. E = C & N(x) & N(y) & ~(1<<x) & ~(1<<y) & ~(1<<u) & ~(1<<v)
        #    (u,v are not in C, so no need to mask them if C is defined correctly? 
        #     C contains neighbors of u and v. u is not neighbor of u. So u not in C. Same for v.)
        # 5. Check if E contains a non-edge (z,w).
        #    i.e., exists z < w in E such that not (adj[z] & (1<<w)).
        
        # To check step 5 efficiently:
        # Iterate z in E.
        # Let F = E & (1<<z) mask? No.
        # Let F = E & ~((1<<z) | ... bits less than z? No, just check bits in E that are > z)
        # If there is any w in E, w > z, such that not (adj[z] & (1<<w)), then we found it.
        
        # Implementation:
        for u in range(n):
            # Iterate v > u
            # We need (u,v) to be a non-edge
            # Mask of vertices > u
            mask_v = ((1 << n) - 1) ^ ((1 << (u + 1)) - 1)
            # Non-neighbors of u in mask_v
            # adj[u] has 1s for neighbors.
            # We want bits in mask_v where adj[u] is 0.
            # non_adj_u_in_mask = mask_v & ~adj[u]
            non_adj_u = mask_v & ~adj[u]
            
            # Iterate v in non_adj_u
            # To iterate bits in a mask:
            m = non_adj_u
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                m ^= v_bit
                
                # u < v is guaranteed by mask_v
                # Check if (u,v) is non-edge (already ensured by non_adj_u)
                
                # C = N(u) & N(v)
                # We need C to contain x,y,z,w
                C = adj[u] & adj[v]
                # Remove u, v from C? 
                # u is not in N(u), so u not in C.
                # v is not in N(v), so v not in C.
                
                # We need to pick x,y from C such that (x,y) is non-edge.
                # And then z,w from C\{x,y} such that (z,w) is non-edge and connected to x,y.
                
                # Iterate x in C
                m_c = C
                while m_c:
                    x_bit = m_c & -m_c
                    x = x_bit.bit_length() - 1
                    m_c ^= x_bit
                    
                    # Iterate y in C, y > x
                    # Mask for y > x
                    mask_y = ((1 << n) - 1) ^ ((1 << (x + 1)) - 1)
                    # y must be in C and mask_y
                    # y must be non-neighbor of x
                    non_adj_x_in_cy = C & mask_y & ~adj[x]
                    
                    m_xy = non_adj_x_in_cy
                    while m_xy:
                        y_bit = m_xy & -m_xy
                        y = y_bit.bit_length() - 1
                        m_xy ^= y_bit
                        
                        # We have (u,v) non-edge, (x,y) non-edge.
                        # x,y in N(u) & N(v).
                        # Now find z,w in C such that:
                        # 1. z,w distinct from u,v,x,y
                        # 2. (z,w) non-edge
                        # 3. z,w in N(x) & N(y)
                        
                        # E = C & N(x) & N(y)
                        E = C & adj[x] & adj[y]
                        # Remove x, y from E
                        E &= ~(x_bit | y_bit)
                        # u,v are not in C, so no need to remove.
                        
                        # Check if E contains a non-edge (z,w)
                        # Iterate z in E
                        m_e = E
                        while m_e:
                            z_bit = m_e & -m_e
                            z = z_bit.bit_length() - 1
                            m_e ^= z_bit
                            
                            # w must be in E, w > z
                            mask_w = ((1 << n) - 1) ^ ((1 << (z + 1)) - 1)
                            # w in E & mask_w
                            # w must be non-neighbor of z
                            non_adj_z_in_ew = E & mask_w & ~adj[z]
                            
                            if non_adj_z_in_ew:
                                return True
        return False

    # Verify n=29
    valid_29 = True
    if has_k222_bitset(adj_29_0, p29) or has_k222_bitset(adj_29_1, p29):
        valid_29 = False
        
    # Prepare fallback colors for n=29
    colors_29 = []
    for i in range(p29):
        for j in range(i + 1, p29):
            if (j - i) % p29 in Q29:
                colors_29.append(0)
            else:
                colors_29.append(1)
                
    if valid_29:
        # Try to extend to n=30 or n=31?
        # We can try Paley 31.
        p31 = 31
        Q31 = {x * x % p31 for x in range(1, p31)}
        adj_31_0 = build_adj(p31, Q31)
        adj_31_1 = [0] * p31
        for i in range(p31):
            mask = (1 << p31) - 1
            mask ^= (1 << i)
            adj_31_1[i] = adj_31_0[i] ^ mask
            
        # Check if Paley 31 is valid (it shouldn't be, R=31)
        # But maybe a subset of 30 is?
        
        # Check full 31 first (fast fail)
        if not has_k222_bitset(adj_31_0, p31) and not has_k222_bitset(adj_31_1, p31):
             # This is highly unlikely
             colors_31 = []
             for i in range(p31):
                 for j in range(i + 1, p31):
                     if (j - i) % p31 in Q31:
                         colors_31.append(0)
                     else:
                         colors_31.append(1)
             return {'n': 31, 'colors': colors_31}
             
        # Try removing one vertex from Paley 31 to get 30
        if time.time() < deadline - 20:
            for v_remove in range(p31):
                if time.time() > deadline - 10:
                    break
                    
                # Build induced subgraph on 30 vertices
                # Map indices
                vertices = [i for i in range(p31) if i != v_remove]
                # new index 0..29
                # We can just filter the adjacency bitsets?
                # It's easier to rebuild or mask.
                # Let's build new adj lists for the 30 vertices.
                
                adj_30_0 = [0] * 30
                adj_30_1 = [0] * 30
                
                for new_i, old_i in enumerate(vertices):
                    for new_j, old_j in enumerate(vertices):
                        if new_i == new_j: continue
                        # Check if edge exists in original graph
                        # bit old_j in adj_31_0[old_i]
                        if adj_31_0[old_i] & (1 << old_j):
                            adj_30_0[new_i] |= (1 << new_j)
                        else:
                            adj_30_1[new_i] |= (1 << new_j)
                            
                if not has_k222_bitset(adj_30_0, 30) and not has_k222_bitset(adj_30_1, 30):
                    colors_30 = []
                    for i in range(30):
                        for j in range(i + 1, 30):
                            if adj_30_0[i] & (1 << j):
                                colors_30.append(0)
                            else:
                                colors_30.append(1)
                    return {'n': 30, 'colors': colors_30}

    # If n=29 is valid, return it
    if valid_29:
        return {'n': 29, 'colors': colors_29}
        
    # Fallback to n=17 if something went wrong
    squares17 = {x * x % 17 for x in range(1, 17)}
    fallback_colors = [0 if (j - i) % 17 in squares17 else 1 for i in range(17) for j in range(i + 1, 17)]
    return {'n': 17, 'colors': fallback_colors}
