    def would_create_c6_fast(adj, u, v):
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Optimize by iterating the smaller set in the outer loop
        # The logic is symmetric if we swap u,v and a,d
        if Nu.bit_count() > Nv.bit_count():
            u, v = v, u
            Nu, Nv = Nv, Nu

        # Iterate over a in N(u)\{v}
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            # B = N(a) excluding u, v, and we'll exclude d per-d
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)  # d != a
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                # B = N(a) \ {u,v,d}
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                
                # C = N(d) \ {u,v,a}
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                
                # For each b in B, check if N(b) & C & ~(1<<b) != 0
                mask_b = B
                found = False
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    # Check if any neighbor of b is in C and is not b itself
                    if adj[b] & C & ~(1 << b):
                        found = True
                        break
                
                if found:
                    return True
        
        return False
