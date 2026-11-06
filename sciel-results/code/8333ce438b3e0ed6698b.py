def construct():
    import random

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return bin(x).count('1')

    def solve(n, adj):
        rem_adj = adj[:]  # Copy
        cliques = []
        
        # Helper to get list of set bits
        def bits(mask):
            res = []
            while mask:
                lsb = mask & -mask
                idx = (lsb.bit_length() - 1)
                res.append(idx)
                mask ^= lsb
            return res

        while True:
            # Find if any edges remain
            # Check if any rem_adj[i] has bits set for j > i
            has_edge = False
            best_u, best_v = -1, -1
            max_common = -1
            
            for u in range(n):
                # Only consider v > u to avoid duplicates
                # Mask out bits <= u
                mask_v = rem_adj[u] & (~((1 << (u + 1)) - 1))
                if mask_v == 0:
                    continue
                has_edge = True
                # Iterate through set bits in mask_v
                m = mask_v
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    # Calculate common neighbors in remaining graph
                    # Common neighbors of u and v are in rem_adj[u] & rem_adj[v]
                    # We want to count how many vertices w exist such that (u,w) and (v,w) are in remaining
                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > max_common:
                        max_common = cnt
                        best_u = u
                        best_v = v
                    m ^= lsb
            
            if not has_edge:
                break
            
            u, v = best_u, best_v
            clique = [u, v]
            
            # Candidates: vertices connected to both u and v in remaining
            candidates = rem_adj[u] & rem_adj[v]
            # Remove u and v from candidates (they are not neighbors of themselves in this mask, but just in case)
            candidates &= ~(1 << u)
            candidates &= ~(1 << v)
            
            while candidates:
                # Find w in candidates with max degree within candidates
                # degree of w in candidates is popcount(rem_adj[w] & candidates)
                best_w = -1
                max_deg = -1
                m = candidates
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    deg = popcount(rem_adj[w] & candidates)
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                    m ^= lsb
                
                clique.append(best_w)
                candidates &= rem_adj[best_w]
                candidates &= ~(1 << best_w)
            
            cliques.append(clique)
            
            # Remove all edges within clique from rem_adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    rem_adj[a] &= ~(1 << b)
                    rem_adj[b] &= ~(1 << a)
        
        return cliques

    adj1 = instance(32, 0.6, 601)
    adj2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1)
    cliques2 = solve(64, adj2)
    
    return [cliques1, cliques2]
