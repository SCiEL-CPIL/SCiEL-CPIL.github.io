def construct():
    import random

    def instance(n, p, seed):
        rng = random.Random(seed)
        # Build adjacency as bitmasks
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve(n, adj_initial):
        # Copy adj
        adj = adj_initial[:]
        cliques = []
        
        # Helper to check if graph is empty
        def is_empty():
            for u in range(n):
                if adj[u]:
                    return False
            return True

        while not is_empty():
            # 1. Find vertex with max degree
            max_deg = -1
            v = -1
            for u in range(n):
                deg = adj[u].bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if v == -1:
                break

            # 2. Build clique greedily
            clique = [v]
            candidates = adj[v]
            
            # Remove v from candidates to avoid self (though adj[v] doesn't have v bit set usually, 
            # but intersection logic is cleaner if we mask)
            candidates &= ~(1 << v)
            
            while candidates:
                # Find candidate u with max degree in current graph
                best_u = -1
                best_deg = -1
                c = candidates
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    u = lsb.bit_length() - 1
                    deg = adj[u].bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                    c &= c - 1 # clear lowest set bit
                
                if best_u == -1:
                    break
                
                clique.append(best_u)
                
                # Update candidates: must be neighbors of best_u too
                candidates = candidates & adj[best_u]
                # Remove best_u from candidates to prevent re-adding
                candidates &= ~(1 << best_u)

            # 3. Add clique to result
            cliques.append(clique)
            
            # 4. Remove edges within clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, w = clique[i], clique[j]
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
        
        return cliques

    adj1 = instance(32, 0.6, 601)
    adj2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1)
    cliques2 = solve(64, adj2)
    
    return [cliques1, cliques2]
