import sys

def construct():
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        remaining_adj = list(adj)
        cliques = []
        
        def find_max_clique(adj_list, P):
            best_clique = []
            
            def expand(P, X, R):
                nonlocal best_clique
                if not P:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                
                # Simple bound
                if len(R) + P.bit_count() <= len(best_clique):
                    return

                # Pivot selection
                pivot = -1
                max_deg = -1
                p_temp = P
                while p_temp:
                    lsb = p_temp & -p_temp
                    idx = lsb.bit_length() - 1
                    deg = (adj_list[idx] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        pivot = idx
                    p_temp ^= lsb
                
                if pivot == -1:
                    return
                
                candidates = P & ~(adj_list[pivot] & P)
                
                c_temp = candidates
                while c_temp:
                    lsb = c_temp & -c_temp
                    v = lsb.bit_length() - 1
                    new_P = P & adj_list[v]
                    expand(new_P, X, R + [v])
                    P ^= lsb
                    X |= lsb
                    c_temp ^= lsb
            
            expand(P, 0, [])
            return best_clique

        while True:
            has_edges = False
            active_mask = 0
            for i in range(n):
                if remaining_adj[i]:
                    has_edges = True
                    active_mask |= (1 << i)
            if not has_edges:
                break
                
            best_clique = find_max_clique(remaining_adj, active_mask)
            
            if not best_clique:
                # Fallback
                for i in range(n):
                    if remaining_adj[i]:
                        lsb = remaining_adj[i] & -remaining_adj[i]
                        j = lsb.bit_length() - 1
                        best_clique = [i, j]
                        break
            
            for i in range(len(best_clique)):
                u = best_clique[i]
                mask = 0
                for j in range(len(best_clique)):
                    if i != j:
                        mask |= (1 << best_clique[j])
                remaining_adj[u] &= ~mask
                
            cliques.append(best_clique)
            
        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    
    return [res1, res2]
