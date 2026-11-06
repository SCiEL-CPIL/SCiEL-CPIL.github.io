"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import sys
    sys.setrecursionlimit(10000)
    
    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [[False]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
        return adj
    
    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        P = set(range(n))
        X = set()
        R = set()
        
        def bk(P, X, R):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot with max |N(u) ∩ P|
            if P:
                u = max(P, key=lambda x: sum(1 for y in P if adj[x][y]))
            else:
                u = -1
            candidates = P - {u} if u >= 0 else set(P)
            for v in list(candidates):
                bk(P & {x for x in P if adj[v][x]}, X & {x for x in X if adj[v][x]}, R | {v})
                P.remove(v)
                X.add(v)
        
        bk(P, X, R)
        return cliques
    
    def solve_clique_transversal(adj, n):
        """Find minimum set of vertices intersecting every maximal clique of size >= 2."""
        cliques = bron_kerbosch(adj, n)
        if not cliques:
            return []
        
        # Map each vertex to the set of clique indices it covers
        m = len(cliques)
        vertex_cliques = [set() for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)
        
        # Only consider vertices that cover at least one clique
        useful = [v for v in range(n) if vertex_cliques[v]]
        if not useful:
            return []
        
        # Precompute coverage as bitmasks for speed (if m <= 64)
        if m <= 64:
            cover_mask = [0] * n
            for v in range(n):
                mask = 0
                for ci in vertex_cliques[v]:
                    mask |= (1 << ci)
                cover_mask[v] = mask
            
            all_mask = (1 << m) - 1
            
            # Greedy upper bound
            uncovered = all_mask
            solution = []
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not cover_mask[v]:
                        continue
                    cnt = bin(uncovered & cover_mask[v]).count('1')
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1:
                    break
                solution.append(best_v)
                uncovered &= ~cover_mask[best_v]
            ub = len(solution)
            
            # Branch and bound
            best_solution = solution[:]
            
            def bnb(selected, uncovered, depth):
                nonlocal ub
                if len(selected) >= ub:
                    return
                if uncovered == 0:
                    if len(selected) < ub:
                        ub = len(selected)
                        best_solution = selected[:]
                    return
                
                # Lower bound: greedy set cover lower bound
                # Simple LB: number of cliques still uncovered / max coverage per vertex
                max_cov = 0
                for v in range(n):
                    if v in selected:
                        continue
                    c = bin(uncovered & cover_mask[v]).count('1')
                    if c > max_cov:
                        max_cov = c
                if max_cov == 0:
                    return
                lb = len(selected) + (uncovered.bit_count() + max_cov - 1) // max_cov
                if lb >= ub:
                    return
                
                # Choose vertex to branch on: pick uncovered clique with fewest candidate vertices
                # Find an uncovered clique
                ci = 0
                while ci < m and not (uncovered & (1 << ci)):
                    ci += 1
                if ci >= m:
                    return
                
                # Vertices that cover clique ci and are not yet selected
                candidates = []
                for v in cliques[ci]:
                    if v not in selected and (uncovered & cover_mask[v]):
                        candidates.append(v)
                
                if not candidates:
                    return
                
                # Sort candidates by coverage (greedy heuristic)
                candidates.sort(key=lambda v: bin(uncovered & cover_mask[v]).count('1'), reverse=True)
                
                for v in candidates:
                    bnb(selected + [v], uncovered & ~cover_mask[v], depth + 1)
            
            bnb([], all_mask, 0)
            return best_solution
        
        else:
            # For larger m, use greedy + local search
            uncovered = set(range(m))
            solution = []
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not vertex_cliques[v]:
                        continue
                    cnt = len(uncovered & vertex_cliques[v])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1:
                    break
                solution.append(best_v)
                uncovered -= vertex_cliques[best_v]
            return solution
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = random_graph(n, p, seed)
        sol = solve_clique_transversal(adj, n)
        out.append(sorted(sol))
    return out
# EVOLVE-BLOCK-END

