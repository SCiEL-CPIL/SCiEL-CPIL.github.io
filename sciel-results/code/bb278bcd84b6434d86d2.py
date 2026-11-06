import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the seed program."""
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u in P ∪ X to minimize branching
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            
            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])
            
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Enumerate maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        m = len(cliques)
        
        # Build vertex_cliques: map vertex index to set of clique indices
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)
                
        # Identify active vertices (those in at least one clique)
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        
        if not active_vertices:
            return []
        
        # Dominance Pruning: If C_u is a subset of C_v, u is dominated by v.
        # Sort active vertices by size of clique set descending.
        active_vertices_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))
        
        keep_vertices = set()
        for v in active_vertices_sorted:
            # Check if v is dominated by any already kept vertex
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)
                
        candidates = list(keep_vertices)
        
        if not candidates:
             return []

        # Branch and Bound
        # State: selected (set of vertices), uncovered (set of clique indices)
        best_size = float('inf')
        best_solution = None
        
        # Greedy Set Cover for initial upper bound
        def greedy_cover(uncovered_set, candidates_list):
            selected = set()
            unc = set(uncovered_set)
            while unc:
                best_v = -1
                best_cov = 0
                for v in candidates_list:
                    if v in selected: continue
                    cov = len(unc & vertex_cliques[v])
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                unc -= vertex_cliques[best_v]
            return selected

        init_sol = greedy_cover(set(range(m)), candidates)
        best_size = len(init_sol)
        best_solution = set(init_sol)
        
        # B&B
        def branch_and_bound(selected, uncovered):
            nonlocal best_size, best_solution
            
            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return
            
            if len(selected) >= best_size:
                return
                
            # Lower Bound
            max_cov = 0
            for v in candidates:
                if v in selected: continue
                cov = len(uncovered & vertex_cliques[v])
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return

            lb = len(selected) + math.ceil(len(uncovered) / max_cov)
            if lb >= best_size:
                return
                
            if time.time() - start_time > DEADLINE:
                return

            # Choose a clique to branch on
            # Heuristic: Choose the clique with the fewest available vertices
            target_clique_idx = -1
            min_avail = float('inf')
            
            for ci in uncovered:
                c = cliques[ci]
                avail = 0
                for v in c:
                    if v in candidates and v not in selected:
                        avail += 1
                if avail < min_avail:
                    min_avail = avail
                    target_clique_idx = ci
                    if min_avail == 1:
                        break
                    
            if target_clique_idx == -1:
                return
            
            target_clique = cliques[target_clique_idx]
            verts_in_target = [v for v in target_clique if v in candidates and v not in selected]
            
            if not verts_in_target:
                return
                
            verts_in_target.sort(key=lambda v: -len(uncovered & vertex_cliques[v]))
            
            for v in verts_in_target:
                new_selected = selected | {v}
                new_uncovered = uncovered - vertex_cliques[v]
                branch_and_bound(new_selected, new_uncovered)
                
        initial_uncovered = set(range(m))
        branch_and_bound(set(), initial_uncovered)
        
        if best_solution is None:
            return []
            
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)
        
    return results
