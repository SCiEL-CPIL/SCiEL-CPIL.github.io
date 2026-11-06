import time
import random as _random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
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
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def build_clique_membership(cliques, n):
        vert_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cliques[v].add(ci)
        return vert_cliques

    def branch_and_bound(cliques, n, vert_cliques, time_limit=140.0):
        """
        Exact branch-and-bound for minimum vertex hitting set.
        Uses a greedy upper bound and LP-relaxation-style lower bound.
        """
        m = len(cliques)
        if m == 0:
            return set()
        
        # Precompute: for each clique, list of vertices in it
        clique_verts = [list(c) for c in cliques]
        
        # Initial upper bound via greedy
        def greedy_ub():
            uncovered = set(range(m))
            result = set()
            while uncovered:
                best_v = -1
                best_cov = 0
                for v in range(n):
                    if v in result:
                        continue
                    cov = len(uncovered & vert_cliques[v])
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v == -1 or best_cov == 0:
                    break
                result.add(best_v)
                uncovered -= vert_cliques[best_v]
            return result, len(result)
        
        initial_sol, ub = greedy_ub()
        
        # Sort vertices by coverage descending for branching
        vert_order = sorted(range(n), key=lambda v: -len(vert_cliques[v]))
        
        best_sol = set(initial_sol)
        best_size = ub
        
        # Lower bound: for each uncovered clique, we need at least 1 vertex.
        # Better: use a packing argument. 
        # Simple LB: number of disjoint uncovered cliques (each needs distinct vertex)
        # Even better: LP dual - assign weights to cliques such that sum weights <= 1 for each vertex
        # For simplicity, use: LB = max over all cliques of (1 if uncovered) + ... 
        # Actually, a valid LB is: for any set of pairwise disjoint cliques, we need one vertex per clique.
        # Find maximal set of disjoint uncovered cliques greedily.
        
        def disjoint_clique_lb(uncovered):
            """Find a greedy maximal set of pairwise disjoint cliques from uncovered."""
            used_verts = set()
            count = 0
            # Sort uncovered cliques by size ascending (smaller cliques are harder to satisfy)
            for ci in sorted(uncovered, key=lambda i: len(clique_verts[i])):
                if ci not in uncovered:
                    continue
                cverts = clique_verts[ci]
                if not any(v in used_verts for v in cverts):
                    used_verts.update(cverts)
                    count += 1
            return count
        
        # Branch and bound
        # State: current selected set, current uncovered cliques
        # Branch on a vertex: include or exclude
        # To make it efficient, branch on cliques: pick an uncovered clique, 
        # branch on which vertex in it to include.
        
        def bb(selected, uncovered):
            nonlocal best_sol, best_size
            
            if time.time() - start_time > time_limit:
                return
            
            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = set(selected)
                return
            
            # Lower bound
            lb = disjoint_clique_lb(uncovered)
            if len(selected) + lb >= best_size:
                return
            
            # Pick the smallest uncovered clique to branch on (most constrained)
            min_ci = min(uncovered, key=lambda i: len(clique_verts[i]))
            cverts = clique_verts[min_ci]
            
            # Branch: for each vertex in this clique, try adding it
            # Order by how many other uncovered cliques it covers
            cverts_sorted = sorted(cverts, key=lambda v: -len(uncovered & vert_cliques[v]))
            
            for v in cverts_sorted:
                if time.time() - start_time > time_limit:
                    return
                
                new_selected = set(selected)
                new_selected.add(v)
                new_uncovered = set(uncovered)
                new_uncovered -= vert_cliques[v]
                
                bb(new_selected, new_uncovered)
        
        # Start with empty selection and all cliques uncovered
        bb(set(), set(range(m)))
        
        return best_sol

    def improve_with_local_search(sol, cliques, n, vert_cliques):
        """Further improve solution with local search."""
        if not sol:
            return sol
        
        m = len(cliques)
        result = set(sol)
        
        # Build coverage count
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        
        improved = True
        iters = 0
        max_iters = 50
        while improved and iters < max_iters and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            
            # Try removals
            for v in list(result):
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps
            for v in result:
                unique_cliques = set()
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        unique_cliques.add(ci)
                
                if not unique_cliques:
                    continue
                
                for w in range(n):
                    if w in result:
                        continue
                    if all(ci in vert_cliques[w] for ci in unique_cliques):
                        # Accept swap
                        for ci in vert_cliques[v]:
                            cover_count[ci] -= 1
                        result.discard(v)
                        for ci in vert_cliques[w]:
                            cover_count[ci] += 1
                        result.add(w)
                        improved = True
                        break
                if improved:
                    break
        
        return result

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = random_graph(n, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        vert_cliques = build_clique_membership(cliques, n)
        
        # Run branch and bound
        sol = branch_and_bound(cliques, n, vert_cliques)
        
        # Improve with local search
        sol = improve_with_local_search(sol, cliques, n, vert_cliques)
        
        results.append(sorted(sol))
    
    return results
