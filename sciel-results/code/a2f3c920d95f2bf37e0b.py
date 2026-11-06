def construct():
    import time
    import sys
    from itertools import combinations
    
    deadline = time.time() + 140  # Safety margin for 150s kill

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        return adj, edges

    def find_maximal_cliques(adj, n):
        """
        Bron-Kerbosch algorithm to find all maximal cliques.
        Uses pivot to improve performance.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2: # We only care about cliques with at least 2 vertices (edges)
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X to maximize |P \ N(u)|
            # If P is empty, X must be empty for us to be in a leaf, but we check above.
            # If P is not empty, pick pivot.
            P_union_X = P.union(X)
            if not P_union_X:
                return
                
            # Pick a vertex with max degree in P (or just first one for simplicity if complex)
            # For dense graphs, picking a high degree vertex is good.
            # Simple pivot: pick any vertex in P_union_X
            # Better: pick u in P_union_X that maximizes len(P - adj[u])
            # To save time, we just pick the first one or one with high degree in P.
            # Let's pick u from P if P else X
            if P:
                # Pick vertex in P with max neighbors in P
                u = max(P, key=lambda v: len(P.intersection(adj[v])))
            else:
                u = next(iter(X))
                
            # Recurse on P \ N(u)
            P_not_Nu = P.difference(adj[u])
            for v in list(P_not_Nu):
                bk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.discard(v)
                X.add(v)

        # Initial call
        # R is empty, P is all vertices, X is empty
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj, edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        edge_set = set(frozenset(e) for e in edges)
        
        # 1. Find all maximal cliques
        # For n=64, this might generate many cliques, but usually manageable.
        # If it's too slow, we can limit it, but let's try first.
        start_time = time.time()
        maximal_cliques = find_maximal_cliques(adj, n)
        
        # If we ran out of time or found too many, fallback to greedy edge picking
        if time.time() - start_time > 10 or len(maximal_cliques) > 5000:
             # Fallback: Greedy set cover using just edges and maybe some triangles?
             # Or just return all edges as cliques of size 2 (valid but suboptimal)
             # Let's try a quick greedy: pick edge, try to extend.
             # This is the old strategy.
             return greedy_fallback(adj, edge_set, n)

        # 2. Greedy Set Cover
        # Candidates: maximal_cliques
        # We want to cover edge_set.
        
        # Pre-compute edge lists for each clique
        clique_edges = []
        for clique in maximal_cliques:
            c_list = list(clique)
            e_list = [frozenset((c_list[i], c_list[j])) for i in range(len(c_list)) for j in range(i+1, len(c_list))]
            clique_edges.append((clique, set(e_list)))
        
        # Sort candidates by size descending? Or just evaluate coverage dynamically.
        # Dynamic evaluation is safer.
        
        uncovered = set(edge_set)
        partition = []
        
        # To speed up, we can track for each candidate how many edges it covers in `uncovered`.
        # But `uncovered` changes.
        # Simple approach: In each step, iterate through candidates and count intersection.
        # Number of candidates might be large. 
        # Optimization: Only consider candidates that have at least 1 uncovered edge.
        
        while uncovered:
            best_clique = None
            best_count = 0
            best_size = 0
            
            for clique, c_edges in clique_edges:
                # Quick check: if no edge in c_edges is in uncovered, skip?
                # Intersection size
                # len(c_edges & uncovered)
                count = len(c_edges & uncovered)
                if count > best_count:
                    best_count = count
                    best_clique = clique
                    best_size = len(clique)
                elif count == best_count and count > 0:
                    # Tie breaker: larger clique is better (removes more edges per clique? 
                    # Actually, if count is same, larger clique might have more "wasted" edges 
                    # that are already covered? No, we only count uncovered.
                    # So if count is same, it doesn't matter much, but larger clique 
                    # might be structurally "better" for future? 
                    # Actually, we want to minimize number of cliques.
                    # If two cliques cover the same number of NEW edges, 
                    # picking the one that leaves a "cleaner" graph for the future is hard to know.
                    # Let's just pick the first one or the larger one.
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique

            if best_clique is None:
                # Should not happen if all edges are in maximal cliques
                # But if we missed some, fallback
                break
            
            partition.append(list(best_clique))
            
            # Remove edges of best_clique from uncovered
            c_list = list(best_clique)
            for i in range(len(c_list)):
                for j in range(i+1, len(c_list)):
                    uncovered.discard(frozenset((c_list[i], c_list[j])))
        
        # If there are still uncovered edges (shouldn't happen), add them as 2-cliques
        if uncovered:
            for e in uncovered:
                partition.append(list(e))
        
        return partition

    def greedy_fallback(adj, edge_set, n):
        # Original strategy, slightly improved
        uncovered = set(edge_set)
        cliques = []
        
        while uncovered:
            # Find vertex with max degree in uncovered
            deg = [0]*n
            for e in uncovered:
                a,b = tuple(e)
                deg[a]+=1
                deg[b]+=1
            start = max(range(n), key=lambda v: deg[v])
            
            clique = [start]
            candidates = [u for u in adj[start] if frozenset((start,u)) in uncovered]
            
            while candidates:
                # Pick candidate with max common neighbors in candidates
                best = max(candidates, key=lambda u: sum(1 for w in candidates if w in adj[u]))
                ok = True
                for c in clique:
                    if frozenset((best,c)) not in uncovered:
                        ok = False
                        break
                if not ok:
                    candidates.remove(best)
                    continue
                clique.append(best)
                candidates = [u for u in candidates if u != best and frozenset((best,u)) in uncovered and u in adj[best]]
            
            if len(clique) >= 2:
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        uncovered.discard(frozenset((clique[i], clique[j])))
                cliques.append(clique)
            else:
                e = next(iter(uncovered))
                a,b = tuple(e)
                cliques.append([a,b])
                uncovered.discard(e)
                
        return cliques

    results = []
    for (n, p, seed) in ((32,0.6,601),(64,0.55,602)):
        if time.time() > deadline:
            # Emergency fallback: return empty or trivial? 
            # Must return valid structure. 
            # If we don't have time, we can't generate the graph properly.
            # But we should have finished the first one.
            # If we are here, we are in trouble.
            # Let's just return a dummy valid partition for the remaining instance
            # to avoid crashing.
            adj, edges = random_graph(n, p, seed)
            # Just return all edges as cliques of size 2
            results.append([list(e) for e in edges])
            continue
            
        cliques = solve_instance(n, p, seed)
        results.append(cliques)
        
    return results

