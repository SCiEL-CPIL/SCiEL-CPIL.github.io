def construct():
    import time
    import random
    
    start_time = time.time()
    DEADLINE = 140.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def find_max_clique(adj, vertices):
        """Find maximum clique in subgraph induced by vertices using Bron-Kerbosch."""
        if not vertices:
            return []
        
        vertices = set(vertices)
        
        # Greedy initial solution
        greedy = []
        remaining = set(vertices)
        while remaining:
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            greedy.append(v)
            remaining = remaining & adj[v]
        
        best = greedy[:]
        
        # Bron-Kerbosch with pivot
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = R[:]
                return
            
            if P:
                # Pivot
                pivot = max(P, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R + [v]
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)
        
        bk([], set(vertices), set())
        return best
    
    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        uncovered = set(edges)
        cliques = []
        
        while uncovered and time_left() > 2.0:
            vertices_with_uncovered = set()
            for u, v in uncovered:
                vertices_with_uncovered.add(u)
                vertices_with_uncovered.add(v)
            
            if not vertices_with_uncovered:
                break
            
            sub_adj = {v: set() for v in vertices_with_uncovered}
            for u, v in uncovered:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            best_clique = find_max_clique(sub_adj, vertices_with_uncovered)
            
            if len(best_clique) < 2:
                u, v = next(iter(uncovered))
                cliques.append([u, v])
                uncovered.discard((u, v))
                uncovered.discard((v, u))
            else:
                cliques.append(best_clique)
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        a, b = best_clique[i], best_clique[j]
                        uncovered.discard((a, b))
                        uncovered.discard((b, a))
        
        for u, v in uncovered:
            cliques.append([u, v])
        
        # Local improvement
        if time_left() > 3.0:
            cliques = local_improve(cliques, adj, n)
        
        return cliques
    
    def local_improve(cliques, adj, n):
        changed = True
        while changed and time_left() > 2.0:
            changed = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    if time_left() < 2.0:
                        break
                    ci = set(cliques[i])
                    cj = set(cliques[j])
                    union = ci | cj
                    verts = list(union)
                    is_clique = True
                    for a_idx in range(len(verts)):
                        for b_idx in range(a_idx + 1, len(verts)):
                            if verts[b_idx] not in adj[verts[a_idx]]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        cliques[i] = list(union)
                        cliques.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return cliques
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
