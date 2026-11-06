def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
        
    # Build original adjacency
    orig_adj = {i: set() for i in range(n)}
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)
        
    # Build residual adjacency
    res_adj = {i: set(orig_adj[i]) for i in range(n)}
    
    cliques = []
    start_time = time.time()
    
    # Phase 1: Greedy Construction
    greedy_time_limit = time_limit * 0.5
    while any(len(s) > 0 for s in res_adj.values()):
        current_time = time.time()
        if current_time - start_time > greedy_time_limit:
            break
            
        clique = find_large_clique(res_adj, n)
        
        if not clique:
            # Find any edge
            found = False
            for u in range(n):
                for v in res_adj[u]:
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                if found: break
            if not found: break
                
        # Remove edges in clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                res_adj[u].discard(v)
                res_adj[v].discard(u)
                
        cliques.append(clique)
        
    # If we exited early, finish with fast greedy
    if any(len(s) > 0 for s in res_adj.values()):
        while any(len(s) > 0 for s in res_adj.values()):
            # Find any edge
            found = False
            for u in range(n):
                for v in res_adj[u]:
                    if u < v:
                        C = [u]
                        P = set(res_adj[u])
                        while P:
                            best_u = max(P, key=lambda x: len(P & res_adj[x]))
                            C.append(best_u)
                            P = P & res_adj[best_u]
                        for i in range(len(C)):
                            for j in range(i+1, len(C)):
                                res_adj[C[i]].discard(C[j])
                                res_adj[C[j]].discard(C[i])
                        cliques.append(C)
                        found = True
                        break
                if found: break
            if not found: break

    # Phase 2: Local Search (Merging)
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        merged = False
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                if is_clique_union(cliques[i], cliques[j], orig_adj):
                    new_clique = cliques[i] + cliques[j]
                    cliques.pop(j)
                    cliques.pop(i)
                    cliques.append(new_clique)
                    improved = True
                    merged = True
                    break
            if merged:
                break

    return cliques

def is_clique_union(c1, c2, orig_adj):
    for u in c1:
        for v in c2:
            if v not in orig_adj[u]:
                return False
    return True
