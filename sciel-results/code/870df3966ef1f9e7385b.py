def construct():
    import time
    import random

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, cliques, R, P, X):
        # R, P, X are sets of vertices
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection
        pivot = None
        max_count = -1
        # Combine P and X for pivot search? Standard is P, but P U X is better
        candidates = P.union(X)
        for v in candidates:
            count = len(P.intersection(adj[v]))
            if count > max_count:
                max_count = count
                pivot = v
        
        if pivot is None:
            iter_set = P
        else:
            # Standard Bron-Kerbosch with pivot: iterate over P \ N(pivot)
            iter_set = P.difference(adj[pivot])
        
        for v in iter_set:
            bron_kerbosch(adj, cliques, R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)

    def solve_hitting_set(n, cliques):
        if not cliques:
            return []
        
        # Greedy
        solution = set()
        uncovered = list(cliques)
        
        # Precompute for each vertex, which cliques it touches
        # vertex_to_cliques = {v: [idx of cliques containing v]}
        # But cliques change as we cover them.
        
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in solution:
                    continue
                count = 0
                for c in uncovered:
                    if v in c:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None:
                break # Should not happen if graph is connected and cliques exist
                
            solution.add(best_v)
            # Remove covered cliques
            new_uncovered = []
            for c in uncovered:
                if best_v not in c:
                    new_uncovered.append(c)
            uncovered = new_uncovered
        
        # Local Search: Deletion
        changed = True
        while changed:
            changed = False
            for v in list(solution):
                # Check if v is redundant
                # v is redundant if every clique containing v is also covered by solution - {v}
                is_redundant = True
                for c in cliques:
                    if v in c:
                        # Check if any other vertex in c is in solution
                        if not any(u in solution for u in c if u != v):
                            is_redundant = False
                            break
                if is_redundant:
                    solution.remove(v)
                    changed = True
        
        # Local Search: Swap (optional, might be expensive)
        # Try removing v and adding u to see if size decreases
        # This is complex. Greedy + Deletion is usually good enough for a jump.
        
        return sorted(solution)

    out = []
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback to seed behavior if timeout
             touched = set()
             rng = random.Random(seed)
             for u in range(n):
                 for v in range(u + 1, n):
                     if rng.random() < p:
                         touched.add(u)
                         touched.add(v)
             out.append(sorted(touched))
             continue
             
        adj = generate_graph(n, p, seed)
        cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, cliques, set(), set(range(n)), set())
        
        res = solve_hitting_set(n, cliques)
        out.append(res)
        
    return out
