def construct():
    from random import Random
    import time
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k vertices
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate cross edges and C-C edges
        cross_edges = {}  # v -> set of C-neighbors
        cc_edges = set()  # C-C edges
        
        for (a, b) in edges:
            if a < k and b < k:
                cc_edges.add((a, b))
            elif a >= k and b < k:
                # b is in C, a is in I
                if a not in cross_edges:
                    cross_edges[a] = set()
                cross_edges[a].add(b)
            elif a < k and b >= k:
                if b not in cross_edges:
                    cross_edges[b] = set()
                cross_edges[b].add(a)
        
        # Track which C-C edges are covered
        covered_cc = set()
        cliques = []
        
        # Process I-vertices
        for v in sorted(cross_edges.keys()):
            S = set(cross_edges[v])
            while S:
                # Greedily find largest subset T of S where all internal CC edges are uncovered
                # Start with all of S, remove conflicts
                T = list(S)
                # Greedy: try to include vertices one by one
                T_new = []
                for c in sorted(T):
                    # Check if adding c to T_new creates any covered CC edge
                    ok = True
                    for c2 in T_new:
                        pair = (min(c, c2), max(c, c2))
                        if pair in covered_cc:
                            ok = False
                            break
                    if ok:
                        T_new.append(c)
                if not T_new:
                    # Shouldn't happen since single vertex is always ok
                    T_new = [sorted(S)[0]]
                # Create clique
                clique = [v] + T_new
                cliques.append(clique)
                # Mark CC edges within T_new as covered
                for i in range(len(T_new)):
                    for j in range(i + 1, len(T_new)):
                        pair = (min(T_new[i], T_new[j]), max(T_new[i], T_new[j]))
                        covered_cc.add(pair)
                # Remove T_new from S
                S -= set(T_new)
        
        # Cover remaining CC edges
        remaining_cc = cc_edges - covered_cc
        if remaining_cc:
            # Greedy clique cover of remaining graph
            rem_edges = set(remaining_cc)
            vertices_involved = set()
            for (a, b) in rem_edges:
                vertices_involved.add(a)
                vertices_involved.add(b)
            
            while rem_edges:
                # Find a maximal clique in the remaining graph
                # Start with the vertex of highest degree
                degrees = {v: 0 for v in vertices_involved}
                for (a, b) in rem_edges:
                    degrees[a] = degrees.get(a, 0) + 1
                    degrees[b] = degrees.get(b, 0) + 1
                
                if not degrees:
                    break
                    
                start = max(degrees, key=lambda x: degrees[x])
                # Greedy: build clique starting from start
                clique = [start]
                candidates = [v for v in vertices_involved if v != start]
                for v in sorted(candidates):
                    # Check if v is adjacent to all in clique in rem_edges
                    ok = True
                    for c in clique:
                        pair = (min(v, c), max(v, c))
                        if pair not in rem_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(v)
                
                if len(clique) < 2:
                    # Take any edge
                    a, b = next(iter(rem_edges))
                    clique = [a, b]
                
                cliques.append(clique)
                # Remove edges within clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        rem_edges.discard(pair)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        results.append(cliques)
    
    return results
