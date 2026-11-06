def construct():
    import sys
    sys.setrecursionlimit(100000)

    def generate_edges(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []

        # adj will store the set of neighbors for each vertex in the CURRENT uncovered subgraph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        result = []

        while True:
            # Check if any edges remain
            remaining_edges = 0
            for u in range(n):
                if adj[u]:
                    remaining_edges += len(adj[u])
            if remaining_edges == 0:
                break
            
            # 1. Select a starting vertex.
            # Heuristic: Pick the vertex with the highest degree in the remaining graph.
            # This corresponds to picking a "central" vertex in the line graph context.
            start_v = max(range(n), key=lambda x: len(adj[x]))
            
            if not adj[start_v]:
                # Should not happen if remaining_edges > 0, but safety check
                # Find any vertex with an edge
                found = False
                for i in range(n):
                    if adj[i]:
                        start_v = i
                        found = True
                        break
                if not found:
                    break

            # 2. Greedily build a maximal clique starting from start_v
            clique = {start_v}
            # Candidates are neighbors of start_v in the remaining graph
            candidates = set(adj[start_v])
            
            # To make the greedy choice better, we can order candidates by their degree in the remaining graph
            # or their degree within the candidate set.
            
            while candidates:
                # Pick candidate u that has the most neighbors within the current candidate set.
                # This helps form a dense sub-clique.
                # If tie, pick one with highest global degree.
                best_u = None
                best_score = -1
                
                for u in candidates:
                    # Count neighbors of u that are also in candidates
                    # Note: adj[u] contains all neighbors in the remaining graph.
                    # We need intersection with candidates.
                    common = len(adj[u] & candidates)
                    # Tie-breaker: global degree
                    global_deg = len(adj[u])
                    score = (common, global_deg)
                    
                    if best_u is None or score > best_score:
                        best_score = score
                        best_u = u
                        
                if best_u is None:
                    break

                # Add best_u to clique
                clique.add(best_u)
                
                # Update candidates: must be adjacent to best_u
                # New candidates are intersection of old candidates and adj[best_u]
                candidates = candidates & adj[best_u]

            # 3. Remove all edges internal to the clique from the remaining graph
            clique_list = list(clique)
            for i in range(len(clique_list)):
                u = clique_list[i]
                for j in range(i + 1, len(clique_list)):
                    v = clique_list[j]
                    if v in adj[u]:
                        adj[u].discard(v)
                        adj[v].discard(u)

            # 4. Add clique to result
            result.append(sorted(clique))

        return result

    return [solve(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
