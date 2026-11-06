def construct():
    import random
    import heapq

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
                    edges.add((u, v))
        return adj, edges

    def solve(n, adj, edges):
        remaining = set(edges)
        cliques = []
        
        # Precompute neighbors for faster lookup if needed, 
        # but we are working with 'remaining' set changes.
        
        while remaining:
            # 1. Find a vertex with high degree in the remaining graph
            # To do this efficiently, we can iterate through remaining edges 
            # and count degrees, or just pick a random edge and pick one of its endpoints?
            # Better: Pick the vertex with the highest degree in G_remaining.
            
            deg = {v: 0 for v in range(n)}
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Find max degree vertex
            max_deg = -1
            start_v = -1
            for v in range(n):
                if deg[v] > max_deg:
                    max_deg = deg[v]
                    start_v = v
            
            if start_v == -1:
                break # Should not happen if remaining is not empty

            # 2. Build a maximal clique starting from start_v in G_remaining
            clique = [start_v]
            # Candidates are neighbors of start_v in G_remaining
            candidates = set()
            for u, v in remaining:
                if u == start_v:
                    candidates.add(v)
                elif v == start_v:
                    candidates.add(u)
            
            # Greedily add vertices to clique
            while candidates:
                # Pick candidate with highest degree within the potential clique extension
                # A simple heuristic: pick one with max degree in the subgraph of candidates
                # Or just pick one arbitrarily? Let's pick one with max connections to other candidates.
                
                best_cand = None
                best_score = -1
                for c in candidates:
                    # Count how many other candidates are connected to c in G_remaining
                    # And c must be connected to all current clique members (it is, by construction of candidates set initially, 
                    # but we need to ensure it remains connected to ALL clique members as we add more)
                    
                    # Actually, the standard greedy maximal clique algorithm:
                    # Candidates must be connected to ALL vertices in `clique`.
                    # Since we only add to candidates if they are connected to the new vertex, 
                    # and they were connected to previous ones, this invariant holds.
                    
                    score = 0
                    for c2 in candidates:
                        if c == c2: continue
                        if (min(c, c2), max(c, c2)) in remaining:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = c
                
                if best_cand is None:
                    break

                clique.append(best_cand)
                
                # Update candidates: 
                # 1. Remove best_cand
                candidates.discard(best_cand)
                # 2. Remove any candidate not connected to best_cand
                to_remove = []
                for c in candidates:
                    if (min(c, best_cand), max(c, best_cand)) not in remaining:
                        to_remove.append(c)
                for c in to_remove:
                    candidates.discard(c)

            cliques.append(clique)
            
            # 3. Remove all edges within the formed clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    remaining.discard(pair)

        return cliques

    adj1, edges1 = instance(32, 0.6, 601)
    adj2, edges2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1, edges1)
    cliques2 = solve(64, adj2, edges2)
    
    return [cliques1, cliques2]
