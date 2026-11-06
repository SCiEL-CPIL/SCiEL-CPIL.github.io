def construct():
    import random
    import heapq

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
                    edges.append((u, v))
        return adj, edges

    def solve(n, adj, edges):
        # Use a set for fast lookup of remaining edges
        remaining = set(tuple(sorted(e)) for e in edges)
        if not remaining:
            return []
        
        cliques = []
        
        # Precompute initial degrees or just work with the set
        # To find the "best" edge, we need to estimate the clique size.
        # Score of edge (u,v) is the number of common neighbors w such that (u,w) and (v,w) are in remaining.
        
        while remaining:
            best_edge = None
            best_score = -1
            best_clique = None
            
            # Iterate through remaining edges to find the one with the largest potential clique
            # This is O(E * N) which might be slow for n=64 (E ~ 1000, N=64 -> 64k ops per step).
            # Number of steps is roughly E / avg_clique_size.
            # For n=64, p=0.55, E ~ 1100. Max clique size ~ 10-15?
            # Steps ~ 100. 100 * 64k = 6.4M ops. This is fine in Python if optimized.
            
            # Optimization: Instead of iterating all edges, we can maintain a priority queue?
            # But the graph changes (edges removed), so scores change.
            # Recalculating is safer.
            
            for edge in remaining:
                u, v = edge
                # Count common neighbors in the remaining graph
                count = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check if (u,w) and (v,w) are in remaining
                    # Since remaining is a set of tuples, we need to check both
                    # Note: edge is (u,v) with u<v. 
                    # For w, we need to check (min(u,w), max(u,w)) and (min(v,w), max(v,w))
                    e1 = (u, w) if u < w else (w, u)
                    e2 = (v, w) if v < w else (w, v)
                    if e1 in remaining and e2 in remaining:
                        count += 1
                
                if count > best_score:
                    best_score = count
                    best_edge = edge
                    best_clique = [u, v]
            
            # Now expand the best_edge's clique greedily
            # We have the best_edge (u,v). We want to add vertices w that are connected to all in clique.
            # We prioritize adding vertices that have high connectivity to the rest of the clique?
            # Or just greedily add any valid vertex?
            # Let's add vertices in order of their degree in the remaining graph restricted to the clique's potential.
            
            u, v = best_edge
            clique = [u, v]
            
            # Candidates are vertices w such that (u,w) and (v,w) are in remaining.
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in remaining and e2 in remaining:
                    # Calculate score for w: how many other candidates is it connected to?
                    # Or just add it if it connects to all current clique members.
                    # We will do a greedy expansion:
                    # 1. Start with candidates list.
                    # 2. Pick w with max common neighbors in candidates?
                    # Simpler: Just iterate and add if valid.
                    pass
            
            # Better expansion strategy:
            # 1. Get all candidates C.
            # 2. While C is not empty:
            #    - Pick w in C with max degree within C (or just any).
            #    - Add w to clique.
            #    - Remove from C any vertex x that is not connected to w in remaining.
            
            C = []
            for w in range(n):
                if w == u or w == v:
                    continue
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in remaining and e2 in remaining:
                    C.append(w)
            
            # Greedy selection from C
            while C:
                # Pick the vertex in C with the highest degree within C
                best_w = None
                max_deg = -1
                for w in C:
                    deg = 0
                    for x in C:
                        if w == x:
                            continue
                        e_wx = (w, x) if w < x else (x, w)
                        if e_wx in remaining:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                
                if best_w is None:
                    break
                    
                clique.append(best_w)
                
                # Update C: remove best_w and any vertex not connected to best_w
                new_C = []
                for x in C:
                    if x == best_w:
                        continue
                    e_bw = (best_w, x) if best_w < x else (x, best_w)
                    if e_bw in remaining:
                        new_C.append(x)
                C = new_C
            
            # Remove all edges within the clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    pair = tuple(sorted((clique[i], clique[j])))
                    remaining.discard(pair)
            
            cliques.append(clique)
            
        return cliques

    adj1, edges1 = instance(32, 0.6, 601)
    adj2, edges2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1, edges1)
    cliques2 = solve(64, adj2, edges2)
    
    return [cliques1, cliques2]
