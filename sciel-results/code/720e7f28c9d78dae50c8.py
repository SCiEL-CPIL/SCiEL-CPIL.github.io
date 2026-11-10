import sys
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        # Generate edges
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        remaining_edges = set(edges)
        cliques = []
        
        # Precompute adjacency for fast neighbor lookup
        # adj[v] is a set of neighbors
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # We need to maintain "remaining" adjacency or just check against remaining_edges
        # Checking against remaining_edges is O(1) per edge check, but iterating neighbors is O(deg).
        # Since we remove edges, the "effective" degree drops.
        
        while remaining_edges:
            # 1. Find vertex with max degree in remaining graph
            # We can approximate this by just picking a vertex from a random edge, 
            # or maintaining degrees.
            # Let's compute degrees for active vertices.
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
                
            # Find max degree vertex
            max_v = -1
            max_deg = -1
            for i in range(n):
                if deg[i] > max_deg:
                    max_deg = deg[i]
                    max_v = i
                    
            if max_v == -1:
                break
                
            # 2. Extend max_v to a maximal clique
            # Greedy extension:
            clique = [max_v]
            # Candidates are neighbors of max_v in remaining graph
            candidates = set()
            for u, v in remaining_edges:
                if u == max_v:
                    candidates.add(v)
                elif v == max_v:
                    candidates.add(u)
                    
            while candidates:
                # Pick candidate with max degree within candidates
                # deg_in_cand[u] = number of neighbors of u in candidates
                best_cand = -1
                best_score = -1
                for u in candidates:
                    # Count neighbors in candidates
                    cnt = 0
                    for w in candidates:
                        if u == w: continue
                        # Check if (u, w) is in remaining_edges
                        if u < w:
                            if (u, w) in remaining_edges:
                                cnt += 1
                        else:
                            if (w, u) in remaining_edges:
                                cnt += 1
                    if cnt > best_score:
                        best_score = cnt
                        best_cand = u
                        
                if best_cand == -1:
                    break
                    
                clique.append(best_cand)
                
                # Update candidates: intersect with neighbors of best_cand
                new_candidates = set()
                for u in candidates:
                    if u == best_cand: continue
                    # Check if edge (best_cand, u) exists in remaining graph
                    if best_cand < u:
                        if (best_cand, u) in remaining_edges:
                            new_candidates.add(u)
                    else:
                        if (u, best_cand) in remaining_edges:
                            new_candidates.add(u)
                candidates = new_candidates
                
            # 3. Remove edges of clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    remaining_edges.discard((u, v))
                    
            cliques.append(clique)
            
        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    
    return [res1, res2]
