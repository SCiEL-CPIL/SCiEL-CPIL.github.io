import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0 # 10s margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_maximal_clique(vertices, covered_edges):
        # vertices: set of vertices
        # covered_edges: set of covered edges in C
        # Returns a maximal clique in the graph induced by vertices where edge (u,v) exists if (u,v) not in covered_edges
        if not vertices:
            return set()
        
        # Calculate degrees in the available graph
        # Available graph G on vertices: edge (u,v) if (u,v) not in covered_edges
        # We want a large clique.
        # Heuristic: Greedy by degree.
        
        # To speed up, we can just try a few random orderings or use a fixed heuristic.
        # Let's use the "degree" heuristic.
        
        # Compute degree of each vertex in G
        deg = {}
        for u in vertices:
            deg[u] = 0
        for u in vertices:
            for v in vertices:
                if u < v and (u, v) not in covered_edges:
                    deg[u] += 1
                    deg[v] += 1
        
        # Sort by degree descending
        sorted_verts = sorted(vertices, key=lambda x: deg[x], reverse=True)
        
        clique = set()
        candidates = set(sorted_verts)
        
        while candidates:
            # Pick vertex with highest degree in current candidate set?
            # Or just pick the first one in sorted_verts that is still in candidates?
            # The degree changes as we restrict.
            # Let's pick the vertex in candidates with the highest degree in the induced subgraph of candidates.
            # This is O(|V|^2) per step. |V| <= 30. 30*30*30 = 27000 ops. Fast.
            
            best_v = None
            best_deg = -1
            for v in candidates:
                # Count neighbors in candidates
                d = 0
                for u in candidates:
                    if u == v: continue
                    e = (min(u, v), max(u, v))
                    if e not in covered_edges:
                        d += 1
                if d > best_deg:
                    best_deg = d
                    best_v = v
            
            if best_v is None:
                break
            
            clique.add(best_v)
            # Restrict candidates to neighbors of best_v in G
            new_candidates = set()
            for u in candidates:
                if u == best_v: continue
                e = (min(u, best_v), max(u, best_v))
                if e not in covered_edges:
                    new_candidates.add(u)
            candidates = new_candidates
            
        return clique

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # N(v) for v in I
        N = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edges:
                    nbrs.add(u)
            N[v] = nbrs
        
        covered_cc = set()
        
        # Sort I vertices by neighborhood size (ascending)
        sorted_I = sorted(range(k, n), key=lambda v: len(N[v]))
        
        for v in sorted_I:
            if time.time() > DEADLINE:
                # Fallback: just return what we have + 2-cliques for rest?
                # Better to break and handle remaining edges
                break
            
            nbrs = N[v]
            if not nbrs:
                continue
            
            remaining_nbrs = set(nbrs)
            
            while remaining_nbrs:
                # Find maximal clique in remaining_nbrs w.r.t covered_cc
                S = find_maximal_clique(remaining_nbrs, covered_cc)
                
                if not S:
                    # Should not happen if remaining_nbrs is not empty and no edges covered between them?
                    # Wait, if all edges between remaining_nbrs are covered, then no clique of size > 1.
                    # But we can always form a clique of size 1? No, clique must have >= 2 vertices.
                    # If S is empty, it means no edges available?
                    # Actually, find_maximal_clique returns a set. If it returns empty, it means no vertex?
                    # My implementation returns a set. If candidates is not empty, it picks best_v.
                    # So S should have at least 1 vertex.
                    # But a clique of size 1 is not allowed in the output (must be >= 2).
                    # However, if S has size 1, say {u}, then {u, v} is a clique of size 2.
                    # So we can still add it.
                    pass
                
                clique = sorted(S) + [v]
                cliques.append(clique)
                
                # Mark C-C edges within S as covered
                s_list = sorted(S)
                for i in range(len(s_list)):
                    for j in range(i+1, len(s_list)):
                        covered_cc.add((s_list[i], s_list[j]))
                
                remaining_nbrs -= S
        
        # Cover remaining C-C edges
        # Find all edges in C not in covered_cc
        remaining_edges = set()
        for u in range(k):
            for u2 in range(u+1, k):
                if (u, u2) not in covered_cc:
                    remaining_edges.add((u, u2))
        
        # Clique cover of remaining_edges
        # Vertices involved in remaining_edges
        verts_involved = set()
        for u, v in remaining_edges:
            verts_involved.add(u)
            verts_involved.add(v)
        
        current_remaining = set(remaining_edges)
        
        while current_remaining:
            if time.time() > DEADLINE:
                # Fallback: 2-cliques
                for u, v in current_remaining:
                    cliques.append([u, v])
                break
            
            # Find a maximal clique in the graph defined by current_remaining
            # Vertices are verts_involved
            # Edge exists if in current_remaining
            
            # Helper to find maximal clique in a graph defined by edge set
            def find_clique_in_edge_set(verts, edge_set):
                if not edge_set:
                    return set()
                
                # Calculate degrees
                deg = {v: 0 for v in verts}
                for u, v in edge_set:
                    deg[u] += 1
                    deg[v] += 1
                
                sorted_verts = sorted(verts, key=lambda x: deg[x], reverse=True)
                
                clique = set()
                candidates = set(sorted_verts)
                
                while candidates:
                    best_v = None
                    best_deg = -1
                    for v in candidates:
                        d = 0
                        for u in candidates:
                            if u == v: continue
                            e = (min(u, v), max(u, v))
                            if e in edge_set:
                                d += 1
                        if d > best_deg:
                            best_deg = d
                            best_v = v
                    
                    if best_v is None:
                        break
                    
                    clique.add(best_v)
                    new_candidates = set()
                    for u in candidates:
                        if u == best_v: continue
                        e = (min(u, best_v), max(u, best_v))
                        if e in edge_set:
                            new_candidates.add(u)
                    candidates = new_candidates
                    
                return clique

            S = find_clique_in_edge_set(verts_involved, current_remaining)
            
            if not S:
                # No clique found? Should not happen if edge_set not empty.
                # Pick an edge and make a 2-clique
                u, v = next(iter(current_remaining))
                S = {u, v}
            
            clique = sorted(S)
            cliques.append(clique)
            
            # Remove edges of S from current_remaining
            s_list = sorted(S)
            for i in range(len(s_list)):
                for j in range(i+1, len(s_list)):
                    current_remaining.discard((s_list[i], s_list[j]))
            
            # Update verts_involved? Not strictly necessary if we pass full set, 
            # but find_clique_in_edge_set uses verts to calculate degrees.
            # If a vertex has no edges left, it won't be in candidates effectively.
            # But it's better to prune verts_involved to those with degree > 0 in current_remaining
            # for efficiency, but k is small.
        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
