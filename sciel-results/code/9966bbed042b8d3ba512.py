"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0

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

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # Compute N(v) for each v in I
        N = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edges:
                    nbrs.add(u)
            N[v] = nbrs
        
        # Track which C-C edges are covered
        covered_cc = set()
        
        # Helper to check if a set of vertices forms a clique in the available graph
        def is_available_clique(vertices):
            verts = sorted(vertices)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    e = (verts[i], verts[j])
                    if e in covered_cc:
                        return False
            return True

        # Helper to get all C-C edges within a set of vertices
        def get_cc_edges(vertices):
            verts = sorted(vertices)
            res = set()
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    res.add((verts[i], verts[j]))
            return res

        # Sort I vertices by size of their neighborhood in C (ascending)
        # Processing smaller neighborhoods first leaves more edges available for larger ones
        i_vertices = sorted([v for v in range(k, n) if N[v]], key=lambda v: len(N[v]))
        
        for v in i_vertices:
            nbrs = N[v]
            if not nbrs:
                continue
            
            remaining_nbrs = set(nbrs)
            
            # Greedy coloring of the conflict graph on remaining_nbrs
            # Conflict exists if edge is already covered
            # We want to partition remaining_nbrs into independent sets of the conflict graph
            # which are cliques in the available graph.
            
            # We'll use a simple greedy approach:
            # Repeatedly find a maximal available clique, remove it, repeat.
            # To improve, we can pick the "best" starting vertex or use Welsh-Powell on conflict graph.
            
            while remaining_nbrs:
                # Find a maximal clique in the available graph induced by remaining_nbrs
                # Greedy: pick a vertex, then add others that are compatible.
                # To get larger cliques, maybe pick vertex with max degree in available graph?
                
                # Calculate available degree for each vertex in remaining_nbrs
                avail_deg = {}
                for u in remaining_nbrs:
                    deg = 0
                    for w in remaining_nbrs:
                        if u == w: continue
                        e = (min(u, w), max(u, w))
                        if e not in covered_cc:
                            deg += 1
                    avail_deg[u] = deg
                
                # Start with vertex with highest available degree
                start = max(remaining_nbrs, key=lambda x: avail_deg[x])
                S = {start}
                
                changed = True
                while changed:
                    changed = False
                    candidates = list(remaining_nbrs - S)
                    # Sort candidates by available degree to S? Or just iterate
                    for u in candidates:
                        is_connected = True
                        for w in S:
                            e = (min(u, w), max(u, w))
                            if e in covered_cc:
                                is_connected = False
                                break
                        if is_connected:
                            S.add(u)
                            changed = True
                
                # Form clique S U {v}
                clique = sorted(S) + [v]
                cliques.append(clique)
                
                # Mark C-C edges within S as covered
                new_edges = get_cc_edges(S)
                covered_cc.update(new_edges)
                
                # Remove S from remaining_nbrs
                remaining_nbrs -= S
        
        # Now cover remaining C-C edges
        # Try to form larger cliques from remaining edges
        remaining_edges_set = set()
        for u in range(k):
            for u2 in range(u+1, k):
                if (u, u2) in edges and (u, u2) not in covered_cc:
                    remaining_edges_set.add((u, u2))
        
        # Build graph of remaining edges
        rem_adj = {u: set() for u in range(k)}
        for (u, v) in remaining_edges_set:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        remaining_vertices = set(rem_adj.keys())
        
        while remaining_vertices:
            # Find a maximal clique in the remaining graph
            # Greedy
            start = next(iter(remaining_vertices))
            S = {start}
            
            changed = True
            while changed:
                changed = False
                candidates = list(remaining_vertices - S)
                for u in candidates:
                    is_connected = True
                    for w in S:
                        if w not in rem_adj[u]:
                            is_connected = False
                            break
                    if is_connected:
                        S.add(u)
                        changed = True
            
            if len(S) >= 2:
                clique = sorted(S)
                cliques.append(clique)
                # Remove edges within S from remaining_edges_set
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        e = (clique[i], clique[j])
                        if e in remaining_edges_set:
                            remaining_edges_set.remove(e)
                # Update rem_adj
                for u in S:
                    remaining_vertices.discard(u)
                    for v in S:
                        if u != v:
                            if v in rem_adj[u]:
                                rem_adj[u].discard(v)
                            if u in rem_adj[v]:
                                rem_adj[v].discard(u)
            else:
                # Should not happen if remaining_edges_set is not empty, but handle isolated vertices
                remaining_vertices.discard(start)
        
        # If any edges remain (shouldn't happen if logic is correct), cover with 2-cliques
        for e in remaining_edges_set:
            cliques.append(list(e))

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
# EVOLVE-BLOCK-END
