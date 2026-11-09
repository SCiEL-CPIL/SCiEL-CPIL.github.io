"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges (clique)
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # C-I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def dsatur_coloring(vertices, adj):
        """
        Colors the graph (vertices, adj) using DSATUR heuristic.
        Returns a list of color classes (lists of vertices).
        adj is a dict of vertex -> set of neighbors.
        """
        if not vertices:
            return []

        uncolored = set(vertices)
        colors = {}
        color_sets = {v: set() for v in vertices}
        deg = {v: len(adj[v]) for v in vertices}

        while uncolored:
            best_v = None
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = len(color_sets[v])
                if sat > best_sat or (sat == best_sat and deg[v] > best_deg):
                    best_sat = sat
                    best_deg = deg[v]
                    best_v = v

            c = 0
            while c in color_sets[best_v]:
                c += 1
            colors[best_v] = c
            uncolored.remove(best_v)

            for u in adj[best_v]:
                if u in uncolored:
                    if c not in color_sets[u]:
                        color_sets[u].add(c)

        max_color = max(colors.values()) if colors else 0
        color_classes = [[] for _ in range(max_color + 1)]
        for v, c in colors.items():
            color_classes[c].append(v)

        return [cls for cls in color_classes if cls]

    def polish_clique_cover(vertices, adj, color_classes, deadline):
        """
        Local search: try to merge pairs of color classes if their union
        is still an independent set in the coloring graph (i.e., a clique
        in the original edge graph).  This reduces the number of cliques.
        """
        if len(color_classes) <= 1:
            return color_classes

        # Build a set for fast membership
        cls_set = [set(cls) for cls in color_classes]
        active = list(range(len(color_classes)))

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to merge every pair
            i_list = list(active)
            for ai in range(len(i_list)):
                if time.time() > deadline:
                    break
                a = i_list[ai]
                for bi in range(ai + 1, len(i_list)):
                    b = i_list[bi]
                    # Check if union is independent in adj
                    # i.e., no edge between any vertex in cls[a] and cls[b]
                    ok = True
                    for u in cls_set[a]:
                        for w in cls_set[b]:
                            if w in adj.get(u, set()):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # Merge b into a
                        for v in cls_set[b]:
                            cls_set[a].add(v)
                        cls_set[b] = set()
                        # Remove b from active
                        active.remove(b)
                        improved = True
                        break
                if improved:
                    break

        # Rebuild list
        result = []
        for idx in active:
            if cls_set[idx]:
                result.append(sorted(cls_set[idx]))
        return result

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # Identify C and I
        C = list(range(k))
        I = list(range(k, n))
        
        # Adjacency for C
        # We only care about edges within C for the "used" tracking
        # and edges between C and I for the partitioning.
        
        # N_C[v] for v in I: set of neighbors in C
        N_C = {v: set() for v in I}
        for v in I:
            for u in C:
                if (u, v) in edges:
                    N_C[v].add(u)
                    
        # used_edges: set of edges in C that are already covered by a clique involving an I-vertex
        used_edges = set()
        
        # Sort I vertices by size of N_C (ascending)
        # Processing constrained vertices first
        I_sorted = sorted(I, key=lambda v: len(N_C[v]))
        
        # For each v in I, partition N_C[v] into cliques in the available graph on C
        for v in I_sorted:
            if time.time() > DEADLINE:
                # Fallback: cover remaining with 2-cliques if time is critical
                break
                
            neighbors = N_C[v]
            if not neighbors:
                continue
                
            # Build available graph on neighbors
            # A vertex u is in neighbors.
            # Edge (u, w) is available if (u, w) is in edges AND (u, w) not in used_edges.
            # Note: All pairs in C are in edges. So just check used_edges.
            
            # We need to partition neighbors into sets S such that S is a clique in available graph.
            # This is equivalent to coloring the complement of the available graph.
            
            # Build complement adjacency for neighbors
            # Comp edge (u, w) exists if (u, w) is NOT available.
            # i.e., if (u, w) is in used_edges.
            
            comp_adj = {u: set() for u in neighbors}
            for u in neighbors:
                for w in neighbors:
                    if u == w: continue
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        comp_adj[u].add(w)
                        comp_adj[w].add(u)
                        
            # Color comp_adj
            color_classes = dsatur_coloring(neighbors, comp_adj)
            
            for S in color_classes:
                if not S: continue
                # S is a clique in the available graph
                # Create clique {v} U S
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Mark edges in S as used
                S_sorted = sorted(S)
                for i in range(len(S_sorted)):
                    for j in range(i+1, len(S_sorted)):
                        e = (S_sorted[i], S_sorted[j])
                        used_edges.add(e)
                        
        # Now cover remaining edges in C
        # Remaining edges are those in E_C not in used_edges
        remaining_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                e = (u, w)
                if e in edges and e not in used_edges:
                    remaining_edges.add(e)
                    
        if remaining_edges:
            # Vertices involved in remaining edges
            rem_vertices = set()
            for u, w in remaining_edges:
                rem_vertices.add(u)
                rem_vertices.add(w)

            # Build graph of remaining edges
            rem_adj = {v: set() for v in rem_vertices}
            for u, w in remaining_edges:
                rem_adj[u].add(w)
                rem_adj[w].add(u)

            # We need to cover all edges in this graph with cliques.
            # This is the Clique Cover problem on rem_adj.
            # Equivalent to Coloring the complement of rem_adj.

            comp_rem_adj = {v: set() for v in rem_vertices}
            for v in rem_vertices:
                for w in rem_vertices:
                    if v == w:
                        continue
                    if w not in rem_adj[v]:
                        comp_rem_adj[v].add(w)
                        comp_rem_adj[w].add(v)

            color_classes = dsatur_coloring(rem_vertices, comp_rem_adj)

            # Polish: try to merge color classes to reduce clique count
            if time.time() < DEADLINE:
                polished = polish_clique_cover(rem_vertices, comp_rem_adj, color_classes, DEADLINE)
                color_classes = polished

            for S in color_classes:
                if len(S) >= 2:
                    cliques.append(sorted(S))
                elif len(S) == 1:
                    # Should not happen since every vertex in rem_vertices has degree >= 1
                    pass
                    
        # Cleanup and validation
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)

        return cleaned

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Fallback: cover all edges as 2-cliques (valid but not minimal)
            edges = instance(*spec)
            cliques = [[u, v] for u, v in sorted(edges)]
            result.append(cliques)
            continue
        cliques = partition_cliques(*spec)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
