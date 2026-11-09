"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import math

    t0 = time.time()
    DEADLINE = t0 + 130.0

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

        # Helper to get all C-C edges within a set of vertices
        def get_cc_edges(vertices):
            verts = sorted(vertices)
            res = set()
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    res.add((verts[i], verts[j]))
            return res

        # Sort I vertices by size of their neighborhood in C (ascending)
        sorted_I = sorted(range(k, n), key=lambda v: len(N[v]))

        for v in sorted_I:
            nbrs = N[v]
            if not nbrs:
                continue

            remaining_nbrs = set(nbrs)

            while remaining_nbrs:
                # Find a maximal clique in the "available" graph on remaining_nbrs
                # where edge (u,w) exists if (u,w) not in covered_cc
                # Use greedy with degree ordering for better results
                
                # Compute degree in available graph for each vertex in remaining_nbrs
                degrees = {}
                for u in remaining_nbrs:
                    deg = 0
                    for w in remaining_nbrs:
                        if u == w:
                            continue
                        e = (min(u, w), max(u, w))
                        if e not in covered_cc:
                            deg += 1
                    degrees[u] = deg
                
                # Greedy: pick vertex with highest degree, then iteratively add
                # vertices connected to all current members
                candidates = sorted(remaining_nbrs, key=lambda x: -degrees[x])
                
                if not candidates:
                    break
                
                start = candidates[0]
                S = {start}
                
                # Iteratively add vertices that connect to all in S
                changed = True
                while changed:
                    changed = False
                    for u in list(remaining_nbrs - S):
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
        remaining_edges = []
        for u in range(k):
            for u2 in range(u + 1, k):
                if (u, u2) in edges and (u, u2) not in covered_cc:
                    remaining_edges.append((u, u2))

        # Try to merge some remaining edges into larger cliques
        # For each remaining edge, try to extend it into a larger clique
        used_edges = set()
        for e in remaining_edges:
            if e in used_edges:
                continue
            u, v = e
            # Try to find a maximal clique containing this edge
            # Start with {u, v} and try to add other vertices
            clique_set = {u, v}
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w in clique_set:
                        continue
                    # Check if w connects to all in clique_set and all those edges are uncovered
                    can_add = True
                    for x in clique_set:
                        edge = (min(w, x), max(w, x))
                        if edge not in edges or edge in covered_cc:
                            can_add = False
                            break
                    if can_add:
                        clique_set.add(w)
                        changed = True
            
            # If we found a larger clique, use it
            if len(clique_set) > 2:
                clique = sorted(clique_set)
                cliques.append(clique)
                # Mark all edges in this clique as covered
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        covered_cc.add((clique[i], clique[j]))
                # Mark these edges as used
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        used_edges.add((clique[i], clique[j]))
            else:
                cliques.append(list(e))
                used_edges.add(e)
                covered_cc.add(e)

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
