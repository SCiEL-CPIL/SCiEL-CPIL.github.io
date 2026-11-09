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
        
        # Helper to get all C-C edges within a set of vertices
        def get_cc_edges(vertices):
            verts = sorted(vertices)
            res = set()
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    res.add((verts[i], verts[j]))
            return res

        # Process vertices in I in order of increasing neighborhood size
        # This helps because vertices with smaller neighborhoods can be processed first
        # and their C-C edges will be covered, making it easier to find large cliques
        # for vertices with larger neighborhoods
        i_vertices = sorted(range(k, n), key=lambda v: len(N[v]))
        
        for v in i_vertices:
            nbrs = N[v]
            if not nbrs:
                continue
            
            # We need to cover all cross edges (u, v) for u in nbrs.
            # This requires u to be in a clique with v.
            # We partition nbrs into subsets S such that the C-C edges within S
            # are not already covered.
            
            remaining_nbrs = set(nbrs)
            
            while remaining_nbrs:
                # Find a maximal subset S of remaining_nbrs such that
                # all C-C edges within S are not in covered_cc.
                # This is equivalent to finding a maximal clique in the graph
                # induced by remaining_nbrs where edges exist if NOT in covered_cc.
                
                # Use a greedy approach: start with the vertex that has the most
                # available connections (fewest covered edges to other remaining vertices)
                best_start = None
                best_score = -1
                for u in remaining_nbrs:
                    # Count how many vertices in remaining_nbrs are "available" to u
                    # (i.e., the edge (u, w) is not in covered_cc)
                    available_count = 0
                    for w in remaining_nbrs:
                        if u == w:
                            continue
                        e = (min(u, w), max(u, w))
                        if e not in covered_cc:
                            available_count += 1
                    if available_count > best_score:
                        best_score = available_count
                        best_start = u
                
                S = {best_start}
                
                # Try to add vertices to S one by one
                changed = True
                while changed:
                    changed = False
                    candidates = list(remaining_nbrs - S)
                    # Sort candidates by how many connections they have to S
                    candidates.sort(key=lambda u: -sum(
                        1 for w in S 
                        if (min(u, w), max(u, w)) not in covered_cc
                    ))
                    for u in candidates:
                        # Check if u is connected to all in S in G_avail
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
            for u2 in range(u+1, k):
                if (u, u2) in edges and (u, u2) not in covered_cc:
                    remaining_edges.append((u, u2))
        
        # Cover remaining edges with 2-cliques for simplicity and correctness
        for e in remaining_edges:
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
