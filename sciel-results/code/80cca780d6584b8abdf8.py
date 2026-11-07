"""Clique partition of split graphs using optimized partitioning strategy."""
import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: 1 clique for C, 1 per cross edge
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        # Build adjacency for I-vertices
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # For each I-vertex, find its neighbors in C
        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh

        # Strategy:
        # 1. Process I-vertices in order of decreasing degree in C.
        # 2. For each I-vertex v, greedily pack its neighbors into the largest possible
        #    clique that doesn't conflict with previously used C-C edges.
        # 3. Cover remaining C-C edges by coloring the graph of used C-C edges.
        
        used_cc = set()
        cliques = []
        
        # Sort I-vertices by degree descending to pack larger cliques first
        I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
        
        for v in I_verts:
            neigh = list(N_C[v])
            d = len(neigh)
            if d == 0:
                continue
            
            # Greedy packing for this vertex
            remaining_neigh = list(neigh)
            groups = []
            
            while remaining_neigh:
                # Start a new group with the first available neighbor
                group = [remaining_neigh.pop(0)]
                
                # Try to add other neighbors to this group if they don't conflict
                # A conflict exists if the C-C edge between two neighbors in the group
                # is already used by a previous Type B clique.
                i = 0
                while i < len(remaining_neigh):
                    cand = remaining_neigh[i]
                    ok = True
                    for g in group:
                        e = (cand, g) if cand < g else (g, cand)
                        if e in used_cc:
                            ok = False
                            break
                    if ok:
                        group.append(cand)
                        remaining_neigh.pop(i)
                    else:
                        i += 1
                
                groups.append(group)
            
            # Create cliques for the groups
            for group in groups:
                if len(group) == 1:
                    cliques.append([v, group[0]])
                else:
                    cliques.append([v] + group)
                    # Mark C-C edges within this group as used
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            a, b = group[i], group[j]
                            e = (a, b) if a < b else (b, a)
                            used_cc.add(e)
        
        # Cover remaining C-C edges
        # The remaining C-C edges are those NOT in used_cc.
        # We need to cover them with cliques.
        # This is equivalent to finding a clique cover of the graph G_rem where V=G_rem=range(k)
        # and E(G_rem) = { (u,v) in K_k | (u,v) not in used_cc }
        # The minimum number of cliques to cover G_rem is the chromatic number of its complement.
        # The complement of G_rem is the graph H where E(H) = used_cc.
        # So we need to color the graph H.
        
        # Build adjacency for H (used_cc edges)
        H_adj = [set() for _ in range(k)]
        for (a, b) in used_cc:
            H_adj[a].add(b)
            H_adj[b].add(a)
        
        # Greedy coloring of H
        # Vertices in H are 0..k-1
        # We want to partition V into independent sets of H (which are cliques in G_rem)
        # Actually, an independent set in H is a set of vertices with no edges in H.
        # If u, v are in an independent set of H, then (u,v) is NOT in used_cc.
        # So (u,v) IS in G_rem.
        # So an independent set in H is a clique in G_rem.
        # Correct.
        
        uncolored = set(range(k))
        color_count = 0
        # Order vertices by degree in H descending for better coloring
        order = sorted(range(k), key=lambda x: len(H_adj[x]), reverse=True)
        # But we need to pick from uncolored.
        # Let's just iterate and pick the one with highest degree in remaining uncolored H subgraph.
        
        while uncolored:
            # Find vertex with max degree in H induced by uncolored
            best_v = None
            best_deg = -1
            for v in uncolored:
                deg = len(H_adj[v] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            color_class = [best_v]
            uncolored.discard(best_v)
            
            # Try to add other vertices to this color class (independent set in H)
            # A vertex u can be added if it has no edge to any vertex in color_class in H.
            # i.e., u is not in H_adj[c] for any c in color_class.
            # To make this efficient, we can maintain a set of candidates.
            # But k is small (<=30), so O(k^2) is fine.
            
            candidates = list(uncolored)
            # Sort candidates by degree in H induced by uncolored (ascending?) 
            # Or just iterate.
            for u in candidates:
                if u not in uncolored:
                    continue
                # Check if u is compatible with all in color_class
                compatible = True
                for c in color_class:
                    if u in H_adj[c]:
                        compatible = False
                        break
                if compatible:
                    color_class.append(u)
                    uncolored.discard(u)
            
            # Add this clique (color class) to our cover
            if len(color_class) >= 2:
                cliques.append(sorted(color_class))
            # If len(color_class) == 1, it's a single vertex, which is a trivial clique, 
            # but the problem requires cliques to have at least 2 vertices.
            # However, if a vertex is isolated in G_rem (i.e., connected to all others in H),
            # it doesn't need to be in a C-only clique because it has no remaining C-C edges to cover.
            # Wait, the requirement is "every edge must lie in exactly one listed clique".
            # If a vertex has no remaining C-C edges, it doesn't need to be in any C-only clique.
            # So we should only add cliques with size >= 2.
            # The loop above only adds if len(color_class) >= 2. This is correct.
            
            color_count += 1

        results.append(cliques)

    # Final validation and type conversion
    for i in range(len(results)):
        new_list = []
        for clique in results[i]:
            new_clique = [int(x) for x in clique]
            new_list.append(new_clique)
        results[i] = new_list
    return results
