"""Clique partition of split graphs using global matching-based pair packing."""
import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Separate C-edges and cross-edges
        c_edges = []
        cross_edges = []
        for (u, v) in edges:
            if u < k and v < k:
                c_edges.append((u, v))
            else:
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_edges.append((c, i))

        c_edge_set = set(c_edges)
        cliques = []
        covered_c = set()
        
        # Build adjacency: which I-vertices are adjacent to which C-vertices
        # c_to_i[c] = set of I-vertices adjacent to C-vertex c
        c_to_i = defaultdict(set)
        i_to_c = defaultdict(set)
        for (c, i) in cross_edges:
            c_to_i[c].add(i)
            i_to_c[i].add(c)
        
        # Global matching: assign C-edges to I-vertices
        # For each C-edge (u,v), find an I-vertex i adjacent to both u and v
        # We want to maximize the number of such assignments (matching)
        # This is a bipartite matching: left = C-edges, right = I-vertices
        # Edge exists if I-vertex is adjacent to both endpoints of C-edge
        
        # Build bipartite graph for matching
        # Left side: C-edges indexed 0..len(c_edges)-1
        # Right side: I-vertices (those in I that have cross edges)
        
        # For efficiency, use Hopcroft-Karp style or simple augmenting path
        # Given the sizes (k up to 30, so C-edges up to 435), simple approach works
        
        # Build adjacency list for bipartite matching
        # left_adj[ce_idx] = list of I-vertices that can cover this C-edge
        left_adj = []
        for idx, (u, v) in enumerate(c_edges):
            common = c_to_i[u] & c_to_i[v]
            left_adj.append(sorted(common))
        
        # Simple augmenting path matching
        # match_left[ce_idx] = i_vertex or -1
        # match_right[i_vertex] = ce_idx or -1
        num_left = len(c_edges)
        i_vertices = sorted(i_to_c.keys())
        i_to_idx = {iv: idx for idx, iv in enumerate(i_vertices)}
        num_right = len(i_vertices)
        
        match_left = [-1] * num_left
        match_right = [-1] * num_right
        
        def try_augment(l, visited):
            for r in left_adj[l]:
                r_idx = i_to_idx[r]
                if r_idx in visited:
                    continue
                visited.add(r_idx)
                if match_right[r_idx] == -1 or try_augment(match_right[r_idx], visited):
                    match_left[l] = r
                    match_right[r_idx] = l
                    return True
            return False
        
        # Find maximum matching
        matching_size = 0
        for l in range(num_left):
            visited = set()
            if try_augment(l, visited):
                matching_size += 1
        
        # Now assign matched C-edges to I-vertices
        # For each matched pair (ce_idx -> i_vertex), create clique {i, u, v}
        # and mark those cross-edges as covered
        
        # Track which cross-edges are covered by 3-cliques
        # cross_covered[i] = set of C-vertices whose cross-edge to i is in a 3-clique
        cross_covered = defaultdict(set)
        
        for ce_idx, i in enumerate(match_left):
            if i == -1:
                continue
            u, v = c_edges[ce_idx]
            cliques.append(sorted([i, u, v]))
            covered_c.add((min(u, v), max(u, v)))
            cross_covered[i].add(u)
            cross_covered[i].add(v)
        
        # Phase 2: Handle remaining cross-edges as 2-cliques
        for i in i_to_c:
            for c in sorted(i_to_c[i]):
                if c not in cross_covered[i]:
                    cliques.append(sorted([i, c]))
        
        # Phase 3: Handle remaining C-edges with greedy maximal clique packing
        uncovered_c = set(e for e in c_edges if e not in covered_c)
        
        while uncovered_c:
            # Find vertex with max degree in uncovered_c
            deg = {}
            for (u, v) in uncovered_c:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break
                
            # Pick vertex with max degree
            u_max = max(deg.keys(), key=lambda x: deg[x])
            
            # Build maximal clique starting with u_max
            S_c = {u_max}
            
            # Get candidates: neighbors of u_max in uncovered_c
            candidates = set()
            for (u, v) in uncovered_c:
                if u == u_max:
                    candidates.add(v)
                elif v == u_max:
                    candidates.add(u)
            
            # Greedily add candidates
            changed = True
            while changed:
                changed = False
                for c in sorted(candidates):
                    # Check if c is connected to all in S_c via uncovered edges
                    ok = True
                    for v in S_c:
                        if v == c:
                            continue
                        e = (min(c, v), max(c, v))
                        if e not in uncovered_c:
                            ok = False
                            break
                    if ok:
                        S_c.add(c)
                        candidates.discard(c)
                        # Remove edges from c to S_c
                        for v in S_c:
                            if v == c:
                                continue
                            e = (min(c, v), max(c, v))
                            uncovered_c.discard(e)
                        changed = True
                        break  # Restart with updated S_c
            
            # Create clique
            clique = sorted(list(S_c))
            if len(clique) >= 2:
                cliques.append(clique)
            
            # Safety: if we made no progress, break
            if len(S_c) == 1:
                break
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback: trivial partition (all edges as cliques)
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Validate and convert types
        final_cliques = []
        for c in cliques:
            c_ints = list(set(int(x) for x in c))
            if len(c_ints) >= 2:
                final_cliques.append(sorted(c_ints))
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
