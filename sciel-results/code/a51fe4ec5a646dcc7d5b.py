"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

DEADLINE = 110.0  # safety margin under 120s

def construct():
    t0 = time.time()
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _is_clique(cl, adj):
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if not adj[cl[i]][cl[j]]:
                    return False
        return True

    def _verify_and_fix(cliques, adj, n):
        # Verify every clique is a clique and every edge is covered exactly once
        edge_count = {}
        for cl in cliques:
            if len(cl) < 2:
                continue
            if not _is_clique(cl, adj):
                return None
            for a, b in itertools.combinations(cl, 2):
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
        # Check every edge is covered exactly once
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    if edge_count.get((u, v), 0) != 1:
                        return None
                else:
                    if (u, v) in edge_count:
                        return None
        return cliques

    def _solve_split(n, k, p, seed):
        if time.time() - t0 > DEADLINE:
            return None
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True

        # Cross edges: between [0,k) and [k,n)
        cross_edges = [(u, v) for u, v in edges if u < k and v >= k]
        # Edges within K-part (always a clique)
        k_edges = [(u, v) for u, v in edges if u < k and v < k]

        # Build cross-adjacency: for each independent vertex v, which clique vertices it connects to
        # cross_adj[v] = bitmask of clique vertices connected to v
        cross_adj = {}
        for u, v in cross_edges:
            # u in [0,k), v in [k,n)
            cross_adj.setdefault(v, 0)
            cross_adj[v] |= (1 << u)
            # Also track from clique side
            # We'll build a graph on independent vertices: edge if they share a clique neighbor

        indep_verts = [v for v in range(k, n) if v in cross_adj]
        # Build graph on independent vertices: edge if they share at least one clique neighbor
        m = len(indep_verts)
        # Map independent vertex to index
        iv_index = {v: i for i, v in enumerate(indep_verts)}
        # For each pair of independent vertices, check if they share a clique neighbor
        # Two independent vertices v1, v2 can be in the same clique iff they share a common clique neighbor
        # (and the clique must be a clique, so all pairs in it must share)
        
        # Actually, for a set of independent vertices S to form a clique with some clique vertices C:
        # - Every pair in S must share at least one common neighbor in C
        # - Every vertex in S must be connected to every vertex in C
        # - C must be a subset of [0,k) (which is a clique)
        
        # The key insight: if we pick a set of clique vertices C, then the independent vertices that connect to ALL of C
        # can be added. So the problem is: find cliques in the "joint" graph.
        
        # Let me think differently. A clique in the split graph is either:
        # 1. A subset of [0,k) (the clique part) - these are "free" in the sense they don't use cross edges
        # 2. A set C ∪ S where C ⊆ [0,k), S ⊆ [k,n), C is non-empty (for cross edges to exist),
        #    S is non-empty, every s in S is connected to every c in C, and S is an independent set (always true)
        #    For S to be a clique with C: we need every pair (s1,s2) in S to... wait, s1 and s2 are both in the independent set,
        #    so they are NOT adjacent to each other. So S can have at most 1 vertex!
        
        # WAIT. This is a critical insight. In a split graph, the independent set vertices are NOT adjacent to each other.
        # So any clique can contain at most ONE independent vertex.
        # Therefore, every clique in the partition is either:
        #   - A subset of the clique part [0,k), OR
        #   - A clique vertex c plus an independent vertex v (where c~v), OR
        #   - A single independent vertex (but we need at least 2 vertices, so this is invalid)
        
        # So the cross edges must each be in a clique of the form {c, v} or {c1, c2, v} or {c, v1}... 
        # Wait, {c1, c2, v} is a clique if c1~c2 (yes, both in K) and c1~v and c2~v.
        # And {c, v1, v2} is NOT a clique because v1 and v2 are not adjacent.
        
        # So a clique containing independent vertices can have at most ONE independent vertex.
        # The clique is C ∪ {v} where C ⊆ [0,k) is a clique (always true) and v is connected to all of C.
        
        # So the problem reduces to:
        # - Each cross edge (c, v) must be in exactly one clique
        # - Each clique containing v is of the form C_v ∪ {v} where C_v ⊆ N(v) ∩ [0,k)
        # - Each clique vertex c can appear in multiple cliques (with different v's), but each edge (c,v) 
        #   must be in exactly one clique
        # - K-part edges (c1, c2) must be in exactly one clique
        # - A clique C ∪ {v} contains edges: all (c1,c2) for c1,c2 in C, and all (c,v) for c in C
        # - A pure K-clique C contains edges: all (c1,c2) for c1,c2 in C
        
        # So the problem is: partition all edges into cliques where:
        # - Each clique is either a subset of [0,k) or C ∪ {v} for some v in [k,n)
        # - K-part edges are covered exactly once
        # - Cross edges are covered exactly once
        
        # This is equivalent to: for each independent vertex v, choose a subset C_v ⊆ N(v) ∩ [0,k)
        # such that the union of all C_v (with multiplicity for edges) covers each cross edge exactly once,
        # and the K-part edges are covered by cliques that are subsets of [0,k).
        
        # Actually, let me re-think. The cliques that contain v are exactly one: C_v ∪ {v}.
        # The edges in this clique are: (c,v) for c in C_v, and (c1,c2) for c1,c2 in C_v.
        # The K-part edges not covered by any C_v ∪ {v} clique must be covered by pure K-cliques.
        
        # To minimize the total number of cliques, we want to maximize the "efficiency" of each clique.
        # A clique C ∪ {v} covers |C| cross edges and |C|(|C|-1)/2 K-edges in one clique.
        # A pure K-clique C covers |C|(|C|-1)/2 K-edges in one clique.
        
        # The K-edges are "free" in the sense that they're always there. The question is how to group them.
        # Since [0,k) is a clique, we can put ALL K-edges into one clique [0,k) if we want.
        # But that might not be optimal if we can "absorb" some K-edges into cross cliques.
        
        # Actually, the optimal strategy is:
        # 1. For each independent vertex v, assign it to a clique C_v ∪ {v} where C_v ⊆ N(v)
        # 2. The remaining K-edges (those not in any C_v) form a graph on [0,k) that we need to cover
        #    with cliques (which are just subsets of [0,k) since it's a clique)
        
        # To minimize total cliques:
        # - We want each C_v to be as large as possible (to cover more cross edges per clique)
        # - But C_v must be a subset of N(v), and the K-edges in C_v are "used up"
        # - The remaining K-edges need to be covered by additional cliques
        
        # Simplest approach: for each v, set C_v = N(v) ∩ [0,k) (all its neighbors in K)
        # Then the remaining K-edges are those (c1,c2) where c1 or c2 is not in any C_v that contains both.
        # Wait, no. A K-edge (c1,c2) is covered if there exists some v such that c1,c2 ∈ C_v.
        # With C_v = N(v), (c1,c2) is covered if both c1 and c2 are neighbors of v.
        
        # The remaining K-edges form a graph on [0,k) where (c1,c2) is an edge if NO independent vertex
        # is adjacent to both c1 and c2. We need to cover this residual graph with cliques.
        # Since [0,k) is a clique, any subset is a clique, so we just need to cover the residual edges.
        # The minimum number of cliques to cover a graph is the clique cover number, which is NP-hard in general,
        # but for our residual graph (which is a subgraph of a clique), it's just the minimum number of cliques
        # to cover the edges.
        
        # Actually, since the residual graph is a subgraph of a complete graph, the minimum clique cover
        # is just the chromatic number of the complement... no, that's not right either.
        
        # For a graph G, the minimum number of cliques to cover all edges is equal to the chromatic number
        # of the complement of G (each clique in G corresponds to an independent set in complement G,
        # and covering edges with cliques is the same as coloring the complement).
        
        # Wait, that's not quite right. Covering edges with cliques means each edge is in at least one clique.
        # This is the edge clique cover number, which is different from vertex clique cover.
        
        # For the residual graph on [0,k), since it's a subgraph of K_k, we can use a greedy approach:
        # repeatedly find the largest clique in the residual graph, remove its edges, repeat.
        
        # But actually, for the residual graph, the simplest valid solution is to use individual edges
        # (2-vertex cliques) for any residual edges. This gives at most k(k-1)/2 additional cliques.
        
        # A better approach: use a greedy maximum clique cover on the residual graph.
        
        # Let me implement the simple version first:
        # 1. For each v in [k,n), C_v = N(v) ∩ [0,k)
        # 2. Mark all K-edges covered by some C_v
        # 3. For remaining K-edges, use a greedy clique cover
        
        # But wait, there's a subtlety: if C_v is empty (v has no neighbors in K), then v has no cross edges,
        # so it doesn't need a clique.
        
        # Also, I need to make sure that the K-edges in C_v are not double-counted.
        # A K-edge (c1,c2) is covered by C_v if both c1,c2 ∈ C_v. It might be covered by multiple C_v's.
        # But in the partition, each edge must be in EXACTLY ONE clique. So I can't have (c1,c2) in two different C_v's.
        
        # This means I need to assign each K-edge to at most one C_v, and the C_v's must be such that
        # no K-edge is in two different C_v's. This is a constraint satisfaction problem.
        
        # Alternative simpler approach: don't use any K-edges in cross cliques. Instead, put all K-edges
        # in one big clique [0,k), and each cross edge (c,v) in its own 2-clique {c,v}.
        # Total cliques: 1 + |cross_edges|
        # This is valid and simple. Let me check if this is better than the seed.
        # Seed: |all_edges| = k(k-1)/2 + |cross_edges|
        # My approach: 1 + |cross_edges|
        # Improvement: k(k-1)/2 - 1
        
        # For (30,10): 45 - 1 = 44 improvement
        # For (60,20): 190 - 1 = 189 improvement  
        # For (90,30): 435 - 1 = 434 improvement
        
        # Can we do better? Yes, by grouping cross edges.
        # If two cross edges (c,v1) and (c,v2) share the same c, they can't be in the same clique
        # (because v1 and v2 are not adjacent). So each cross edge needs its own clique if we don't
        # use K-edges in cross cliques.
        
        # But if we allow K-edges in cross cliques, we can group: {c1,c2,v} covers (c1,v), (c2,v), (c1,c2).
        # The constraint is that (c1,c2) can only be in one such clique.
        
        # Let me try a greedy approach:
        # 1. Start with all edges uncovered
        # 2. Greedily find the largest possible clique, add it, remove its edges
        # 3. Repeat until all edges are covered
        
        # This is the standard greedy edge clique cover. For split graphs, this should work well.
        
        # Let me implement a greedy maximum clique cover:
        # - At each step, find the largest clique in the remaining edge graph
        # - Add it to the partition
        # - Remove all its edges from the remaining graph
        # - Repeat
        
        # To find the largest clique efficiently, I can use a greedy heuristic:
        # - For each vertex, compute its degree in the remaining graph
        # - Pick the vertex with highest degree, then greedily add its neighbors that form a clique
        
        # Actually, for split graphs, the structure is simple enough that I can be more clever.
        
        # Let me just implement a straightforward greedy:
        remaining = set(edges)
        cliques = []
        while remaining:
            if time.time() - t0 > DEADLINE:
                return None
            # Find a good clique from remaining edges
            # Pick an edge, then try to extend it to a maximum clique
            # For efficiency, pick the edge with the most "potential" (high degree endpoints)
            
            # Simple approach: pick the first edge, then try to extend
            # Better: pick the edge whose endpoints have the most common neighbors in remaining
            
            # For split graphs, let's use the structure:
            # A clique is either in K-part or C ∪ {v}
            # Try to find the largest C ∪ {v} where all edges are in remaining
            
            # For each independent vertex v, find the largest C ⊆ N(v) such that all edges in C∪{v} are in remaining
            # The edges in C∪{v} are: (c,v) for c in C, and (c1,c2) for c1,c2 in C
            # All must be in remaining.
            
            best_clique = None
            best_size = 0
            
            # Try cliques containing independent vertices
            for v in range(k, n):
                if time.time() - t0 > DEADLINE:
                    return None
                # Find neighbors of v in K that are connected to v in remaining
                candidates = []
                for c in range(k):
                    if (min(c,v), max(c,v)) in remaining:
                        candidates.append(c)
                if not candidates:
                    continue
                # Try to find the largest subset C of candidates such that all (c1,c2) for c1,c2 in C are in remaining
                # Greedy: start with all candidates, remove those that don't connect to others
                # Actually, since K is a clique, all (c1,c2) edges exist. The question is whether they're in remaining.
                # So I need C such that for all c1,c2 in C, (c1,c2) in remaining.
                
                # Greedy: sort candidates by "connectivity" and add one by one
                # For each candidate c, it can be added if (c, c') is in remaining for all c' already in C
                C = []
                for c in candidates:
                    can_add = True
                    for c2 in C:
                        if (min(c,c2), max(c,c2)) not in remaining:
                            can_add = False
                            break
                    if can_add:
                        C.append(c)
                if len(C) > best_size:
                    best_size = len(C)
                    best_clique = C + [v]
            
            # Also try pure K-cliques
            # Find the largest clique in the remaining K-part edges
            # K-part edges remaining: (c1,c2) in remaining where c1,c2 < k
            k_remaining = [(c1,c2) for (c1,c2) in remaining if c1 < k and c2 < k]
            if k_remaining:
                # Greedy: find a large clique in the K-part
                # Since it's a subgraph of K_k, use greedy coloring of complement or just greedy clique
                # Simple: pick a vertex with max degree, extend
                k_deg = {}
                for c1, c2 in k_remaining:
                    k_deg[c1] = k_deg.get(c1, 0) + 1
                    k_deg[c2] = k_deg.get(c2, 0) + 1
                if k_deg:
                    start = max(k_deg, key=lambda x: k_deg[x])
                    C = [start]
                    for c in range(k):
                        if c == start:
                            continue
                        can_add = True
                        for c2 in C:
                            if (min(c,c2), max(c,c2)) not in remaining:
                                can_add = False
                                break
                        if can_add:
                            C.append(c)
                    if len(C) > best_size:
                        best_size = len(C)
                        best_clique = C
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: use a single edge
                best_clique = list(next(iter(remaining)))
            
            cliques.append(best_clique)
            # Remove all edges in this clique from remaining
            for a, b in itertools.combinations(best_clique, 2):
                remaining.discard((min(a,b), max(a,b)))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() - t0 > DEADLINE:
            # Fallback: return seed-like answer for remaining instances
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[int(u), int(v)] for u, v in edges])
            continue
        cliques = _solve_split(*spec)
        if cliques is None:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [[int(u), int(v)] for u, v in edges]
        # Verify
        n, k, p, seed = spec
        adj = [[False] * n for _ in range(n)]
        for u, v in instance(n, k, p, seed):
            adj[u][v] = adj[v][u] = True
        fixed = _verify_and_fix(cliques, adj, n)
        if fixed is None:
            # Fallback to edge-by-edge
            edges = instance(n, k, p, seed)
            fixed = [[int(u), int(v)] for u, v in edges]
        results.append([[int(v) for v in cl] for cl in fixed])
    
    return results
# EVOLVE-BLOCK-END
