"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_partition(n, k, p, seed):
        # Generate the graph
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))

        # Build neighbor lists for independent set vertices
        # For each independent vertex v, neighbors are in [0, k)
        indep_neighbors = {}
        for v in range(k, n):
            neighbors = set()
            for (u, v2) in cross_edges:
                if v2 == v:
                    neighbors.add(u)
            indep_neighbors[v] = neighbors

        # Strategy: 
        # 1. Use one big clique for all clique-part vertices: [0, 1, ..., k-1]
        #    This covers all edges within the clique part.
        # 2. For each independent vertex v, we need to cover its cross-edges.
        #    Since v is independent, no two independent vertices can be in the same clique.
        #    So each clique containing v can only contain v and some subset of its neighbors in the clique part.
        #    To minimize the number of cliques, we want to pack as many edges as possible into each clique.
        #    A clique {v} ∪ S where S ⊆ neighbors(v) covers all edges (u, v) for u ∈ S.
        #    So for each v, we need to partition its neighbor set into as few subsets as possible?
        #    Wait, no. The edges incident to v are (u, v) for u ∈ neighbors(v).
        #    If we form a clique {v, u1, u2, ..., um}, it covers edges (v, u1), (v, u2), ..., (v, um).
        #    It also covers edges (ui, uj) for all i,j, but those are already covered by the big clique [0..k-1].
        #    BUT the rule says: "every edge must lie in exactly one listed clique."
        #    So if we include {v, u1, u2} and also [0..k-1], the edge (u1, u2) is in BOTH cliques. This is INVALID.
        
        # Therefore, we cannot have overlapping cliques on the clique-part vertices if those vertices form edges between them.
        # The only way to avoid overlap is:
        # - Either the big clique [0..k-1] is the ONLY clique containing any pair of clique-part vertices.
        # - OR we don't use a big clique, and instead cover all clique-part edges via other cliques.
        # But independent vertices can't be in the same clique. So any clique containing an independent vertex v
        # can contain at most one independent vertex.
        # If a clique contains v and some clique-part vertices S, it covers edges (v, u) for u in S.
        # It also covers edges (u1, u2) for u1, u2 in S.
        # To avoid double-covering (u1, u2), we must ensure that no other clique contains both u1 and u2.
        # This implies that the clique-part vertices must be partitioned among the cliques that contain independent vertices,
        # AND any clique-part vertices not used in such cliques must form their own cliques, but those cliques
        # cannot overlap with the S sets.
        
        # Actually, a simpler approach:
        # Since the clique part is a complete graph, any two clique-part vertices form an edge.
        # If we have two cliques C1 = {v1, u1, u2} and C2 = {v2, u2, u3}, then edge (u2, ?) 
        # Wait, u2 is in both C1 and C2. The edge (u2, u2) doesn't exist. But what about (u2, u1)?
        # (u1, u2) is in C1. Is it in C2? No, because u1 is not in C2.
        # What about (u2, u3)? It is in C2. Is it in C1? No, because u3 is not in C1.
        # So the edges within S1 (neighbors of v1 in the clique) are covered by C1.
        # The edges within S2 (neighbors of v2 in the clique) are covered by C2.
        # If S1 and S2 overlap, say u2 is in both, then edges (u2, u1) are in C1, (u2, u3) are in C2.
        # But what about edge (u1, u3)? It is not in C1 (u3 not in C1) and not in C2 (u1 not in C2).
        # So it must be covered by another clique.
        
        # This suggests that we should partition the clique-part vertices into disjoint sets S_v for each independent vertex v,
        # and possibly a remaining set R.
        # Then cliques are:
        # - For each v: {v} ∪ S_v (if S_v is non-empty, and |S_v| >= 1 so that the clique has size >= 2)
        #   Wait, if S_v has 1 element, say {u}, then {v, u} is a clique of size 2. That's fine.
        #   If S_v is empty, we can't form a clique with v alone (size 1). So v must be paired with at least one neighbor.
        #   But what if v has no neighbors? Then v is isolated. But the task says "every edge must lie in exactly one listed clique".
        #   If v has no edges, it doesn't need to be in any clique. But the task says "Each clique is a list of at least two distinct int vertices".
        #   It doesn't say every vertex must be in a clique. Only every EDGE must be in exactly one clique.
        #   So isolated vertices can be ignored.
        
        # So the plan:
        # 1. Assign each independent vertex v to a set S_v ⊆ neighbors(v).
        # 2. Ensure S_v are pairwise disjoint.
        # 3. Let R = [0, k) \ (union of S_v).
        # 4. The cliques are:
        #    - For each v with S_v non-empty: {v} ∪ S_v
        #    - For the remaining clique-part vertices R, we need to cover all edges within R.
        #      Since R is a clique, we can just use one clique R (if |R| >= 2).
        #      If |R| == 1, no edges to cover.
        #      If |R| == 0, no clique needed.
        # 5. But wait, what about edges between S_v and S_w for v != w?
        #    These edges are NOT covered by {v} ∪ S_v or {w} ∪ S_w.
        #    They are also not in R (since S_v and S_w are disjoint from R).
        #    So we have uncovered edges!
        
        # This approach fails because edges between different S_v sets are not covered.
        
        # Alternative approach:
        # Use only cliques of the form {v} ∪ S_v where S_v ⊆ neighbors(v), and ensure that the union of all S_v covers [0, k).
        # But then edges between S_v and S_w are still not covered.
        
        # Actually, the only way to cover edges between clique-part vertices is to have a clique that contains both.
        # Since independent vertices can't be together, the only cliques that can contain two clique-part vertices are:
        # - Cliques that contain no independent vertices (pure clique-part cliques)
        # - Cliques that contain exactly one independent vertex v and some clique-part vertices S_v.
        #   But such a clique covers edges within S_v, but not edges between S_v and other S_w.
        
        # Therefore, the simplest valid solution is:
        # 1. One big clique: [0, 1, ..., k-1]
        # 2. For each independent vertex v, for each neighbor u of v, a clique {v, u}.
        #    But wait, if we do this, then edge (u, w) for u, w in [0, k) is in the big clique.
        #    Is it in any other clique? No, because other cliques are {v, u} which only have one clique-part vertex.
        #    So this is valid!
        #    Number of cliques: 1 + (number of cross edges)
        
        # Can we do better?
        # Yes, if we can pack multiple cross edges into one clique.
        # For example, if v is connected to u1 and u2, we can use {v, u1, u2} instead of {v, u1} and {v, u2}.
        # But then edge (u1, u2) is in {v, u1, u2} AND in the big clique [0..k-1]. This is INVALID.
        
        # So we CANNOT use the big clique AND pack cross edges.
        
        # Alternative:
        # Don't use the big clique. Instead, cover all edges using cliques of the form {v} ∪ S_v.
        # But as noted, edges between S_v and S_w are not covered.
        
        # So we must either:
        # A) Use the big clique, and each cross edge is its own clique {v, u}. Total: 1 + E_cross.
        # B) Don't use the big clique, and find a way to cover all edges with cliques {v} ∪ S_v.
        #    This requires that for any two clique-part vertices u, w, there is some v such that u, w ∈ S_v.
        #    This is a covering design problem, which is hard.
        
        # Given the time constraint, let's go with approach A, which is simple and valid.
        # Total cliques: 1 + E_cross.
        
        # But wait, what if k=1? Then the big clique has size 1, which is invalid.
        # In that case, we don't use the big clique.
        # If k=1, there are no edges within the clique part. So we just need to cover cross edges.
        # Each cross edge (0, v) can be its own clique {0, v}.
        
        # Also, what if an independent vertex v has no neighbors? Then it doesn't need to be in any clique.
        
        # Let's implement approach A.
        
        cliques = []
        
        # Add the big clique if k >= 2
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Add cliques for each cross edge
        for (u, v) in cross_edges:
            cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(build_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
