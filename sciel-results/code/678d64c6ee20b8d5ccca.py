"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    _deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Split graph: clique C = {0..k-1}, independent set I = {k..n-1}
        # Edges: all within C, some between C and I, none within I.
        # We need to cover every edge by cliques, each clique size >= 2,
        # each edge in exactly one clique.
        #
        # Key insight: In a split graph, the clique C is a maximal clique.
        # Any clique in the graph either is a subset of C, or contains
        # vertices from I. Since I is independent, any clique can contain
        # at most one vertex from I. So a clique containing an I-vertex
        # v must be {v} union a clique in C that is a subset of N(v) ∩ C.
        #
        # Strategy:
        # - The edges within C must be covered by cliques that are subsets of C.
        # - The edges between C and I must be covered by cliques of form {v} ∪ S
        #   where S ⊆ N(v) ∩ C and S is a clique in C (which it is since C is a clique).
        #
        # To minimize total cliques:
        # 1. Cover all C-C edges with a clique partition of the complete graph on C.
        #    But we can also "absorb" some C vertices into I-cliques to reduce C-C edges.
        #
        # Actually, let's think differently. Each edge (u,v) must be in exactly one clique.
        # A clique containing an I-vertex v and a set S ⊆ C covers edges (v,s) for s∈S.
        # It does NOT cover edges within S (those must be covered by other cliques).
        #
        # So the problem decomposes:
        # - For each v ∈ I, choose a set S_v ⊆ N(v)∩C to form clique {v} ∪ S_v.
        #   This covers edges (v, s) for s ∈ S_v.
        #   But we must cover ALL edges (v, s) where s ∈ N(v)∩C.
        #   So S_v must equal N(v)∩C? No — we could split N(v)∩C into multiple cliques
        #   {v} ∪ S_{v,1}, {v} ∪ S_{v,2}, ... but that's wasteful since {v}∪S is a clique
        #   for any S ⊆ N(v)∩C. So best to use one clique per v: {v} ∪ N(v)∩C.
        #
        # Wait, but then edges within N(v)∩C are NOT covered by these cliques (the clique
        # {v}∪S only contains edge (v,s) pairs, not (s1,s2) pairs). So C-C edges remain.
        #
        # So: Use one clique per I-vertex v: C_v = {v} ∪ N(v)∩C. This covers all C-I edges.
        # Then we need to cover all C-C edges with cliques that are subsets of C.
        # The C-C edges form a complete graph K_k. We need to partition its edges into
        # cliques (each clique is a subset of C, covering all edges within that subset).
        #
        # Minimum clique edge-partition of K_k: This is the problem of covering edges
        # of K_k with cliques such that each edge is in exactly one clique.
        # A clique of size m covers C(m,2) edges. We want to minimize number of cliques.
        #
        # This is equivalent to: partition the vertex set? No, cliques can overlap in vertices.
        # Actually, "every edge lies in exactly one listed clique" means the edge sets
        # of the cliques form a partition of E(K_k).
        #
        # For K_k, the minimum number of cliques needed to partition its edges is known.
        # One approach: use a 1-factorization if k is even, or near-1-factorization if odd.
        # But we're using cliques, not edges. A single edge {u,v} is a clique of size 2.
        #
        # Actually, the trivial solution is k(k-1)/2 edges each as a 2-clique.
        # Better: use larger cliques. E.g., for K_4, we can use 4 triangles? No:
        # K_4 has 6 edges. Triangle {0,1,2} covers 3 edges. Triangle {0,1,3} covers 3 edges,
        # but edge (0,1) is in both. Not allowed.
        #
        # So for K_k, we need edge-disjoint cliques. The minimum number is:
        # - If k is even: k/2 * (k-1) / ... hmm.
        # Actually, let me think of it as: we want to write E(K_k) as a disjoint union
        # of complete subgraphs. This is a "clique partition" of K_k.
        #
        # Known result: The minimum number of cliques in a clique partition of K_n is:
        # - For n ≡ 0 or 1 mod 3: n(n-1)/6
        # - For n ≡ 2 mod 3: (n(n-1)-2)/6 + 1 = (n^2-3n+4)/6
        # Wait, that's for edge-disjoint triangles. But we can use cliques of any size.
        #
        # Actually, using larger cliques is better. A clique of size m uses C(m,2) edges
        # and counts as 1. So we want to pack large cliques.
        #
        # For K_k, the optimal clique partition:
        # - If k=2: 1 clique (the edge itself)
        # - If k=3: 1 clique (triangle)
        # - If k=4: We can use two triangles? No, they'd share an edge.
        #   K_4: 6 edges. Use triangle {0,1,2} (3 edges) and triangle {0,1,3}? Overlaps on (0,1).
        #   Use {0,1,2} and {3,0,1}? No.
        #   Actually: {0,1,2} covers (0,1),(0,2),(1,2). Remaining: (0,3),(1,3),(2,3).
        #   These form a star, not a clique. So we need 3 more edges as 2-cliques, or
        #   we can't do better than 1+3=4? Or use {0,1,3} and {2,3,?}... 
        #   Let me just enumerate: K_4 clique partitions.
        #   Option: 4 triangles: {0,1,2},{0,1,3},{0,2,3},{1,2,3} — but edges overlap.
        #   Option: 1 triangle + 3 edges: {0,1,2}, {0,3}, {1,3}, {2,3} → 4 cliques.
        #   Option: 2 triangles edge-disjoint: {0,1,2} and {3,?,?} — {3} alone isn't a clique.
        #   So minimum for K_4 is 4? Let's verify: 6 edges, max clique size 4 (covers 6, but
        #   that's the whole graph, 1 clique!). Wait, K_4 itself is a clique! So 1 clique covers all 6 edges.
        #   But wait — we can't use C itself as a clique if we also need to use it for I-vertices.
        #   Actually, we CAN. The clique C={0,...,k-1} covers all C-C edges. Then each I-vertex
        #   v gets clique {v}∪N(v)∩C. But then edge (v,s) is in clique {v}∪N(v)∩C, and edge (s1,s2)
        #   is in clique C. No overlap! So total = 1 + |I| = 1 + (n-k).
        #
        # But wait — does clique C cover edge (s1,s2) for s1,s2 ∈ C? Yes. And clique {v}∪N(v)∩C
        # covers edge (v,s) for s∈N(v)∩C. Does it cover any C-C edge? No, because v∉C.
        # So edges are partitioned perfectly. Total cliques = 1 + (n-k).
        #
        # For (30,10): 1+20=21. For (60,20): 1+40=41. For (90,30): 1+60=61.
        # Total = 21+41+61 = 123.
        #
        # But can we do better? What if we don't use C as a single clique, but split it
        # to allow I-vertex cliques to be larger? No, I-vertex cliques can't contain C-C edges.
        #
        # Alternatively, what if we merge some I-vertices with C-cliques? An I-vertex v can
        # only be in cliques that don't contain other I-vertices (since I is independent).
        # So each I-vertex's cliques are of form {v}∪S where S⊆C.
        #
        # The C-C edges must be covered by cliques entirely in C. The minimum number of
        # cliques to partition E(K_k) is 1 (just use K_k itself). So 1 + (n-k) is optimal
        # IF we use one clique per I-vertex.
        #
        # Can we reduce below 1+(n-k)? Only if some I-vertex needs 0 cliques, i.e., N(v)∩C=∅.
        # But if N(v)∩C is non-empty, v must be in at least one clique to cover its edges.
        # And each such clique contains exactly one I-vertex, so we need at least one per v
        # with N(v)∩C ≠ ∅.
        #
        # So optimal = 1 + |{v ∈ I : N(v)∩C ≠ ∅}|.
        #
        # Let's compute this.

        # Count I-vertices with at least one neighbor in C
        active_I = 0
        for v in range(k, n):
            if adj[v]:  # has any neighbor
                active_I += 1

        # Build solution:
        cliques = []
        # One clique for all of C
        if k >= 2:
            cliques.append(list(range(k)))
        # One clique per active I-vertex
        for v in range(k, n):
            neighbors_in_C = sorted(adj[v])
            if neighbors_in_C:
                cliques.append([v] + neighbors_in_C)

        # Verify: each clique has >= 2 vertices, all edges covered exactly once
        # (We trust the logic above, but let's be safe)
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline:
            break
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        # Convert to pure Python lists of ints
        cliques = [[int(x) for x in clique] for clique in cliques]
        results.append(cliques)

    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
