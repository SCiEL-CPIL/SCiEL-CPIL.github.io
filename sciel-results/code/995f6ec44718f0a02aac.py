"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # The clique C = {0, ..., k-1} is a maximal clique.
        # Remove it and its incident edges.
        remaining_edges = []
        for u, v in edges:
            if u < k and v < k:
                continue
            remaining_edges.append((u, v))

        # Now we need to cover remaining_edges with cliques of size >= 2.
        # Since the graph is split, any clique containing vertices from the independent set
        # can have at most one such vertex (because independent set vertices are not adjacent).
        # So cliques are either:
        #   - within the clique part (but we removed the big one, so none left if k is the whole clique)
        #   - one clique vertex + one independent vertex (edge)
        #   - one clique vertex + multiple independent vertices? No, independent vertices not adjacent.
        # So every remaining edge is between a clique vertex u < k and an independent vertex v >= k.
        # A clique in the remaining graph can contain at most one independent vertex.
        # Thus each clique is either a single edge (u, v) or... wait, can a clique have two clique vertices and one independent?
        # Yes! If u, v < k and u~v (they are, since clique), and w >= k with u~w and v~w, then {u,v,w} is a clique.
        # So cliques can be:
        #   - {u, v} where u,v < k (but we removed all such edges, so these would be redundant)
        #   - {u, w} where u < k, w >= k (single cross edge)
        #   - {u, v, w} where u,v < k, w >= k, and u~w, v~w (triangle)
        #   - larger: {u1, u2, ..., um, w} where all ui < k, w >= k, and all ui ~ w.
        #
        # So the problem reduces to: for each independent vertex w, let S_w = {u < k : u~w}.
        # We need to cover all edges (u, w) for u in S_w with cliques that include w and a subset of S_w.
        # Each clique containing w covers a subset of S_w (all pairwise connected, which they are since clique).
        # We want to minimize total number of cliques.
        #
        # This is equivalent to: for each w, partition S_w into as few subsets as possible, where each subset
        # forms a clique with w. Since all u in S_w are mutually adjacent, any subset works.
        # So for each w, we just need to cover S_w with cliques. Each clique can cover any subset of S_w.
        # To minimize count, we'd use one clique per w if |S_w| >= 1? No, we need each clique to have >= 2 vertices.
        # A clique {w} alone is not allowed. So each clique must include w and at least one u.
        # So for w, if |S_w| = s, we need at least ceil(s/1) = s cliques? No, one clique can include w and ALL of S_w.
        # So one clique per w (if s >= 1) covers all edges incident to w.
        # But wait: can we share cliques between different w's? No, because independent vertices are not adjacent.
        # So each clique contains at most one independent vertex.
        # Therefore, the minimum number of cliques to cover all cross edges is:
        #   sum over w >= k of (1 if |S_w| >= 1 else 0)
        # But we also need to check: does this cover all edges? Yes, each edge (u,w) is in the clique for w.
        # And each clique has >= 2 vertices (w and at least one u).
        #
        # However, we must also ensure that every edge is in EXACTLY one clique.
        # If we make one clique per w containing all of S_w, then edge (u,w) is only in w's clique. Good.
        #
        # But can we do better? What if we split S_w into multiple cliques? That would increase the count.
        # So one clique per w is optimal for the cross edges.
        #
        # Total cliques = 1 (for the big clique {0,...,k-1}) + number of w with |S_w| >= 1.
        
        # Count independent vertices with at least one neighbor in the clique
        cross_cliques = []
        for w in range(k, n):
            neighbors = sorted(adj[w] & set(range(k)))
            if neighbors:
                cross_cliques.append([w] + neighbors)
        
        all_cliques = [[list(range(k))] if k >= 2 else []] + cross_cliques
        # Filter out any clique with < 2 vertices
        all_cliques = [c for c in all_cliques if len(c) >= 2]
        
        return all_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = build_clique_partition(n, k, p, seed)
        # Convert to ensure pure Python types
        cliques = [list(int(v) for v in clique) for clique in cliques]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
