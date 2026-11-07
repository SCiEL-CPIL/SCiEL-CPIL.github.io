import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # N(v) for v in I
        N = {}
        for v in range(k, n):
            N[v] = set(neighbors[v])
        
        # For each K-edge (u,w), which v's have both u,w in N(v)?
        # We'll assign each K-edge to at most one v.
        # Strategy: for each v, S_v ⊆ N(v), with |S_v ∩ S_w| ≤ 1 for all v≠w.
        # Greedy: process v's in order of |N(v)| descending.
        
        # Sort v's by degree descending
        v_list = list(range(k, n))
        v_list.sort(key=lambda v: -len(N[v]))
        
        S = {}  # S[v] = set of K-vertices assigned to v's clique
        # Track for each K-vertex u, which v's have u in their S, and how many common vertices
        # Actually, we need: for each pair of K-vertices (u,w), at most one v has both in S_v.
        # Equivalent: for each v, S_v can contain u,w only if no earlier v' has both u,w in S_{v'}.
        
        # Track: for each K-edge (u,w), has it been "used" by some v's S?
        edge_used_by = {}  # (u,w) -> v that has both in S_v
        
        for v in v_list:
            S[v] = set()
            for u in N[v]:
                # Can we add u to S[v]?
                # Check: for all w already in S[v], edge (u,w) must not be used by another v.
                can_add = True
                for w in S[v]:
                    e = (min(u, w), max(u, w))
                    if e in edge_used_by and edge_used_by[e] != v:
                        can_add = False
                        break
                if can_add:
                    S[v].add(u)
                    # Mark all edges from u to existing S[v] members as used by v
                    for w in S[v]:
                        if w != u:
                            e = (min(u, w), max(u, w))
                            if e not in edge_used_by:
                                edge_used_by[e] = v
        
        # Now form cliques
        cliques = []
        
        # For each v, if |S[v]| >= 2, one clique S[v] ∪ {v}
        # If |S[v]| == 1, one 2-clique [u, v]
        # If |S[v]| == 0, no clique (but v has no K-neighbors, so no K-I edges)
        for v in range(k, n):
            if len(S[v]) >= 2:
                cliques.append(sorted(list(S[v])) + [v])
            elif len(S[v]) == 1:
                u = next(iter(S[v]))
                cliques.append([u, v])
            # If 0, nothing
        
        # Now cover remaining K-edges (not in edge_used_by) with K-cliques
        all_k_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_k_edges.add((u, w))
        
        remaining_k_edges = all_k_edges - set(edge_used_by.keys())
        
        # Greedy clique cover for remaining K-edges
        # Since K is a clique, any subset is a clique. So we just need to partition the "active" vertices.
        # Actually, we need to cover all remaining edges. Since K is complete, we can just put all 
        # vertices that have remaining edges into one big clique? No, that would cover edges already covered.
        # We need: each remaining edge in exactly one new clique.
        # Since K is a clique, we can group vertices arbitrarily. But we must not include edges already covered.
        # An edge (u,w) is already covered if it's in edge_used_by. So in a new K-clique T, all edges within T
        # must be remaining (not in edge_used_by). But since K is complete, edges within T are all K-edges.
        # So T can only contain vertices u,w where (u,w) is remaining.
        # This means: find a clique in the graph of remaining edges. Since the remaining graph is a subgraph of K,
        # and K is complete, the remaining graph has edge (u,w) iff (u,w) not in edge_used_by.
        # We need min clique cover of this graph.
        
        # Greedy: repeatedly find a maximal clique in remaining graph, remove its edges.
        remaining = set(remaining_k_edges)
        while remaining:
            # Pick an edge, build maximal clique
            e = next(iter(remaining))
            u, w = e
            clique = {u, w}
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x not in clique:
                        if all((min(x, y), max(x, y)) in remaining for y in clique):
                            clique.add(x)
                            changed = True
                            break
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for a, b in combinations(clique, 2):
                    remaining.discard((min(a, b), max(a, b)))
        
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
