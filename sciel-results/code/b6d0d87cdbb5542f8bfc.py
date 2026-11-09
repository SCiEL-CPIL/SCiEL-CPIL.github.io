import time
from random import Random

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_adj(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        return adj, edge_set

    def solve_one(n, k, p, seed, time_limit):
        adj, edge_set = build_adj(n, k, p, seed)
        C = list(range(k))
        C_set = set(C)
        I = list(range(k, n))
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))

        # For each I vertex, get its C-neighborhood
        Nv = {}
        for v in I:
            Nv[v] = adj[v] & C_set

        # Precompute: for each subset of C that is a clique, store it
        # We'll work with the induced subgraph on C (which is complete, so any subset is a clique)
        # Actually C is a clique, so any subset of C is a clique!
        # So partitioning Nv[v] into cliques of C is just partitioning into any groups.
        # But we want to minimize the number of groups while also covering internal edges.

        # Key: if we put vertices {a,b} in the same group for v, then edge (a,b) is covered.
        # We want to maximize coverage of internal edges while minimizing groups.

        # Strategy: for each v, we choose a partition of Nv[v] into subsets.
        # Each subset S means we create clique {v} + S.
        # The number of cliques contributed by v = number of subsets.
        # Internal edge (a,b) is covered if there exists v with a,b in the same subset for v.

        # Let's think of it as: we assign each pair (a,b) in C to be "covered" by some v
        # if a,b are in the same part of v's partition.

        # Simple approach: for each v, greedily build maximal cliques (subsets) of Nv[v]
        # that cover as many uncovered internal edges as possible.

        # Better: use a global approach.
        # For each internal edge (a,b), it needs to be covered by at least one v
        # where both a and b are in Nv[v] and in the same part.

        # Let's define: for each v, a partition of Nv[v] into parts.
        # Cost = sum_v (number of parts for v) + (number of cliques needed for uncovered internal edges)

        # Uncovered internal edges: those (a,b) such that for all v with a,b in Nv[v],
        # a and b are in different parts of v's partition.

        # This is complex. Let's use a heuristic:
        # 1. For each v, start with each element of Nv[v] in its own part.
        # 2. Merge parts if it helps cover internal edges (reduces uncovered edges).
        # 3. The trade-off: merging two parts reduces part count by 1 but may cover some edges.

        # Actually, merging parts in v's partition:
        # - Reduces the number of cliques for v by 1
        # - Covers all internal edges between the two merged subsets that were previously uncovered
        # So merging is always beneficial if it covers any previously uncovered edge!

        # Wait, that's not quite right. Merging always reduces clique count by 1.
        # It covers edges that were uncovered. So it's a win-win if it covers any edge.
        # If it covers no edge, it's still a win (reduces count by 1 with no downside).

        # So the optimal for each v independently: merge all of Nv[v] into one part!
        # That gives 1 clique {v} + Nv[v] for each v.

        # But wait - that covers ALL internal edges where both endpoints are in Nv[v] for some v.
        # The remaining uncovered internal edges need additional cliques.

        # Let me verify: if v has Nv[v] = {a,b,c}, then clique {v,a,b,c} covers edges (a,b),(a,c),(b,c).
        # Yes! So the simplest approach: for each v, one clique {v} + Nv[v].
        # Then cover remaining internal edges with greedy clique cover.

        # But can we do better? What if we split Nv[v] into multiple parts?
        # That would increase the number of cliques for v, which seems bad.
        # Unless... it allows us to use fewer cliques overall? No, splitting only adds cliques.

        # So the optimal is: one clique per v (with its full C-neighborhood), then cover leftovers.

        # Wait, but what if Nv[v] is empty? Then no clique for v.
        # And what if two v's have the same Nv[v]? We still need separate cliques because
        # each clique can only contain one I-vertex.

        # Let me reconsider. The problem is:
        # - Each edge (v,u) for v in I, u in C must be in exactly one clique.
        # - That clique must be {v} + S where u in S and S is a clique in C.
        # - So for each v, we partition Nv[v] into subsets S_1, ..., S_m.
        # - Each subset gives a clique {v} + S_i.
        # - m = number of cliques for v.
        # - We want to minimize total cliques.

        # Since C is a clique, any subset is valid. So we can put all of Nv[v] in one subset.
        # This gives 1 clique per v (if Nv[v] non-empty).

        # The only reason to split would be if it helps reduce the number of cliques
        # needed for the remaining internal C-edges. But splitting v's neighborhood
        # doesn't help cover more internal edges - it covers fewer (only edges within each subset).

        # So the optimal strategy is:
        # 1. For each v in I with non-empty Nv[v], create clique {v} + Nv[v].
        # 2. Find all internal C-edges not covered by any such clique.
        # 3. Cover those remaining edges with a minimum clique cover of the remaining subgraph of C.

        # Let me implement this.

        cliques = []
        covered_internal = set()

        for v in I:
            nv = Nv[v]
            if not nv:
                continue
            clique = sorted([v] + list(nv))
            cliques.append(clique)
            nv_list = list(nv)
            for i in range(len(nv_list)):
                a = nv_list[i]
                for j in range(i+1, len(nv_list)):
                    b = nv_list[j]
                    covered_internal.add((min(a,b), max(a,b)))

        remaining = internal_edges - covered_internal

        if remaining:
            # Cover remaining internal edges with greedy clique cover
            rem_adj = {}
            for u in range(k):
                rem_adj[u] = set()
            for a, b in remaining:
                rem_adj[a].add(b)
                rem_adj[b].add(a)

            rem_edges = set(remaining)
            while rem_edges:
                # Find a maximal clique in the remaining graph
                # Start with the edge with max degree
                best_e = max(rem_edges, key=lambda e: (len(rem_adj[e[0]]), len(rem_adj[e[1]])))
                clique = {best_e[0], best_e[1]}
                # Try to add vertices
                changed = True
                while changed:
                    changed = False
                    for u in range(k):
                        if u in clique:
                            continue
                        if all(u in rem_adj[x] for x in clique):
                            clique.add(u)
                            changed = True
                            break

                cliques.append(sorted(clique))
                cl_list = list(clique)
                for i in range(len(cl_list)):
                    a = cl_list[i]
                    for j in range(i+1, len(cl_list)):
                        b = cl_list[j]
                        rem_edges.discard((min(a,b), max(a,b)))

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if time.time() > deadline - 5:
            while len(results) < 3:
                results.append([])
            return results

        n, k, p, seed = spec
        result = solve_one(n, k, p, seed, deadline - time.time())
        cleaned = []
        for clique in result:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results
