import time
from random import Random
from itertools import combinations

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

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
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)

        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        I_verts = list(range(k, n))
        m = len(I_verts)
        m_cc = k * (k - 1) // 2

        cc_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cc_edge_idx = {}
        for idx, e in enumerate(cc_edges):
            cc_edge_idx[e] = idx

        E_N = []
        d_v = []
        for i, v in enumerate(I_verts):
            neigh = list(N_C[v])
            d_v.append(len(neigh))
            s = set()
            for a, b in combinations(neigh, 2):
                e = (a, b) if a < b else (b, a)
                if e in cc_edge_idx:
                    s.add(cc_edge_idx[e])
            E_N.append(s)

        # For each I-vertex, the C-C edges in N_C[v]
        # If we make a big clique {v} ∪ N_C[v], it covers all edges in E_N[i]
        # But two big cliques for v1 and v2 can't share any C-C edge
        # So E_N[i] for different i's must be disjoint
        # Otherwise we need to be more careful

        # Strategy: partition C-C edges among I-vertices
        # Each I-vertex v gets a subset S_v of C-vertices
        # such that:
        # 1. N_C[v] ⊆ S_v (to cover all cross edges of v)
        # 2. All edges in E(S_v) are assigned to v
        # 3. For any two vertices v1, v2, the C-C edges covered by v1 and v2 are disjoint
        # 4. The union of all covered C-C edges plus uncovered edges = all C-C edges
        # 5. Uncovered C-C edges need separate C-only cliques

        # Objective: minimize |{big cliques}| + sum(d_v for non-big) + theta(uncovered)

        # Key insight: for a big clique {v} ∪ S_v, the "cost" is 1
        # but it covers d_v cross edges and |E(S_v)| C-C edges
        # For a non-big vertex, cost is d_v (one per cross edge)

        # Let's think about which vertices to make "big"
        # A vertex v is "big" if we use one clique for all its cross edges
        # The constraint is that the C-C edges covered by different big vertices must be disjoint

        # Let's model this as: choose a set S of big vertices
        # For each v in S, E_N[v] must be disjoint from E_N[w] for w != v in S
        # If not disjoint, we can't make both big (or we need to reduce the set)

        # Actually, the constraint is more subtle. If v and w are both big,
        # then E_N[v] ∩ E_N[w] must be empty (they can't share any C-C edge)

        # Let's check: can we make all vertices big?
        # Only if all E_N[v] are pairwise disjoint

        # For now, let's try:
        # 1. Find a maximal set S of vertices with pairwise disjoint E_N
        # 2. Make those big
        # 3. Make others individual
        # 4. Cover remaining C-C edges with C-only cliques

        # Find maximal set with pairwise disjoint E_N
        # This is like finding a maximal set of non-overlapping sets
        # Greedy: sort by |E_N[v]| descending, add if disjoint from all already added

        S_candidates = list(range(m))
        S_candidates.sort(key=lambda i: len(E_N[i]), reverse=True)

        S_big = []
        used_edges = set()
        for i in S_candidates:
            if d_v[i] < 2:
                continue
            e_set = E_N[i]
            if e_set & used_edges:
                continue
            S_big.append(i)
            used_edges |= e_set

        S_big_set = frozenset(S_big)

        # Now compute remaining C-C edges
        consumed = set()
        for i in S_big:
            consumed |= E_N[i]
        remaining = set(range(m_cc)) - consumed

        # Cover remaining C-C edges with C-only cliques
        # The remaining graph is a subgraph of K_k
        # We need a clique cover of the remaining edges
        # This is equivalent to coloring the complement graph of the remaining edges

        rem_adj = [set() for _ in range(k)]
        for e_idx in remaining:
            a, b = cc_edges[e_idx]
            rem_adj[a].add(b)
            rem_adj[b].add(a)

        active = set()
        for e_idx in remaining:
            a, b = cc_edges[e_idx]
            active.add(a)
            active.add(b)

        ccliques = []
        if active:
            # Check if complete
            if len(remaining) == len(active) * (len(active) - 1) // 2:
                ccliques = [sorted(active)]
            else:
                # Greedy coloring on complement
                # In the remaining graph, edges are "conflicts"
                # We want to partition active into cliques of the original graph
                # = independent sets in the complement
                # Complement: two vertices are adjacent if they are NOT adjacent in remaining

                comp_adj = [set() for _ in range(k)]
                for a in active:
                    for b in active:
                        if a != b and b not in rem_adj[a]:
                            comp_adj[a].add(b)

                # Greedy: repeatedly find a maximal independent set in complement
                # (which is a clique in original remaining graph)
                uncolored = set(active)
                while uncolored:
                    # Pick vertex with highest degree in comp_adj among uncolored
                    best_v = max(uncolored, key=lambda x: len(comp_adj[x] & uncolored))
                    color_class = [best_v]
                    candidates = list(uncolored - {best_v})
                    candidates.sort(key=lambda x: -len(comp_adj[x] & uncolored))
                    for v in candidates:
                        if v not in uncolored:
                            continue
                        if all(v not in comp_adj[c] for c in color_class):
                            color_class.append(v)
                            uncolored.discard(v)
                    if len(color_class) >= 2:
                        ccliques.append(sorted(color_class))
                    uncolored -= set(color_class)

        # Build cliques
        cliques = []
        for i in range(m):
            v = I_verts[i]
            if i in S_big_set:
                cliques.append([v] + sorted(N_C[v]))
            else:
                for u in sorted(N_C[v]):
                    cliques.append([v, u])
        cliques.extend(ccliques)

        if not cliques:
            cliques = [list(range(k))]

        # Validate
        edge_count = {}
        for clique in cliques:
            if len(clique) < 2:
                continue
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                edge_count[e] = edge_count.get(e, 0) + 1

        missing = [e for e in edge_set if edge_count.get(e, 0) == 0]
        over = [e for e, c in edge_count.items() if c > 1]

        if missing or over:
            cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    cliques.append([u, v])

        results.append(cliques)

    return results
