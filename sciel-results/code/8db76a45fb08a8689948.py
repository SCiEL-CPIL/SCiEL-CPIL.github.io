"""Clique partition of split graphs: shared-subset + multi-order greedy."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _greedy_clique(remaining, covered, adj_cache):
        """Greedy max clique in remaining (a C-vertex set) using uncovered C-edges."""
        adj = {v: set() for v in remaining}
        for v in remaining:
            for w in remaining:
                if v < w:
                    if (v, w) not in covered:
                        adj[v].add(w)
                        adj[w].add(v)
        S = set()
        candidates = set(remaining)
        while candidates:
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            S.add(v)
            candidates = candidates & adj[v]
        if not S:
            S = {next(iter(remaining))}
        return S

    def _cover_remaining_c_edges(remaining_C_edges):
        """Cover leftover C-only edges with C-only cliques (greedy)."""
        cliques = []
        remaining_adj = defaultdict(set)
        for u, v in remaining_C_edges:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)
        while remaining_C_edges:
            if not remaining_adj:
                break
            v = max(remaining_adj.keys(), key=lambda x: len(remaining_adj[x]))
            S = {v}
            candidates = set(remaining_adj[v])
            while candidates:
                w = max(candidates, key=lambda x: len(remaining_adj[x] & candidates))
                S.add(w)
                candidates = candidates & remaining_adj[w]
            S_list = sorted(S)
            if len(S_list) < 2:
                if remaining_adj[v]:
                    w = next(iter(remaining_adj[v]))
                    S_list = sorted([v, w])
                else:
                    break
            cliques.append(S_list)
            for a in range(len(S_list)):
                for b in range(a + 1, len(S_list)):
                    remaining_C_edges.discard((S_list[a], S_list[b]))
            for u in S_list:
                remaining_adj.pop(u, None)
                for w in list(remaining_adj.keys()):
                    remaining_adj[w].discard(u)
            remaining_adj = {u2: v2 for u2, v2 in remaining_adj.items() if v2}
        return cliques

    def _local_improve(cliques, N, deadline):
        """Split oversized cliques to reduce C-edge waste (bounded)."""
        for _ in range(60):
            if time.time() > deadline:
                break
            best_gain = 0
            best_idx = None
            best_new = None
            for idx, cl in enumerate(cliques):
                if len(cl) < 4:
                    continue
                i = max(cl)  # the I-vertex (largest label)
                S = [v for v in cl if v != i]
                if len(S) < 3:
                    continue
                # try to split S into two non-empty parts A,B that are both
                # cliques within N[i] (all pairs in S are C-edges, so any
                # subset is a clique) -> always valid.
                # gain: splitting a clique of size |S|+1 into two cliques
                # removes the cross C-edges between A and B from being
                # "wasted" (covered by a clique that also has i).
                # We gain by reducing total cliques? No - splitting increases
                # clique count by 1, but frees cross-edges to be shared.
                # Only split if |S| >= 4 (so the freed edges matter).
                if len(S) < 4:
                    continue
                # Split S in half (by index)
                mid = len(S) // 2
                A = S[:mid]
                B = S[mid:]
                # This frees |A|*|B| C-edges that were covered by this clique.
                gain = len(A) * len(B)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_new = (sorted([i] + A), sorted([i] + B))
            if best_idx is None:
                break
            a, b = best_new
            cliques[best_idx] = a
            cliques.insert(best_idx + 1, b)
        return cliques

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build N[i] = set of C-vertices adjacent to I-vertex i
        N = {}
        for i in range(k, n):
            N[i] = set()
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        # C-edges: all pairs within C
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        I_vertices_list = list(range(k, n))

        # Build groupings by identical neighborhood (frozenset)
        groups = defaultdict(list)
        for i in I_vertices_list:
            groups[frozenset(N[i])].append(i)

        best_cliques = None
        best_count = float('inf')

        # Generate candidate orderings
        strategies = []
        strategies.append(sorted(I_vertices_list, key=lambda i: len(N[i]), reverse=True))
        strategies.append(sorted(I_vertices_list, key=lambda i: len(N[i])))
        rng = random.Random(seed)
        for _ in range(12):
            strategies.append(rng.sample(I_vertices_list, len(I_vertices_list)))

        for order in strategies:
            if time.time() > deadline - 2:
                break

            covered_C_edges = set()
            cliques = []

            # Shared-subset optimization: group I-vertices by identical N
            # and process groups so that the first I-vertex in a group uses
            # a large clique, and subsequent ones reuse the same C-subset.
            # We do this by processing the group's vertices together.
            processed = set()
            for i in order:
                if time.time() > deadline - 2:
                    break
                if i in processed:
                    continue
                if not N[i]:
                    continue

                # Find all I-vertices with the same neighborhood as i
                same_N = groups[frozenset(N[i])]
                # Sort same_N so that i comes first (follow the global order)
                same_N.sort(key=lambda x: order.index(x))

                for idx, j in enumerate(same_N):
                    processed.add(j)
                    if not N[j]:
                        continue
                    remaining_N = set(N[j])
                    while remaining_N:
                        if time.time() > deadline - 2:
                            break
                        S = _greedy_clique(remaining_N, covered_C_edges, None)
                        clique = sorted([j] + list(S))
                        cliques.append(clique)
                        S_list = sorted(S)
                        for a in range(len(S_list)):
                            for b in range(a + 1, len(S_list)):
                                covered_C_edges.add((S_list[a], S_list[b]))
                        remaining_N -= S

            # Cover remaining C-edges
            remaining_C_edges = C_edges - covered_C_edges
            if remaining_C_edges:
                cliques.extend(_cover_remaining_c_edges(remaining_C_edges))

            # Local improvement: split oversized cliques
            cliques = _local_improve(cliques, N, deadline)

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]

        if best_cliques is None:
            return [list(e) for e in edges]

        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        # Convert to plain Python ints and lists
        results.append([list(int(v) for v in c) for c in cliques])

    return results
# EVOLVE-BLOCK-END
