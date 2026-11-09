import time
import sys
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)

        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))

        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs

        # Group I vertices by their exact neighborhood in C
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)

        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))

        best_partition = None
        best_count = float('inf')

        # Try multiple strategies with different orderings and clique-finding heuristics
        strategies = [
            ('desc', 'greedy'),   # descending shadow size, greedy clique
            ('asc', 'greedy'),    # ascending shadow size, greedy clique
            ('desc', 'max_common'),  # descending, max common neighbors
            ('asc', 'max_common'),   # ascending, max common neighbors
        ]

        for strat_name, clique_method in strategies:
            if time.time() > deadline - 5.0:
                break

            partition = []
            used_c_edges = set()

            if strat_name == 'desc':
                group_order = sorted_groups
            else:
                group_order = sorted(sorted_groups, key=lambda x: len(x[0]))

            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue

                # Find a large clique within shadow using unused C-C edges
                available = list(shadow)

                if clique_method == 'greedy':
                    # Sort by number of unused C-C edges to other available vertices
                    def unused_degree(u):
                        cnt = 0
                        for w in available:
                            if u != w:
                                e = (min(u, w), max(u, w))
                                if e in internal_edges and e not in used_c_edges:
                                    cnt += 1
                        return cnt

                    available.sort(key=lambda u: -unused_degree(u))

                    clique = []
                    for u in available:
                        ok = True
                        for w in clique:
                            e = (min(u, w), max(u, w))
                            if e not in internal_edges or e in used_c_edges:
                                ok = False
                                break
                        if ok:
                            clique.append(u)

                else:  # max_common
                    # Use max common neighbors heuristic
                    if not available:
                        clique = []
                    else:
                        # Start with vertex having max unused degree
                        start = max(available, key=lambda u: sum(
                            1 for w in available if u != w and
                            (min(u, w), max(u, w)) in internal_edges and
                            (min(u, w), max(u, w)) not in used_c_edges
                        ))
                        clique = [start]
                        remaining = [u for u in available if u != start]

                        while remaining:
                            # Find vertex with max common unused edges to all in clique
                            best_v = None
                            best_score = -1
                            for v in remaining:
                                score = 0
                                for w in clique:
                                    e = (min(v, w), max(v, w))
                                    if e in internal_edges and e not in used_c_edges:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_v = v
                            if best_v is None or best_score < len(clique):
                                break
                            clique.append(best_v)
                            remaining.remove(best_v)

                # Form the clique: all I vertices in group + the C clique
                if len(clique) >= 1:
                    new_clique = sorted(i_vertices + clique)
                    partition.append(new_clique)
                    # Mark C-C edges as used
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            used_c_edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))

                # Cover remaining cross edges for this group's I vertices
                # For each I vertex v, edges to C vertices not in clique
                for v in i_vertices:
                    remaining_nbrs = shadow - set(clique)
                    for u in remaining_nbrs:
                        partition.append(sorted([v, u]))

            # Cover remaining C-C edges
            rem_c_edges = internal_edges - used_c_edges
            rem_set = set(rem_c_edges)

            while rem_set:
                if time.time() > deadline - 3.0:
                    break
                rem_adj = {u: set() for u in range(k)}
                for (a, b) in rem_set:
                    rem_adj[a].add(b)
                    rem_adj[b].add(a)

                max_deg = -1
                start_v = None
                for u in range(k):
                    if len(rem_adj[u]) > max_deg:
                        max_deg = len(rem_adj[u])
                        start_v = u
                if start_v is None:
                    break

                clique = [start_v]
                candidates = sorted(rem_adj[start_v], key=lambda x: -len(rem_adj[x]))
                for w in candidates:
                    ok = True
                    for u in clique:
                        if w not in rem_adj[u]:
                            ok = False
                            break
                    if ok:
                        clique.append(w)

                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))

            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition

        if best_partition is None:
            return []

        cleaned = []
        for clique in best_partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
