import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        cliques = []
        covered_edges_in_C = set()

        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(neighbors[v] & set(range(k))), reverse=True)

        for v in I_vertices:
            N_v = neighbors[v] & set(range(k))
            if not N_v:
                continue

            # Greedily find the largest valid subset of N_v
            S = set(N_v)
            changed = True
            while changed:
                changed = False
                for u in list(S):
                    conflict = False
                    for w in S:
                        if u < w:
                            if (u, w) in covered_edges_in_C:
                                conflict = True
                                break
                    if conflict:
                        S.discard(u)
                        changed = True

            if len(S) >= 2:
                cliques.append([v] + sorted(S))
                for u1, u2 in combinations(sorted(S), 2):
                    covered_edges_in_C.add((u1, u2))
            elif len(S) == 1:
                u = next(iter(S))
                cliques.append([v, u])

            for u in N_v:
                if u not in S:
                    cliques.append([v, u])

        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges_in_C:
                    remaining_edges.add((u, w))

        rem_set = set(remaining_edges)
        while rem_set:
            if time.time() > deadline:
                break
            u, w = next(iter(rem_set))
            clique_vertices = {u, w}

            candidates = []
            for x in range(k):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in rem_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)

            clique_vertices.update(candidates)

            for u1, u2 in combinations(sorted(clique_vertices), 2):
                rem_set.discard((u1, u2))

            cliques.append(sorted(clique_vertices))

        # Final pass: handle any remaining edges that were not covered
        # (shouldn't happen if the while loop completed, but safety net)
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in range(k):
                if u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))

        covered_all = set()
        for cl in cliques:
            for u1, u2 in combinations(cl, 2):
                covered_all.add((min(u1, u2), max(u1, u2)))

        uncovered = all_edges - covered_all
        # Group uncovered edges into cliques
        unc_set = set(uncovered)
        while unc_set:
            u, w = next(iter(unc_set))
            clique_vertices = {u, w}
            candidates = []
            for x in range(n):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in unc_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)
            clique_vertices.update(candidates)
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                unc_set.discard((u1, u2))
            if len(clique_vertices) >= 2:
                cliques.append(sorted(clique_vertices))

        results.append(cliques)

    return results
