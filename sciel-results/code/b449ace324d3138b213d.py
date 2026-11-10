def construct():
    from random import Random
    import time

    start = time.time()
    deadline = start + 130.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)

        cross = {}
        for (a, b) in edges:
            if a < k and b >= k:
                cross.setdefault(b, set()).add(a)
            elif a >= k and b < k:
                cross.setdefault(a, set()).add(b)

        all_cc = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        used = set()
        cliques = []

        # Mixed (cross) cliques: for each I-vertex, pair up C-neighbors.
        # A pair (a,b) is "free" if the C-C edge (a,b) is still uncovered,
        # because using it here saves a pure-C clique later.
        for i in sorted(cross.keys(), key=lambda x: len(cross[x])):
            S = list(cross[i])
            if not S:
                continue
            if len(S) == 1:
                cliques.append([i, S[0]])
                continue

            # Pair up greedily: each pair consumes one uncovered C-C edge.
            remaining = set(S)
            while len(remaining) >= 2:
                # pick a C-vertex with fewest uncovered edges (cheapest to consume)
                best_a = None
                best_a_cost = None
                for a in remaining:
                    cnt = 0
                    for b in range(k):
                        if b != a and frozenset((a, b)) in all_cc and frozenset((a, b)) not in used:
                            cnt += 1
                    if best_a is None or cnt < best_a_cost:
                        best_a_cost = cnt
                        best_a = a
                # find a partner b in remaining that is adjacent to a in C and uncovered
                best_b = None
                for b in remaining:
                    if b == best_a:
                        continue
                    e = frozenset((best_a, b))
                    if e in all_cc and e not in used:
                        # prefer partner with fewest uncovered edges
                        cnt = 0
                        for c in range(k):
                            if c != b and frozenset((b, c)) in all_cc and frozenset((b, c)) not in used:
                                cnt += 1
                        if best_b is None or cnt < best_b_cost:
                            best_b_cost = cnt
                            best_b = b
                if best_b is not None:
                    used.add(frozenset((best_a, best_b)))
                    cliques.append([i, best_a, best_b])
                    remaining.discard(best_a)
                    remaining.discard(best_b)
                else:
                    # no free partner; emit a 2-clique
                    cliques.append([i, best_a])
                    remaining.discard(best_a)
            if remaining:
                a = next(iter(remaining))
                cliques.append([i, a])

        # Residual pure-C clique cover.
        remaining = all_cc - used
        rem_set = set(remaining)
        while rem_set:
            if time.time() > deadline:
                break
            verts = set()
            for e in rem_set:
                verts.update(e)
            # degree in remaining
            deg = {v: 0 for v in verts}
            for e in rem_set:
                deg[e[0]] += 1
                deg[e[1]] += 1
            start_v = max(verts, key=lambda v: deg[v])
            clique = [start_v]
            cands = [v for v in verts if v != start_v]
            for v in sorted(cands):
                ok = all(frozenset((v, c)) in rem_set for c in clique)
                if ok:
                    clique.append(v)
            if len(clique) < 2:
                e = next(iter(rem_set))
                clique = [e[0], e[1]]
            cliques.append(sorted(clique))
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    rem_set.discard(frozenset((clique[i], clique[j])))

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        results.append(cliques)
    return results
