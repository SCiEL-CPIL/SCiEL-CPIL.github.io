def construct():
    import time
    from random import Random

    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, variant_seed=None):
        cross_edges = instance(n, k, p, seed)

        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)

        # Available C-edges
        all_c_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_c_edges.add((u, w))
        available = set(all_c_edges)

        mixed_cliques = []

        # Determine order of I vertices
        if variant_seed is None:
            # Ascending degree (smallest first) - best known
            I_vertices = sorted(range(k, n), key=lambda x: len(N[x]))
            rng_var = None
        else:
            rng_var = Random(variant_seed)
            if variant_seed % 3 == 0:
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]))
            elif variant_seed % 3 == 1:
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]) + rng_var.random() * 0.5)
            else:
                I_vertices = list(range(k, n))
                rng_var.shuffle(I_vertices)

        for v in I_vertices:
            if not N[v]:
                continue

            remaining_neighbors = list(N[v])
            rem_set = set(remaining_neighbors)

            # Greedy matching: pair up C-neighbors using cheap C-edges
            # Sort by current degree in available (ascending) to consume cheap edges first
            def c_deg(c):
                cnt = 0
                for w in rem_set:
                    if w == c:
                        continue
                    e = (min(c, w), max(c, w))
                    if e in available:
                        cnt += 1
                return cnt

            sorted_rem = sorted(rem_set, key=lambda c: c_deg(c))
            matched = set()
            pairings = []

            for idx_a in range(len(sorted_rem)):
                c_a = sorted_rem[idx_a]
                if c_a in matched:
                    continue
                # Find best partner
                best_partner = None
                best_cost = float('inf')
                for idx_b in range(idx_a + 1, len(sorted_rem)):
                    c_b = sorted_rem[idx_b]
                    if c_b in matched:
                        continue
                    e = (min(c_a, c_b), max(c_a, c_b))
                    if e in available:
                        cost = c_deg(c_a) + c_deg(c_b)
                        if cost < best_cost:
                            best_cost = cost
                            best_partner = c_b
                if best_partner is not None:
                    matched.add(c_a)
                    matched.add(best_partner)
                    pairings.append((c_a, best_partner))
                    # Remove this edge from available
                    e = (min(c_a, best_partner), max(c_a, best_partner))
                    available.discard(e)

            # Create 3-cliques for pairs
            for c_a, c_b in pairings:
                mixed_cliques.append([int(v), int(c_a), int(c_b)])

            # Singles: unmatched C-neighbors
            for c in sorted_rem:
                if c not in matched:
                    mixed_cliques.append([int(v), int(c)])

        # Now cover remaining available edges with pure cliques
        pure_cliques = []
        rem_edges = set(available)

        while rem_edges:
            if time.time() > deadline:
                for e in rem_edges:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break

            # Pick edge with max combined degree
            best_edge = None
            best_score = -1
            for e in rem_edges:
                u, w = e
                deg_u = sum(1 for x in rem_edges if u in x)
                deg_w = sum(1 for x in rem_edges if w in x)
                score = deg_u + deg_w
                if score > best_score:
                    best_score = score
                    best_edge = e

            if best_edge is None:
                break

            u0, w0 = best_edge
            # Greedy maximal clique
            S = [u0, w0]
            S_set = {u0, w0}
            cand = set()
            for e in rem_edges:
                for x in e:
                    if x not in S_set:
                        cand.add(x)

            while cand:
                best_v = None
                best_deg = -1
                for w in cand:
                    ok = True
                    deg = 0
                    for s in S:
                        e = (min(w, s), max(w, s))
                        if e not in rem_edges:
                            ok = False
                            break
                        deg += 1
                    if ok and deg > best_deg:
                        best_deg = deg
                        best_v = w
                if best_v is None:
                    break
                S.append(best_v)
                S_set.add(best_v)
                new_cand = set()
                for w in cand:
                    if w == best_v:
                        continue
                    ok = True
                    for s in S:
                        e = (min(w, s), max(w, s))
                        if e not in rem_edges:
                            ok = False
                            break
                    if ok:
                        new_cand.add(w)
                cand = new_cand

            if len(S) >= 2:
                pure_cliques.append([int(x) for x in S])
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                pure_cliques.append([int(u0), int(w0)])
                rem_edges.discard(best_edge)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []

    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')

        if time.time() > deadline - 5.0:
            ce = instance(n, k, p, seed)
            cliques = [[int(u) for u in range(k)]]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue

        # Try base first
        cliques = solve_instance(n, k, p, seed, None)
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques

        # Try variants
        for i in range(40):
            if time.time() > deadline - 15.0:
                break
            v_seed = seed + 1000 + i
            cliques = solve_instance(n, k, p, seed, v_seed)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
