import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80

    # base adjacency bitsets
    base_adj = [0] * n
    for u, v in base_edges:
        base_adj[u] |= (1 << v)
        base_adj[v] |= (1 << u)

    # coverage of a star centered at s: s-1, s, s+1
    cover = [0] * n
    for s in range(n):
        m = 0
        for v in (s - 1, s, s + 1):
            if 0 <= v < n:
                m |= (1 << v)
        cover[s] = m

    # ---- fast validation of a full edge set ----
    def validate(added):
        edge_set = set()
        for u, v in added:
            mn, mx = (u, v) if u < v else (v, u)
            if (mn, mx) in edge_set:
                return False
            edge_set.add((mn, mx))
            # existing edge?
            if base_adj[mn] & (1 << mx):
                return False
        adj = base_adj[:]
        for u, v in added:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        # triangle-free
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & (1 << v)) and (adj[u] & adj[v]):
                    return False
        # connectivity
        dist = [-1] * n
        dist[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            nb = adj[u]
            for v in range(n):
                if (nb >> v) & 1 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if -1 in dist:
            return False
        # diameter <= 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                nb = adj[u]
                for v in range(n):
                    if (nb >> v) & 1 and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d > 4 for d in dist):
                return False
        return True

    best = None

    # ---- 1-hub star search (kept from current program) ----
    def search_one_hub():
        nonlocal best
        for k in (6, 7):
            for c in range(n):
                if time.time() > deadline:
                    return
                far_mask = 0
                for v in range(n):
                    if abs(v - c) > 2:
                        far_mask |= (1 << v)
                candidates = [s for s in range(n) if abs(s - c) >= 3]
                candidates.sort(key=lambda s: (cover[s] & far_mask).bit_count(), reverse=True)

                def bt(idx, chosen, covered):
                    nonlocal best
                    if time.time() > deadline:
                        return False
                    if len(chosen) == k:
                        if (covered & far_mask) == far_mask:
                            sol = [(c, s) for s in chosen]
                            if validate(sol):
                                best = sol
                                return True
                        return False
                    if idx >= len(candidates):
                        return False
                    rem = 0
                    for i in range(idx, len(candidates)):
                        rem |= cover[candidates[i]]
                    if (covered | rem) & far_mask != far_mask:
                        return False
                    if bt(idx + 1, chosen, covered):
                        return True
                    s = candidates[idx]
                    ok = True
                    for w in chosen:
                        if base_adj[s] & (1 << w):
                            ok = False
                            break
                    if ok:
                        if bt(idx + 1, chosen + [s], covered | cover[s]):
                            return True
                    return False

                if bt(0, [], 0):
                    return

    # ---- 2-hub search (NEW structural family) ----
    def search_two_hubs():
        nonlocal best
        # hubs must be far apart so their spokes don't collide into triangles.
        hub_pairs = []
        for h1 in range(n):
            for h2 in range(h1 + 1, n):
                if abs(h2 - h1) < 6:
                    continue
                hub_pairs.append((h1, h2))
        for (h1, h2) in hub_pairs:
            if time.time() > deadline:
                return
            # spokes for h1: dist>=3 from h1, and not creating triangle with h2's spokes
            cand1 = [s for s in range(n) if abs(s - h1) >= 3 and abs(s - h2) >= 3 and s != h2]
            cand2 = [s for s in range(n) if abs(s - h2) >= 3 and abs(s - h1) >= 3 and s != h1]
            # far vertices (path dist > 2 from BOTH hubs) need coverage
            far_mask = 0
            for v in range(n):
                if abs(v - h1) > 2 and abs(v - h2) > 2:
                    far_mask |= (1 << v)
            cand1.sort(key=lambda s: (cover[s] & far_mask).bit_count(), reverse=True)
            cand2.sort(key=lambda s: (cover[s] & far_mask).bit_count(), reverse=True)

            # enumerate spoke subsets (small, since dist>=3 restricts heavily)
            def subsets(lst, max_size):
                from itertools import combinations
                for r in range(0, max_size + 1):
                    for combo in combinations(lst, r):
                        yield combo

            total_budget = 6  # max total spokes (h1_spokes + h2_spokes), connecting edge optional
            found = False
            for s1 in subsets(cand1, 4):
                if time.time() > deadline:
                    return
                cov1 = 0
                indep1 = True
                for a in s1:
                    cov1 |= cover[a]
                    for b in s1:
                        if a < b and (base_adj[a] & (1 << b)):
                            indep1 = False
                if not indep1:
                    continue
                rem1 = total_budget - len(s1)
                for s2 in subsets(cand2, rem1):
                    if time.time() > deadline:
                        return
                    cov2 = 0
                    indep2 = True
                    for a in s2:
                        cov2 |= cover[a]
                        for b in s2:
                            if a < b and (base_adj[a] & (1 << b)):
                                indep2 = False
                    if not indep2:
                        continue
                    if (cov1 | cov2) & far_mask != far_mask:
                        continue
                    # cross-independence between s1 and s2
                    cross_ok = True
                    for a in s1:
                        for b in s2:
                            if base_adj[a] & (1 << b):
                                cross_ok = False
                    if not cross_ok:
                        continue
                    # spokes must not create triangles with the hubs
                    # (already ensured by dist>=3 from own hub and dist>=3 from other hub)
                    edges = []
                    for s in s1:
                        edges.append((h1, s))
                    for s in s2:
                        edges.append((h2, s))
                    # try without connecting edge
                    if validate(edges):
                        best = edges
                        found = True
                        return
                    # try with connecting edge (h1,h2) if not existing
                    if not (base_adj[h1] & (1 << h2)):
                        if validate(edges + [(h1, h2)]):
                            best = edges + [(h1, h2)]
                            found = True
                            return

    # ---- known valid 7-edge fallback ----
    def fallback():
        nonlocal best
        sol = [(11, 1), (11, 4), (11, 7), (11, 15), (11, 18), (11, 21), (11, 23)]
        if validate(sol):
            best = sol
        else:
            # brute small
            from itertools import combinations
            cands = [(u, v) for u in range(n) for v in range(u + 1, n)
                     if not (base_adj[u] & (1 << v)) and abs(v - u) >= 3]
            for k in range(5, 8):
                for combo in combinations(cands, k):
                    if time.time() > deadline:
                        return
                    if validate(list(combo)):
                        best = list(combo)
                        return

    # run searches in order of expected edge count
    search_two_hubs()
    if best is None:
        search_one_hub()
    if best is None:
        fallback()

    if best is None:
        return []
    return [[int(u), int(v)] for u, v in best]
