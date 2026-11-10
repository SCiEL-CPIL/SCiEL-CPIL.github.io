"""Rewritten solver: multi-start greedy + best-improvement local search with deadline guard."""

def construct():
    import time, math, random

    t0 = time.time()
    DEADLINE = 82.0  # safety margin under 90s kill

    N = 255
    vals = list(range(2, 257))          # vals[i] = i + 2
    inv = [1.0 / v for v in vals]

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---- precompute conflicts: conflict[i] = set of j with lcm(vals[i],vals[j]) <= 256
    conflict = [set() for _ in range(N)]
    for i in range(N):
        a = vals[i]
        for j in range(i + 1, N):
            b = vals[j]
            g = math.gcd(a, b)
            if (a // g) * b <= 256:     # lcm(a,b) <= 256  -> incompatible
                conflict[i].add(j)
                conflict[j].add(i)
    conf_frozen = [frozenset(s) for s in conflict]   # fast membership
    conf_list = [list(s) for s in conflict]          # iteration

    def cur_sum(in_set):
        return sum(inv[i] for i in range(N) if in_set[i])

    # ---------------- greedy builders (all return a maximal independent set) ----------
    def greedy_order(order):
        in_set = [False] * N
        blocked = [0] * N
        S = []
        for i in order:
            if blocked[i] == 0:
                in_set[i] = True
                S.append(i)
                for j in conf_list[i]:
                    if not in_set[j]:
                        blocked[j] += 1
        return S, in_set, blocked

    def greedy_weighted(perturb):
        in_set = [False] * N
        blocked = [0] * N
        S = []
        remaining = set(range(N))
        while remaining and time_left() > 0.3:
            cands = [i for i in remaining if blocked[i] == 0]
            if not cands:
                break
            if perturb > 0.0:
                best = max(cands, key=lambda i: inv[i] + perturb * rng.random())
            else:
                best = max(cands, key=lambda i: inv[i])
            in_set[best] = True
            S.append(best)
            remaining.discard(best)
            for j in conf_list[best]:
                if not in_set[j]:
                    blocked[j] += 1
        return S, in_set, blocked

    # ---------------- local search (best-improvement) --------------------------------
    def local_search(S, in_set, blocked, start_time):
        K = 35  # candidate pool size for 1-for-2

        def remove(i):
            in_set[i] = False
            S.remove(i)
            for j in conf_list[i]:
                if not in_set[j]:
                    blocked[j] -= 1

        def add(i):
            in_set[i] = True
            S.append(i)
            for j in conf_list[i]:
                if not in_set[j]:
                    blocked[j] += 1

        best_score = cur_sum(in_set)
        start = time.time()
        while time_left() > 1.0 and (time.time() - start) < 2.5:
            improved = False

            # ---- 1-for-1 swap (best improvement over all a) ----
            ba, bb, bg = -1, -1, 0.0
            for a in list(S):
                remove(a)
                best_b, best_g = -1, 0.0
                for b in range(N):
                    if (not in_set[b]) and blocked[b] == 0:
                        g = inv[b] - inv[a]
                        if g > best_g:
                            best_g = g
                            best_b = b
                if best_b >= 0 and best_g > bg:
                    bg, ba, bb = best_g, a, best_b
                add(a)
            if ba >= 0:
                remove(ba)
                add(bb)
                best_score += bg
                improved = True

            # ---- 1-for-2 (best improvement) ----
            if not improved and time_left() > 1.0:
                ba, b1, b2, bg = -1, -1, -1, 0.0
                for a in list(S):
                    remove(a)
                    A = [i for i in range(N) if (not in_set[i]) and blocked[i] == 0]
                    if len(A) >= 2:
                        A.sort(key=lambda i: inv[i], reverse=True)
                        top = A[:K]
                        for x in range(len(top)):
                            ix = top[x]
                            for y in range(x + 1, len(top)):
                                iy = top[y]
                                if iy in conf_frozen[ix]:
                                    continue
                                g = inv[ix] + inv[iy] - inv[a]
                                if g > bg:
                                    bg, ba, b1, b2 = g, a, ix, iy
                    add(a)
                if ba >= 0:
                    remove(ba)
                    add(b1)
                    add(b2)
                    best_score += bg
                    improved = True

            # ---- 2-for-1 (capped; only for moderate |S|) ----
            if not improved and len(S) <= 80 and time_left() > 1.5:
                ba1, ba2, bb, bg = -1, -1, -1, 0.0
                L = list(S)
                for x in range(len(L)):
                    a1 = L[x]
                    for y in range(x + 1, len(L)):
                        a2 = L[y]
                        remove(a1)
                        remove(a2)
                        best_b, best_g = -1, 0.0
                        for b in range(N):
                            if (not in_set[b]) and blocked[b] == 0:
                                g = inv[b] - inv[a1] - inv[a2]
                                if g > best_g:
                                    best_g = g
                                    best_b = b
                        add(a2)
                        add(a1)
                        if best_b >= 0 and best_g > bg:
                            bg, ba1, ba2, bb = best_g, a1, a2, best_b
                        if time_left() < 1.2:
                            break
                    if time_left() < 1.2:
                        break
                if ba1 >= 0:
                    remove(ba1)
                    remove(ba2)
                    add(bb)
                    best_score += bg
                    improved = True

            if not improved:
                break
        return best_score, in_set, S

    rng = random.Random(20240517)
    best_score = -1.0
    best_set = None

    def consider(S, in_set, blocked):
        nonlocal best_score, best_set
        sc, in_set, S = local_search(S, in_set, blocked, time.time())
        if sc > best_score:
            best_score = sc
            best_set = list(S)

    # ---- deterministic strong starts first (guarantee nonempty best) ----
    S, in_set, blocked = greedy_weighted(0.0)
    consider(S, in_set, blocked)

    S, in_set, blocked = greedy_order(list(range(N)))        # ascending value
    consider(S, in_set, blocked)

    S, in_set, blocked = greedy_order(list(range(N - 1, -1, -1)))  # descending
    consider(S, in_set, blocked)

    # ---- many randomized starts until deadline ----
    mode = 0
    while time_left() > 3.0:
        mode += 1
        if mode % 3 == 0:
            S, in_set, blocked = greedy_weighted(0.05)       # perturbed weighted
        else:
            o = list(range(N))
            rng.shuffle(o)
            S, in_set, blocked = greedy_order(o)             # random order
        consider(S, in_set, blocked)

    if best_set is None:
        best_set = [2]

    return [vals[i] for i in best_set]
