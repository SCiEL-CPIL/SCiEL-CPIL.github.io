"""Rewritten solver: max-weight independent set on the lcm<=256 conflict graph."""
# EVOLVE-BLOCK-START
def construct():
    import time, math, random
    from fractions import Fraction

    t0 = time.time()
    DEADLINE = 82.0  # wall-clock safety margin under the 90s kill

    def time_up():
        return (time.time() - t0) > DEADLINE

    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    w = [1.0 / x for x in nums]

    # ---- build conflict graph: edge iff lcm(a,b) <= 256 ----
    nbr = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            if (a // g) * b <= 256:
                nbr[i].append(j)
                nbr[j].append(i)
                edges.append((i, j))

    def weight(S):
        return sum(w[i] for i in S)

    # ---- initialization: greedy min-weight vertex cover (remove to kill edges) ----
    def init_cover():
        S = set(range(n))
        # remove cheaper endpoint of each live conflict, repeat to fix
        order = sorted(edges, key=lambda e: max(w[e[0]], w[e[1]]), reverse=True)
        progress = True
        while progress and not time_up():
            progress = False
            for (i, j) in order:
                if i in S and j in S:
                    if w[i] <= w[j]:
                        S.discard(i)
                    else:
                        S.discard(j)
                    progress = True
        return S

    # ---- initialization: greedy add by weight descending ----
    def init_greedy_add():
        S = set()
        for i in sorted(range(n), key=lambda x: w[x], reverse=True):
            if all(j not in S for j in nbr[i]):
                S.add(i)
        return S

    # ---- local search: best-improvement add / swap until no gain ----
    def local_search(S):
        S = set(S)
        improved = True
        while improved and not time_up():
            improved = False
            best_gain = 1e-12
            best_move = None  # ('add', v) or ('swap', v_out, v_in)

            # add move
            for v in range(n):
                if v in S:
                    continue
                lost = sum(w[u] for u in nbr[v] if u in S)
                gain = w[v] - lost
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', v)

            # swap move: remove v_in_S, add u_not_in_S (u compatible with S\{v})
            for v in S:
                for u in nbr[v]:
                    if u in S:
                        continue
                    # u's neighbors in S must be exactly {v} to be compatible after removal
                    ok = True
                    for x in nbr[u]:
                        if x in S and x != v:
                            ok = False
                            break
                    if not ok:
                        continue
                    gain = w[u] - w[v]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap', v, u)

            if best_move is not None:
                if best_move[0] == 'add':
                    v = best_move[1]
                    S.add(v)
                    for u in nbr[v]:
                        S.discard(u)
                else:
                    v, u = best_move[1], best_move[2]
                    S.discard(v)
                    S.add(u)
                improved = True

        # completion: greedily add any compatible vertex by weight
        for i in sorted(range(n), key=lambda x: w[x], reverse=True):
            if i not in S and all(j not in S for j in nbr[i]):
                S.add(i)
        return S

    # ---- run several restarts, keep best ----
    best = set()
    best_val = -1.0
    inits = [init_cover(), init_greedy_add()]

    # a few randomized cover inits
    rng = random.Random(12345)
    for _ in range(3):
        S = set(range(n))
        # random order removal
        eo = edges[:]
        rng.shuffle(eo)
        progress = True
        while progress and not time_up():
            progress = False
            for (i, j) in eo:
                if i in S and j in S:
                    if rng.random() < 0.5:
                        S.discard(i)
                    else:
                        S.discard(j)
                    progress = True
        inits.append(S)

    for S in inits:
        if time_up():
            break
        S2 = local_search(S)
        val = weight(S2)
        if val > best_val:
            best_val = val
            best = S2

    # ---- final exact rational sum ----
    total = Fraction(0, 1)
    for i in best:
        total += Fraction(1, nums[i])

    result = sorted(nums[i] for i in best)
    return result
# EVOLVE-BLOCK-END
