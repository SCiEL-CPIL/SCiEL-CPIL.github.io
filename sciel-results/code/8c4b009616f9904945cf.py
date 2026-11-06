"""Rewritten solver: maximum independent set on the equal-lcm-triple conflict graph."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd

    _t0 = time.time()
    _DEADLINE = _t0 + 50.0  # safety margin under the 90s wall-clock kill

    N = 128
    def factor_pp(n):
        comps = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                pp = 1
                while n % d == 0:
                    n //= d
                    pp *= d
                comps.append(pp)
            d += 1
        if n > 1:
            comps.append(n)
        return comps

    comp_sets = [None] + [set(factor_pp(i)) for i in range(1, N + 1)]

    def is_bad_triple(a, b, c):
        sa, sb, sc = comp_sets[a], comp_sets[b], comp_sets[c]
        if sa <= (sb | sc) or sb <= (sa | sc) or sc <= (sa | sb):
            return True
        return False

    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                if is_bad_triple(i, j, k):
                    adj[i] |= (1 << j) | (1 << k)
                    adj[j] |= (1 << i) | (1 << k)
                    adj[k] |= (1 << i) | (1 << j)

    # greedy lower bound (seed)
    def greedy_lb():
        selected = []
        for x in range(1, N + 1):
            ok = True
            for a in selected:
                if (adj[a] >> x) & 1:
                    ok = False
                    break
            if ok:
                selected.append(x)
        return selected

    best_set = greedy_lb()
    best_size = len(best_set)

    ALL = (1 << (N + 1))

    def popcount(x):
        return bin(x).count('1')

    # Maximum independent set, branch & bound with greedy coloring bound.
    # We operate on a bitset of candidate vertices P.
    # Greedy coloring upper bound: repeatedly pick max-degree vertex, color.
    def color_bound(P):
        # returns list of color class sizes (upper bound = sum)
        colors = []
        remaining = P
        while remaining:
            # pick vertex with max degree in remaining
            verts = [i for i in range(1, N + 1) if (remaining >> i) & 1]
            if not verts:
                break
            v = max(verts, key=lambda x: popcount(adj[x] & remaining))
            # color class: independent set within remaining containing v
            color_class = 1 << v
            remaining &= ~(1 << v)
            cand = remaining & ~adj[v]
            while cand:
                w = cand.bit_length() - 1
                if w == 0:
                    break
                color_class |= (1 << w)
                remaining &= ~(1 << w)
                cand &= ~adj[w]
            colors.append(popcount(color_class))
            remaining &= ~color_class
        return sum(colors)

    def expand(R, P):
        nonlocal best_size, best_set
        if time.time() > _DEADLINE:
            return
        if popcount(P) + len(R) <= best_size:
            return
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_set = list(R)
            return
        # upper bound via coloring
        ub = color_bound(P)
        if ub + len(R) <= best_size:
            return
        # pick vertex with max degree in P
        v_list = [i for i in range(1, N + 1) if (P >> i) & 1]
        v = max(v_list, key=lambda x: popcount(adj[x] & P))
        # branch: include v
        R.append(v)
        P_in = P & ~(1 << v) & ~adj[v]
        expand(R, P_in)
        R.pop()
        # branch: exclude v
        P_ex = P & ~(1 << v)
        expand(R, P_ex)

    expand([], ALL)

    if not best_set:
        best_set = [1]
    return best_set
# EVOLVE-BLOCK-END
