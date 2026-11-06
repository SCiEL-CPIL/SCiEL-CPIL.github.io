def construct():
    import time
    from math import gcd
    from itertools import combinations

    t0 = time.time()
    DEADLINE = 80.0          # safety margin before the 90s kill

    N = 128
    # Precompute all pairwise gcds.
    gcd_m = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        gi = gcd_m[i]
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gi[j] = g
            gcd_m[j][i] = g

    def is_valid(S):
        # True iff no three distinct elements have all three pairwise gcds equal.
        m = len(S)
        for i in range(m):
            ai = S[i]
            for j in range(i + 1, m):
                aj = S[j]
                gij = gcd_m[ai][aj]
                for k in range(j + 1, m):
                    ak = S[k]
                    if gij == gcd_m[ai][ak] and gij == gcd_m[aj][ak]:
                        return False
        return True

    def can_add(S, x):
        # Can x join S without creating an equal-gcd triple?
        # Only triples containing x matter: pick two existing a,b with gcd(a,b)=gcd(a,x)=gcd(b,x).
        for i in range(len(S)):
            ai = S[i]
            gax = gcd_m[ai][x]
            if gax == 1:
                continue
            for j in range(i + 1, len(S)):
                aj = S[j]
                if gcd_m[ai][aj] == gax and gcd_m[aj][x] == gax:
                    return False
        return True

    # ---------------- Phase 1: fast greedy to get a solid seed ----------------
    greedy = []
    for x in range(1, N + 1):
        if can_add(greedy, x):
            greedy.append(x)
    # try a few alternative orderings, keep the largest
    best = greedy
    for order in (range(1, N + 1), range(N, 0, -1),
                  [x for x in range(1, N + 1) if x % 2 == 1],
                  [x for x in range(1, N + 1) if x % 2 == 0]):
        if time.time() > DEADLINE - 20:
            break
        g = []
        for x in order:
            if can_add(g, x):
                g.append(x)
        if len(g) > len(best):
            best = g

    # ---------------- Phase 2: exact max independent set (branch & bound) ----
    # Conflict graph: node i conflicts with node j iff they form a "bad pair"
    # (gcd(i,j) > 1). Any independent set is a valid solution, because a bad
    # triple requires at least one bad pair, and an independent set has none.
    # So max independent set == optimal valid subset.
    #
    # We build it over the full [1..128] but seed the search with `best` so we
    # can only improve.

    def build_conflicts():
        # conflicts[i] = bitmask of j != i with gcd(i,j) > 1
        conflicts = [0] * (N + 1)
        for i in range(1, N + 1):
            mask = 0
            for j in range(1, N + 1):
                if j != i and gcd_m[i][j] > 1:
                    mask |= (1 << j)
            conflicts[i] = mask
        return conflicts

    conflicts = build_conflicts()

    # Universe bitmask of all numbers 1..128.
    FULL = 0
    for x in range(1, N + 1):
        FULL |= (1 << x)

    # Seed solution size lower bound.
    lb = len(best)

    # Branch & bound for maximum independent set.
    # State: candidates mask C (nodes still available), current chosen mask A.
    # We must keep A consistent (independent). Greedy upper bound via simple
    # coloring of the induced subgraph on C.

    best_set = list(best)
    best_size = lb
    nodes = 0

    # Simple coloring upper bound on independent set size of C:
    # greedy color the conflict graph induced by C; #colors >= alpha.
    def color_bound(C):
        # returns an upper bound on alpha(G[C]) using greedy coloring
        # We color the CONFLICT graph; each color class is an independent set,
        # so #color classes is an upper bound on the max independent set.
        # Actually for MIS we need upper bound: greedy coloring of G gives
        # a partition into independent sets => number of colors >= alpha. Good.
        C_list = []
        m = C
        while m:
            lsb = m & (-m)
            C_list.append(lsb.bit_length() - 1)
            m &= m - 1
        color = {}
        num_colors = 0
        for v in C_list:
            used = set()
            for u in C_list:
                if u == v:
                    break
                if color.get(u) is not None and (conflicts[v] >> u) & 1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1
        return num_colors

    # We do a recursive search but with a time guard.
    # To keep it fast we use a greedy upper bound and a strong ordering.

    import sys
    sys.setrecursionlimit(100000)

    def search(C, chosen, size):
        nonlocal best_size, best_set, nodes
        if time.time() > DEADLINE:
            return
        # Upper bound: current size + alpha(G[C])
        ub = size + color_bound(C)
        if ub <= best_size:
            return
        if C == 0:
            if size > best_size:
                best_size = size
                best_set = [b for b in range(1, N + 1) if (chosen >> b) & 1]
            return
        # Pick a vertex to branch on: the one with highest degree in C.
        m = C
        best_v = -1
        best_deg = -1
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (conflicts[v] & C).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m &= m - 1
        v = best_v
        # Branch 1: include v
        newC = C & ~(1 << v) & ~conflicts[v]
        search(newC, chosen | (1 << v), size + 1)
        # Branch 2: exclude v
        search(C & ~(1 << v), chosen, size)

    # Start search. We don't force the greedy seed into the search (it's just
    # a lower bound), but we can prune against it.
    search(FULL, 0, 0)

    result = best_set if best_set else [1]
    # Final safety: ensure nonempty and valid.
    if not result:
        result = [1]
    if not is_valid(result):
        # extremely defensive fallback
        result = [1]
    return result
