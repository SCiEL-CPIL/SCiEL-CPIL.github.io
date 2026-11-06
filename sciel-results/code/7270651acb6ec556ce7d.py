import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin vs 90s kill
    N = 512
    B = 512

    def lcm(a, b):
        return a // gcd(a, b) * b

    def remaining_ok():
        return time.time() < deadline - 2.0

    # compat[i] = bitmask of j (1..N) with lcm(i,j) <= B
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if lcm(i, j) <= B:
                mask |= (1 << j)
        compat[i] = mask

    # ---------- greedy baselines ----------
    def greedy_fill(mask):
        for v in range(1, N + 1):
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return mask

    def greedy_order(order):
        mask = 0
        for v in order:
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return greedy_fill(mask)

    best_mask = 0
    best_size = 0

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: bin(compat[x]).count('1'), reverse=True),
        sorted(range(1, N + 1), key=lambda x: bin(compat[x]).count('1')),
        sorted(range(1, N + 1), key=lambda x: x.bit_count()),
    ]
    for order in orders:
        if not remaining_ok():
            break
        m = greedy_order(order)
        s = m.bit_count()
        if s > best_size:
            best_size = s
            best_mask = m

    # ---------- exact max clique: Bron-Kerbosch + Tomita coloring ----------
    def bkc_coloring(P, R, best):
        # P, R: Python int bitmasks
        if P == 0:
            if R.bit_count() > best[0]:
                best[0] = R.bit_count()
                best[1] = R
            return
        # color sort on P
        U = P
        colors = []
        while U:
            C = U
            while C:
                # pick a vertex (lowest set bit)
                lsb = C & (-C)
                v = lsb.bit_length() - 1
                C &= C - 1
                # add v to color class: remove from U
                U &= ~lsb
                # remove v's neighbors from C (cannot share color)
                C &= ~compat[v]
            colors.append(v)  # representative of this color class
        # process in reverse color order
        for v in reversed(colors):
            if not remaining_ok():
                return
            bit = 1 << v
            if not (P & bit):
                continue
            P &= ~bit
            bkc_coloring(P & compat[v], R | bit, best)

    best = [best_size, best_mask]
    if remaining_ok():
        bkc_coloring(compat[1] | compat[2] | (1 << 1) | (1 << 2), 0, best)
        # simpler: start P = all vertices, R = 0
        allv = 0
        for i in range(1, N + 1):
            allv |= (1 << i)
        # (already covered by recursion; keep best)

    if best[1].bit_count() > best_size:
        best_size = best[1].bit_count()
        best_mask = best[1]

    # ---------- local improvement: swap to grow ----------
    def local_improve(mask):
        nonlocal best_size, best_mask
        improved = True
        while improved and remaining_ok():
            improved = False
            cur = mask.bit_count()
            clique = [v for v in range(1, N + 1) if mask & (1 << v)]
            for i in range(len(clique)):
                v = clique[i]
                base = mask & ~(1 << v)
                for w in range(1, N + 1):
                    if mask & (1 << w):
                        continue
                    if (base & ~compat[w]) == 0:
                        nm = greedy_fill(base | (1 << w))
                        if nm.bit_count() > cur:
                            mask = nm
                            cur = nm
                            if cur > best_size:
                                best_size = cur
                                best_mask = mask
                            improved = True
                            break
                if improved:
                    break
        return mask

    best_mask = local_improve(best_mask)

    result = [v for v in range(1, N + 1) if best_mask & (1 << v)]
    return result
