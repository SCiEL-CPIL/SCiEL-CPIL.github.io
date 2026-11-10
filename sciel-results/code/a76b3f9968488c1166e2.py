import time

_DEADLINE = 150.0
_start = time.time()

def _time_left():
    return _DEADLINE - 6.0 - (time.time() - _start)   # 6 s safety margin


def _diffs(row):
    """Bitset of all positive differences in a row."""
    m = len(row)
    mask = 0
    for i in range(m):
        ri = row[i]
        for j in range(i + 1, m):
            mask |= 1 << (row[j] - ri)
    return mask


def construct(n_rows, k):
    global _start
    _start = time.time()
    k1 = k + 1
    need = k1 * (k1 - 1) // 2

    def time_left():
        return _DEADLINE - 6.0 - (time.time() - _start)

    # ---- Greedy baseline (always valid, quick) ----
    def greedy():
        used, rows = 0, []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k1:
                x += 1
                m = 0
                for a in row:
                    m |= 1 << (x - a)
                if not (m & used):
                    row.append(x)
                    used |= m
            rows.append(row)
        return rows

    best = greedy()
    best_m = max(best[-1])

    # ---- Independent-set search over candidate rows for a given M ----
    def solve_for_M(M):
        cands = []
        for r in range(1, M + 1):
            for a in range(r + 1, M + 1):
                for b in range(a + 1, M + 1):
                    for c in range(b + 1, M + 1):
                        for d in range(c + 1, M + 1):
                            for e in range(d + 1, M + 1):
                                cands.append((0, r, a, b, c, d, e))
        masks = [_diffs(r) for r in cands]
        n = len(cands)
        if n < n_rows:
            return None
        # conflict bitsets: conflict[i] has bit j set if masks share a difference
        conflict = [0] * n
        for i in range(n):
            mi = masks[i]
            ci = 0
            for j in range(n):
                if i != j and (mi & masks[j]):
                    ci |= 1 << j
            conflict[i] = ci
        # degree order
        deg = [conflict[i].bit_count() for i in range(n)]
        order = sorted(range(n), key=lambda i: deg[i])
        pos = {i: p for p, i in enumerate(order)}
        ord_cand = [order[p] for p in range(n)]
        ord_deg = [deg[order[p]] for p in range(n)]
        ord_conf = [conflict[order[p]] for p in range(n)]

        best_set = []

        def backtrack(p, chosen, avail):
            nonlocal best_set
            if len(chosen) == n_rows:
                best_set = list(chosen)
                return True
            rem = n - p
            if len(chosen) + rem < n_rows:
                return False
            while p < n and not (avail >> p) & 1:
                p += 1
            while p < n and (avail >> p) & 1:
                if len(chosen) + (n - p) < n_rows:
                    return False
                i = ord_cand[p]
                chosen.append(i)
                new_avail = avail & ~(ord_conf[p] | (1 << p))
                if backtrack(p + 1, chosen, new_avail):
                    return True
                chosen.pop()
                p += 1
            return False

        if backtrack(0, [], (1 << n) - 1):
            return [cands[i] for i in best_set]
        return None

    # ---- Iterative deepening on M ----
    M = 105
    while M < 1000 and time_left() > 0.5:
        res = solve_for_M(M)
        if res is not None:
            res.sort(key=lambda r: r[-1])
            if res[-1][-1] < best_m:
                best = res
                best_m = res[-1][-1]
            return best
        M += 1
    return best
