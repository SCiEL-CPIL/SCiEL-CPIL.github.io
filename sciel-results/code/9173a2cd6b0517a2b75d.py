"""Pairwise disjoint residue classes: maximize sum 1/m via MWIS on a conflict graph."""
import time
import math
import sys

_DEADLINE = 150.0 - 3.0
_START = None


def _time_left():
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)


def _gcd(a, b):
    return math.gcd(a, b)


def _build_graph():
    pairs = []
    for m in range(20, 201):
        for r in range(m):
            pairs.append((m, r))
    n = len(pairs)
    M = [p[0] for p in pairs]
    R = [p[1] for p in pairs]
    w = [1.0 / m for m in M]

    nbr = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if M[i] == M[j]:
                conflict = True
            else:
                g = _gcd(M[i], M[j])
                conflict = (R[i] % g) == (R[j] % g)
            if conflict:
                nbr[i] |= (1 << j)
                nbr[j] |= (1 << i)

    # cand[i]: nodes compatible with i (includes i, excludes i's neighbors)
    allmask = (1 << n) - 1
    cand = [allmask & ~nbr[i] for i in range(n)]
    return pairs, M, R, w, nbr, cand, allmask


def _greedy_simple(w, nbr, order):
    chosen_mask = 0
    total = 0.0
    for i in order:
        if chosen_mask & nbr[i]:
            continue
        chosen_mask |= (1 << i)
        total += w[i]
    return total, chosen_mask


def _validate(pairs, chosen_mask):
    idx = [i for i in range(len(pairs)) if (chosen_mask >> i) & 1]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            m1, r1 = pairs[i]
            m2, r2 = pairs[j]
            g = _gcd(m1, m2)
            if (r1 % g) == (r2 % g):
                return False
    return True


def _branch_and_bound(w, nbr, cand, order_wdesc, init_total, init_mask):
    best_total = init_total
    best_mask = init_mask
    n = len(w)
    sys.setrecursionlimit(1000000)

    def popcount_sum(mask):
        s = 0.0
        mm = mask
        while mm:
            lsb = mm & (-mm)
            i = lsb.bit_length() - 1
            s += w[i]
            mm ^= lsb
        return s

    def dfs(cand_mask, chosen_mask, cur_sum):
        nonlocal best_total, best_mask
        if _time_left() <= 0:
            return
        if cand_mask == 0:
            if cur_sum > best_total:
                best_total = cur_sum
                best_mask = chosen_mask
            return
        ub = cur_sum + popcount_sum(cand_mask)
        if ub <= best_total + 1e-12:
            return
        # pick highest-weight node in cand_mask
        mm = cand_mask
        best_node = -1
        best_w = -1.0
        while mm:
            lsb = mm & (-mm)
            i = lsb.bit_length() - 1
            if w[i] > best_w:
                best_w = w[i]
                best_node = i
            mm ^= lsb
        i = best_node
        bit = 1 << i
        # include i
        new_cand = cand_mask & cand[i]
        new_cand &= ~bit
        dfs(new_cand, chosen_mask | bit, cur_sum + w[i])
        # exclude i
        dfs(cand_mask & ~bit, chosen_mask, cur_sum)

    dfs((1 << n) - 1, 0, 0.0)
    return best_total, best_mask


def construct():
    global _START
    _START = time.time()

    pairs, M, R, w, nbr, cand, allmask = _build_graph()
    n = len(pairs)

    order_wdesc = sorted(range(n), key=lambda i: -w[i])
    order_wasc = sorted(range(n), key=lambda i: w[i])
    order_masc = sorted(range(n), key=lambda i: (M[i], R[i]))

    best_total = 0.0
    best_mask = 0
    for ord_list in (order_wdesc, order_wasc, order_masc):
        t, m = _greedy_simple(w, nbr, ord_list)
        if t > best_total:
            best_total = t
            best_mask = m

    if not _validate(pairs, best_mask):
        best_mask = 0
        best_total = 0.0
        for i in range(n):
            if not (best_mask & nbr[i]):
                best_mask |= (1 << i)
                best_total += w[i]

    if _time_left() > 0.5:
        bt, bm = _branch_and_bound(w, nbr, cand, order_wdesc, best_total, best_mask)
        if bt > best_total:
            best_total = bt
            best_mask = bm
        if not _validate(pairs, best_mask):
            from fractions import Fraction
            s = Fraction(0)
            for i in range(n):
                if (best_mask >> i) & 1:
                    s += Fraction(1, pairs[i][0])
            best_total = float(s)

    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append([pairs[i][0], pairs[i][1]])
    return result
