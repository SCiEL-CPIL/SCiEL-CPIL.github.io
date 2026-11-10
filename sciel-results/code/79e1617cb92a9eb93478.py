import time
from math import gcd

_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()

    # vertices sorted by weight descending => a from 511 down to 1
    verts = list(range(511, 0, -1))
    n = len(verts)
    weights = [1.0 / (512 - verts[i]) for i in range(n)]

    # coprime_mask[i]: bitmask of j such that gcd(verts[i], verts[j]) == 1 (j != i)
    coprime_mask = [0] * n
    for i in range(n):
        m = 0
        vi = verts[i]
        for j in range(n):
            if i != j and gcd(vi, verts[j]) == 1:
                m |= (1 << j)
        coprime_mask[i] = m

    # greedy incumbent (take highest weight feasible)
    chosen_inc = []
    cm = 0
    for i in range(n):
        if (coprime_mask[i] & cm) == cm:
            chosen_inc.append(verts[i])
            cm |= (1 << i)
    s_inc = set(chosen_inc)
    best_w = sum(weights[i] for i in range(n) if verts[i] in s_inc)
    best_set = list(chosen_inc)

    chosen_mask = 0
    chosen_list = []

    def remaining_bound(i):
        s = 0.0
        cm = chosen_mask
        for j in range(i, n):
            if (coprime_mask[j] & cm) == cm:
                s += weights[j]
        return s

    def dfs(i, cur_w):
        if time.time() - _START > 110:
            return
        if cur_w + remaining_bound(i) <= best_w + 1e-12:
            return
        if i == n:
            if cur_w > best_w:
                best_w = cur_w
                best_set = list(chosen_list)
            return
        # include vertex i
        if (coprime_mask[i] & chosen_mask) == chosen_mask:
            chosen_mask |= (1 << i)
            chosen_list.append(verts[i])
            dfs(i + 1, cur_w + weights[i])
            chosen_list.pop()
            chosen_mask &= ~(1 << i)
        # exclude vertex i
        dfs(i + 1, cur_w)

    dfs(0, 0.0)
    return best_set
