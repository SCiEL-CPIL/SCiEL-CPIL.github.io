import time
import sys

sys.setrecursionlimit(100000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']

    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    full = (1 << n) - 1
    N = 1 << n

    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    for s in range(1, N):
        lsb = s & (-s)
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[s] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[s] = True
    is_valid = [is_clique[s] or is_indep[s] for s in range(N)]

    # valid parts grouped by their lowest (anchor) vertex
    valid_parts_by_min = [[] for _ in range(n)]
    for s in range(1, N):
        if is_valid[s]:
            lsb = s & (-s)
            v = lsb.bit_length() - 1
            valid_parts_by_min[v].append(s)

    # max clique size and max independent set size for every subset
    max_clique_sz = [0] * N
    max_indep_sz = [0] * N
    for s in range(1, N):
        lsb = s & (-s)
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        mc_rest = max_clique_sz[rest]
        mc_with = 1 + max_clique_sz[rest & adj[v]]
        max_clique_sz[s] = mc_rest if mc_rest > mc_with else mc_with
        mi_rest = max_indep_sz[rest]
        mi_with = 1 + max_indep_sz[rest & (~adj[v]) & full]
        max_indep_sz[s] = mi_rest if mi_rest > mi_with else mi_with

    def lower_bound(R):
        if R == 0:
            return 0
        cnt = R.bit_count()
        alpha = max_indep_sz[R]
        if alpha == 0:
            alpha = 1
        lb1 = max_clique_sz[R]
        lb2 = (cnt + alpha - 1) // alpha
        return lb1 if lb1 > lb2 else lb2

    deadline = time.time() + 85.0
    INF = 10 ** 9
    memo = {}

    def solve(R):
        if R == 0:
            return 0
        if time.time() > deadline:
            return None
        if R in memo:
            return memo[R]
        lb = lower_bound(R)
        lsb = R & (-R)
        v = lsb.bit_length() - 1
        best = INF
        for P in valid_parts_by_min[v]:
            if (P & R) != P:
                continue
            if 1 + lower_bound(R ^ P) >= best:
                continue
            sub = solve(R ^ P)
            if sub is None:
                return None
            cand = 1 + sub
            if cand < best:
                best = cand
                if best <= lb:
                    break
        if best >= INF:
            return None
        memo[R] = best
        return best

    result = solve(full)

    if result is not None and result < INF:
        parts = []
        R = full
        while R != 0:
            lsb = R & (-R)
            v = lsb.bit_length() - 1
            best = memo[R]
            chosen = None
            for P in valid_parts_by_min[v]:
                if (P & R) != P:
                    continue
                if 1 + memo[R ^ P] == best:
                    chosen = P
                    break
            if chosen is None:
                chosen = 1 << v
            parts.append(chosen)
            R ^= chosen
        final = []
        for mask in parts:
            verts = [int(i) for i in range(n) if mask & (1 << i)]
            final.append(verts)
        return final
    else:
        return [[i] for i in range(n)]
