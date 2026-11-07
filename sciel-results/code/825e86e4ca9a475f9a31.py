# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    size = 1 << n
    full = size - 1

    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True

    for m in range(1, size):
        lb = m & (-m)
        v = lb.bit_length() - 1
        rest = m ^ lb
        is_clique[m] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_indep[m] = is_indep[rest] and ((adj[v] & rest) == 0)

    INF = 10**9
    dp = [INF] * size
    dp[0] = 0

    for m in range(1, size):
        best = INF
        s = m
        while s:
            if is_clique[s] or is_indep[s]:
                r = m ^ s
                cand = dp[r] + 1
                if cand < best:
                    best = cand
                    if best == 1:
                        break
            s = (s - 1) & m
        dp[m] = best

    # Reconstruct
    parts = []
    m = full
    while m:
        s = m
        while s:
            if (is_clique[s] or is_indep[s]) and dp[m] == dp[m ^ s] + 1:
                parts.append([v for v in range(n) if (s >> v) & 1])
                m ^= s
                break
            s = (s - 1) & m
    parts.reverse()

    # Safety verification; fall back to greedy if anything is off
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    valid = True
    used = []
    for part in parts:
        used.extend(part)
        k = len(part)
        if k == 0:
            valid = False
            break
        hits = 0
        for i in range(k):
            for j in range(i + 1, k):
                if (part[i], part[j]) in es:
                    hits += 1
        if not (hits == 0 or hits == k * (k - 1) // 2):
            valid = False
            break
    if not valid or len(used) != n or len(set(used)) != n:
        # Greedy fallback
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                ok = True
                for w in part:
                    e = (v, w) in es
                    if e:
                        # must be all edges
                        for w2 in part:
                            if w2 != v and (v, w2) not in es:
                                ok = False
                                break
                    else:
                        for w2 in part:
                            if w2 != v and (v, w2) in es:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])

    return [list(sorted([int(v) for v in part])) for part in parts]

# EVOLVE-BLOCK-END
