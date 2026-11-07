import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    start = time.time()
    DEADLINE = 85.0

    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    full = (1 << n) - 1
    nonadj = [0] * n
    for v in range(n):
        nonadj[v] = full ^ adj[v] ^ (1 << v)  # non-neighbors, excluding v

    N = 1 << n
    is_clique = [False] * N
    is_ind = [False] * N
    is_clique[0] = True
    is_ind[0] = True
    for mask in range(1, N):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_ind[mask] = is_ind[rest] and ((nonadj[v] & rest) == rest)

    def popcount(x):
        return bin(x).count("1")

    omega = 0
    alpha = 0
    for mask in range(N):
        c = popcount(mask)
        if is_clique[mask] and c > omega:
            omega = c
        if is_ind[mask] and c > alpha:
            alpha = c
    maxpart = max(omega, alpha)
    if maxpart < 1:
        maxpart = 1

    # suffix masks and lower bounds for remaining vertices {i..n-1}
    suffix_mask = [0] * (n + 1)
    for i in range(n, -1, -1):
        suffix_mask[i] = suffix_mask[i + 1] | (1 << i)
    suffix_lb = [0] * (n + 1)
    for i in range(n, -1, -1):
        m = suffix_mask[i]
        cnt = n - i
        if cnt == 0:
            suffix_lb[i] = 0
        else:
            size_lb = (cnt + maxpart - 1) // maxpart
            if is_clique[m] or is_ind[m]:
                struct_lb = 1
            else:
                struct_lb = 2 if cnt >= 2 else 1
            suffix_lb[i] = size_lb if size_lb > struct_lb else struct_lb

    parts_mask = []
    parts_clique = []
    parts_ind = []
    result = [None]

    def dfs(v, k):
        if result[0] is not None:
            return True
        if time.time() - start > DEADLINE:
            return False
        if v == n:
            res = []
            for pm in parts_mask:
                res.append([i for i in range(n) if (pm >> i) & 1])
            result[0] = res
            return True
        parts_used = len(parts_mask)
        if parts_used + suffix_lb[v] > k:
            return False
        # try joining an existing part (keeps it all-clique or all-independent)
        for p in range(parts_used):
            pm = parts_mask[p]
            nc = parts_clique[p] and ((adj[v] & pm) == pm)
            ni = parts_ind[p] and ((nonadj[v] & pm) == pm)
            if nc or ni:
                old_m, old_c, old_i = pm, parts_clique[p], parts_ind[p]
                parts_mask[p] = pm | (1 << v)
                parts_clique[p] = nc
                parts_ind[p] = ni
                if dfs(v + 1, k):
                    return True
                parts_mask[p], parts_clique[p], parts_ind[p] = old_m, old_c, old_i
        # open a new part (restricted-growth: next index only)
        if parts_used < k:
            parts_mask.append(1 << v)
            parts_clique.append(True)
            parts_ind.append(True)
            if dfs(v + 1, k):
                return True
            parts_mask.pop()
            parts_clique.pop()
            parts_ind.pop()
        return False

    for k in range(1, n + 1):
        if k < suffix_lb[0]:
            continue
        parts_mask.clear()
        parts_clique.clear()
        parts_ind.clear()
        if dfs(0, k):
            break
        if time.time() - start > DEADLINE:
            break

    if result[0] is None:
        result[0] = [[i] for i in range(n)]  # valid fallback: singletons

    return [[int(x) for x in part] for part in result[0]]
