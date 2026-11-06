import time
import random
import numpy as np

_DEADLINE = 150.0
_SAFETY = 2.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _edge_index(n, i, j):
    # i < j ; count pairs (a,b), a<b, in lex order up to (i,j)
    # number of pairs with first coord < i : sum_{a=0}^{i-1} (n-1-a) = i*(n-1) - i*(i-1)//2
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _edge_decompose(n, e):
    # find i,j for edge index e
    # i*(n-1) - i*(i-1)//2 <= e < (i+1)*(n-1) - (i+1)*i//2
    lo, hi = 0, n - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * (n - 1) - mid * (mid - 1) // 2 <= e:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    base = i * (n - 1) - i * (i - 1) // 2
    j = i + 1 + (e - base)
    return i, j

def _find_c6s(adj):
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    c4 = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        if deg[i] < 2:
            continue
        A = adj[i]
        c4[i] = A @ A - np.diag(deg)
    total = 0
    for i in range(n):
        if deg[i] < 2:
            continue
        Ai = adj[i]
        total += int((c4 * Ai).sum() - c4[i].sum())
    return total // 12

def _c6_total(adjs):
    return sum(_find_c6s(a) for a in adjs)

def _init_spectral(n, rng):
    if n <= 4:
        return [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    d = min(n - 1, 2 * int(np.sqrt(n)) + 1)
    if d % 2:
        d -= 1
    if d < 2:
        d = 2
    d = min(d, n - 1)
    if d % 2:
        d -= 1
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        rng.shuffle(pool)
        A[i, pool[:d]] = 1
    A = (A + A.T) // 2
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1).astype(np.float64))
    L = D - A.astype(np.float64)
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    top = idx[2:6] if n >= 5 else idx[1:5]
    V = vecs[:, top]
    sig = np.zeros((n, 4), dtype=np.int8)
    for k in range(V.shape[1]):
        col = V[:, k]
        med = np.median(col)
        sig[:, k] = (col > med).astype(np.int8)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            x = int(sig[i, 0] ^ sig[j, 0])
            y = int(sig[i, 1] ^ sig[j, 1])
            u = int(sig[i, 2] ^ sig[j, 2])
            v = int(sig[i, 3] ^ sig[j, 3])
            colors.append(((x << 3) | (y << 2) | (u << 1) | v) % 4)
    return colors

def _build_adj(n, colors):
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    for e, c in enumerate(colors):
        i, j = _edge_decompose(n, e)
        adj[c][i, j] = 1
        adj[c][j, i] = 1
    return adj

def _improve(n, colors, rng):
    colors = list(colors)
    m = n * (n - 1) // 2
    adj = _build_adj(n, colors)
    deg = [adj[c].sum(axis=1) for c in range(4)]
    c4 = [np.zeros((n, n), dtype=np.int64) for _ in range(4)]
    for c in range(4):
        A = adj[c]
        for i in range(n):
            if deg[c][i] >= 2:
                c4[c][i] = A @ A - np.diag(deg[c])
    def total_c6():
        t = 0
        for c in range(4):
            A = adj[c]
            dc = deg[c]
            for i in range(n):
                if dc[i] >= 2:
                    t += int((c4[c] * A).sum() - c4[c][i].sum())
        return t // 12
    def delta_c6(c, i, j, adding):
        s = 0
        Ac = adj[c]
        for k in range(n):
            if k == i or k == j:
                continue
            if Ac[i, k] and Ac[j, k]:
                s += int(c4[c][k, i] + c4[c][k, j])
        return s if adding else -s
    best = list(colors)
    best_c6 = total_c6()
    if best_c6 == 0:
        return best, 0
    temp = 0.1
    cur = list(colors)
    cur_c6 = best_c6
    while _remaining() > 0.5:
        e = rng.randrange(m)
        i, j = _edge_decompose(n, e)
        c = cur[e]
        nc = rng.randrange(4)
        if nc == c:
            continue
        dc6 = delta_c6(nc, i, j, True) + delta_c6(c, i, j, False)
        newc6 = cur_c6 + dc6
        if newc6 <= cur_c6:
            acc = True
        else:
            acc = rng.random() < np.exp(-dc6 / max(temp, 1e-6))
        if acc:
            Ac, An = adj[c], adj[nc]
            for k in range(n):
                if k != i and k != j:
                    if Ac[i, k] and Ac[j, k]:
                        c4[c][k, i] -= 1
                        c4[c][k, j] -= 1
                        c4[c][i, k] -= 1
                        c4[c][j, k] -= 1
                    if An[i, k] and An[j, k]:
                        c4[nc][k, i] += 1
                        c4[nc][k, j] += 1
                        c4[nc][i, k] += 1
                        c4[nc][j, k] += 1
            Ac[i, j] = 0
            Ac[j, i] = 0
            An[i, j] = 1
            An[j, i] = 1
            deg[c][i] -= 1
            deg[c][j] -= 1
            deg[nc][i] += 1
            deg[nc][j] += 1
            c4[c][i] -= Ac[i]
            c4[c][j] -= Ac[j]
            c4[nc][i] += An[i]
            c4[nc][j] += An[j]
            cur[e] = nc
            cur_c6 = newc6
            if cur_c6 < best_c6:
                best_c6 = cur_c6
                best = list(cur)
                if best_c6 == 0:
                    return best, 0
            temp *= 0.99
        else:
            temp *= 0.999
    return best, best_c6

def construct():
    global _START
    _START = time.time()
    rng = random.Random(12345)
    best_colors = None
    best_n = 0
    for n in range(17, 20):
        if _remaining() < 3.0:
            break
        init = _init_spectral(n, rng)
        adj = _build_adj(n, init)
        if _c6_total(adj) == 0:
            best_colors, best_n = init, n
            continue
        improved, c6 = _improve(n, init, rng)
        if c6 == 0:
            best_colors, best_n = improved, n
    if best_colors is None:
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    return {'n': best_n, 'colors': best_colors}
