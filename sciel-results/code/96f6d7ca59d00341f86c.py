import numpy as np
import itertools
import time
import sys

D = 11

def _build_vectors():
    v4 = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0]*D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4.append(v)
    v1 = []
    for pos in range(D):
        for sign in (1, -1):
            v = [0]*D
            v[pos] = 2*sign
            v1.append(v)
    return np.array(v4, dtype=np.int8), np.array(v1, dtype=np.int8)

def construct():
    start = time.time()
    deadline = start + 140.0
    sys.setrecursionlimit(1000000)

    V4, V1 = _build_vectors()
    n = V4.shape[0]

    dots = (V4 @ V4.T).astype(np.int8)
    compat = (dots <= 2)
    np.fill_diagonal(compat, False)

    adj = [0]*n
    for i in range(n):
        idx = np.nonzero(compat[i])[0]
        mask = 0
        for j in idx:
            mask |= (1 << int(j))
        adj[i] = mask

    def lsb_index(x):
        return (x & -x).bit_length() - 1
    def bit(i):
        return 1 << i

    # degree in conflict graph
    deg_conf = np.empty(n, dtype=np.int64)
    for i in range(n):
        deg_conf[i] = (n - 1) - bin(adj[i]).count('1')

    def greedy(order, tlimit):
        P = (1 << n) - 1
        chosen = []
        for v in order:
            if time.time() > tlimit:
                break
            if (P >> v) & 1:
                chosen.append(v)
                P &= ~adj[v]
                P &= ~(1 << v)
        return chosen

    best = []
    rng = np.random.default_rng(1)
    orders = [list(np.argsort(deg_conf)), list(np.argsort(-deg_conf))]
    for _ in range(6):
        orders.append(list(rng.permutation(n)))
    for o in orders:
        c = greedy(o, deadline - 80.0)
        if len(c) > len(best):
            best = c
    best_size = len(best)
    best_clique = list(best)

    def color_sort(P):
        U = P
        seq = []
        c = 0
        while U:
            c += 1
            Q = U
            while Q:
                v = lsb_index(Q)
                seq.append((v, c))
                b = 1 << v
                Q &= ~b
                U &= ~b
                Q &= ~adj[v]
        seq.sort(key=lambda t: t[1])
        return seq

    def expand(R, P, path):
        nonlocal best_size, best_clique
        if P == 0:
            if R > best_size:
                best_size = R
                best_clique = list(path)
            return
        seq = color_sort(P)
        for v, c in reversed(seq):
            if R + c <= best_size:
                break
            if time.time() > deadline:
                return
            path.append(v)
            expand(R + 1, P & adj[v], path)
            path.pop()
            P &= ~(1 << v)

    expand(0, (1 << n) - 1, [])

    # augment: add any compatible vertex to fixed point
    chosen = set(best_clique)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if v in chosen:
                continue
            row = compat[v]
            if row[list(chosen)].all():
                chosen.add(v)
                improved = True
                break

    result_v4 = V4[np.array(list(chosen), dtype=np.int64)]
    final = np.vstack([result_v4, V1]).astype(np.int64)
    return final
