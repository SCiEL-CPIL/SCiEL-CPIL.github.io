import itertools, random, time
import numpy as np

N = 16
ALL4 = list(itertools.combinations(range(N), 4))
A_idx = np.array(ALL4, dtype=np.int64)
A_masks = np.zeros((len(ALL4), N), dtype=np.int16)
for i, s in enumerate(ALL4):
    A_masks[i, s] = 1
AMT = A_masks.T


def has_cube(adj):
    M = np.empty((N, N), dtype=np.int16)
    for u in range(N):
        a = adj[u]
        for v in range(N):
            M[u, v] = (a >> v) & 1
    Y = M @ AMT
    n4 = len(ALL4)
    for i in range(n4):
        rows = A_idx[i]
        X = M[rows] @ AMT
        cond1 = np.all(X == 3, axis=0)
        if not cond1.any():
            continue
        cand = np.nonzero(cond1)[0]
        for j in cand:
            if j <= i:
                continue
            if np.all(Y[A_idx[j], i] == 3):
                return True
    return False


def construct():
    t0 = time.time()
    DEADLINE = 110.0
    best = 0
    best_adj = None
    rng = random.Random(20240517)

    def ec(adj):
        return sum(adj[u].bit_count() for u in range(N)) // 2

    def try_greedy():
        adj = [0] * N
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(edges)
        for (u, v) in edges:
            if time.time() - t0 > DEADLINE - 3:
                return adj
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            if has_cube(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def local_search(adj):
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        improved = True
        while improved and time.time() - t0 < DEADLINE - 3:
            improved = False
            rng.shuffle(edges)
            for (u, v) in edges:
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if has_cube(adj):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    improved = True
        return adj

    while time.time() - t0 < DEADLINE - 3:
        adj = try_greedy()
        adj = local_search(adj)
        c = ec(adj)
        if c > best:
            best = c
            best_adj = adj[:]

    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    return [[u, v] for u in range(N) for v in range(u + 1, N) if (best_adj[u] >> v) & 1]
