import time
import random
import numpy as np

def _is_c4_free(A):
    C = A @ A
    np.fill_diagonal(C, 0)
    return C.max() < 2

def _c4_free_delete_np(A, n, max_steps=50000):
    A = A.copy()
    deg = A.sum(axis=1)
    steps = 0
    while steps < max_steps:
        C = A @ A
        np.fill_diagonal(C, 0)
        m = C.max()
        if m < 2:
            break
        u, v = np.unravel_index(np.argmax(C), C.shape)
        common = np.where((A[u] == 1) & (A[v] == 1))[0]
        if len(common) < 2:
            break
        a, b = common[0], common[1]
        cand = [(u, a), (a, v), (v, b), (b, u)]
        best = max(cand, key=lambda e: deg[e[0]] + deg[e[1]])
        x, y = best
        A[x, y] = 0
        A[y, x] = 0
        deg[x] -= 1
        deg[y] -= 1
        steps += 1
    return A, deg

def _repair(A, n, target, max_add=3000):
    A = A.copy()
    deg = A.sum(axis=1)
    for _ in range(max_add):
        if deg.min() >= target:
            break
        low = int(np.argmin(deg))
        added = False
        for v in range(n):
            if v == low or A[low, v] == 1:
                continue
            A[low, v] = 1
            A[v, low] = 1
            C = A @ A
            np.fill_diagonal(C, 0)
            if C.max() >= 2:
                A[low, v] = 0
                A[v, low] = 0
            else:
                deg[low] += 1
                deg[v] += 1
                added = True
                break
        if not added:
            break
    return A, deg

def _to_result(n, A):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges.append([int(i), int(j)])
    return {'n': int(n), 'edges': edges}

def construct():
    t0 = time.time()
    deadline = t0 + 140
    caps = [46, 45, 44, 43, 42, 41, 40, 39]
    cap_secs = {46: 95, 45: 30, 44: 12, 43: 8, 42: 6, 41: 5, 40: 4, 39: 2}
    p_list = [0.3, 0.35, 0.4, 0.25, 0.45, 0.5, 0.2, 0.55]
    for n in caps:
        target = n - 39
        if target <= 0:
            return {'n': int(n), 'edges': []}
        n_start = time.time()
        seed = 0
        while (time.time() - n_start) < cap_secs.get(n, 5) and time.time() < deadline - 5:
            for p in p_list:
                if time.time() > deadline - 5:
                    break
                np.random.seed(1000003 * (seed + 1) + int(p * 1000))
                R = np.random.random((n, n))
                A = (R < p).astype(np.int32)
                A = np.maximum(A, A.T)
                np.fill_diagonal(A, 0)
                A, deg = _c4_free_delete_np(A, n)
                if _is_c4_free(A) and deg.min() >= target:
                    return _to_result(n, A)
                A2, deg2 = _repair(A, n, target)
                if _is_c4_free(A2) and deg2.min() >= target:
                    return _to_result(n, A2)
            seed += 1
    return {'n': 39, 'edges': []}
