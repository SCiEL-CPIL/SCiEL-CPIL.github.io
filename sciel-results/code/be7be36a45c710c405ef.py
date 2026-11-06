import time
import random
import numpy as np


def construct():
    n = 20
    start = time.time()
    deadline = start + 145.0

    # Precompute all edges and upper-triangle index arrays
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    E = len(all_edges)
    u_idx = np.array([u for u, v in all_edges], dtype=np.int64)
    v_idx = np.array([v for u, v in all_edges], dtype=np.int64)
    both_idx = np.stack([u_idx, v_idx], axis=1)  # (E, 2)

    # Precompute for each edge the set of other vertices (not its endpoints)
    all_mask = np.ones(n, dtype=np.int64)
    other_mask = all_mask[None, :] - both_idx  # (E, n): 1 where vertex is not an endpoint

    # Common-neighbor incidence: for each edge e and vertex w, how many of the two
    # endpoints are adjacent to w.  A K3,3 appears iff some edge has a vertex with value 2.
    def common_incidence(A):
        return A[both_idx[:, 0], :] + A[both_idx[:, 1], :]  # (E, n)

    def is_valid(A):
        inc = common_incidence(A)
        inc = inc * other_mask
        return not np.any(inc >= 2)

    def count_edges(A):
        return int(A.sum() // 2)

    def add_edge(A, i, j):
        A[i, j] = 1
        A[j, i] = 1

    def try_add(A, i, j):
        if A[i, j]:
            return False
        A[i, j] = 1
        A[j, i] = 1
        inc = common_incidence(A)
        bad = (inc >= 2) & other_mask
        if np.any(bad):
            A[i, j] = 0
            A[j, i] = 0
            return False
        return True

    # ---- initialization: bipartite spectral ----
    def init_bipartite(rng, n1=10, n2=10):
        A = np.zeros((n, n), dtype=np.int64)
        W = rng.random((n1, n2))
        W = (W + W.T) / 2.0
        w = np.linalg.eigh(W)[1][:, -1]
        thr = np.median(w)
        for a in range(n1):
            for b in range(n2):
                if w[a] * w[b] > thr:
                    A[a, b] = 1
                    A[b, a] = 1
        return A

    def init_random_dense(rng, n1=10, n2=10, p=0.6):
        A = np.zeros((n, n), dtype=np.int64)
        for a in range(n1):
            for b in range(n2):
                if rng.random() < p:
                    A[a, b] = 1
                    A[b, a] = 1
        return A

    def init_complete_bip(rng, n1=10, n2=10):
        A = np.zeros((n, n), dtype=np.int64)
        for a in range(n1):
            for b in range(n2):
                A[a, b] = 1
                A[b, a] = 1
        return A

    def greedy_fill(A, rng, max_iter=200):
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            order = rng.permutation(E)
            added = False
            for e in order:
                i = int(u_idx[e]); j = int(v_idx[e])
                if not A[i, j] and try_add(A, i, j):
                    added = True
            if not added:
                break
        return A

    def local_search(A, rng, max_rounds=1000):
        best = A.copy()
        best_count = count_edges(best)
        for _ in range(max_rounds):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for _ in range(200):
                if time.time() > deadline - 1.0:
                    break
                e1 = int(rng.randint(0, E - 1))
                i1, j1 = int(u_idx[e1]), int(v_idx[e1])
                if not best[i1, j1]:
                    continue
                trial = best.copy()
                trial[i1, j1] = 0
                trial[j1, i1] = 0
                # try to add a different edge
                for _ in range(3):
                    e2 = int(rng.randint(0, E - 1))
                    if e2 == e1:
                        continue
                    i2, j2 = int(u_idx[e2]), int(v_idx[e2])
                    if trial[i2, j2]:
                        continue
                    trial[i2, j2] = 1
                    trial[j2, i2] = 1
                    inc = common_incidence(trial)
                    if not np.any((inc >= 2) & other_mask):
                        break
                    trial[i2, j2] = 0
                    trial[j2, i2] = 0
                else:
                    continue
                # greedily fill trial
                order = rng.permutation(E)
                for e in order:
                    a = int(u_idx[e]); b = int(v_idx[e])
                    if not trial[a, b]:
                        trial[a, b] = 1
                        trial[b, a] = 1
                        inc = common_incidence(trial)
                        if np.any((inc >= 2) & other_mask):
                            trial[a, b] = 0
                            trial[b, a] = 0
                c = count_edges(trial)
                if c > best_count:
                    best = trial
                    best_count = c
                    improved = True
            if not improved:
                # kick: remove a random edge, refill
                e1 = int(rng.randint(0, E - 1))
                i1, j1 = int(u_idx[e1]), int(v_idx[e1])
                if best[i1, j1]:
                    trial = best.copy()
                    trial[i1, j1] = 0
                    trial[j1, i1] = 0
                    order = rng.permutation(E)
                    for e in order:
                        a = int(u_idx[e]); b = int(v_idx[e])
                        if not trial[a, b]:
                            trial[a, b] = 1
                            trial[b, a] = 1
                            inc = common_incidence(trial)
                            if np.any((inc >= 2) & other_mask):
                                trial[a, b] = 0
                                trial[b, a] = 0
                    c = count_edges(trial)
                    if c > best_count:
                        best = trial
                        best_count = c
                        improved = True
                if not improved:
                    break
        return best, best_count

    rng = random.Random(12345)
    best_A = None
    best_count = -1

    def consider(A):
        nonlocal best_A, best_count
        if not is_valid(A):
            return
        c = count_edges(A)
        if c > best_count:
            best_count = c
            best_A = A.copy()

    # initial seeds
    for _ in range(6):
        A = init_bipartite(rng)
        A = greedy_fill(A, rng)
        consider(A)
    for _ in range(4):
        A = init_random_dense(rng, p=0.5)
        A = greedy_fill(A, rng)
        consider(A)
    A = init_complete_bip(rng)
    A = greedy_fill(A, rng)
    consider(A)

    # local search from best seed
    if best_A is not None:
        A, c = local_search(best_A, rng)
        if c > best_count:
            best_count = c
            best_A = A

    # further restarts
    while time.time() < deadline - 3.0:
        A = init_bipartite(rng)
        A = greedy_fill(A, rng)
        c0 = count_edges(A)
        if c0 > best_count:
            best_count = c0
            best_A = A.copy()
        if time.time() < deadline - 3.0:
            A, c = local_search(A, rng)
            if c > best_count:
                best_count = c
                best_A = A

    if best_A is None:
        return []

    result = []
    for e in range(E):
        i = int(u_idx[e]); j = int(v_idx[e])
        if best_A[i, j]:
            result.append([i, j])
    return result
