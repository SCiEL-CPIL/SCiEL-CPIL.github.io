# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims."""
    n = M.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian assignment."""
    from scipy.optimize import linear_sum_assignment
    n = X.shape[0]
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _swap_delta(a, b, p, i, j):
    """Compute cost delta when swapping p[i] and p[j]. O(n) via 4 dot products."""
    li = p[i]
    lj = p[j]
    # Terms where i or j is first index
    old_first = np.dot(a[i, :], b[li, p]) + np.dot(a[j, :], b[lj, p])
    new_first = np.dot(a[i, :], b[lj, p]) + np.dot(a[j, :], b[li, p])
    # Terms where i or j is second index
    old_second = np.dot(a[:, i], b[p, li]) + np.dot(a[:, j], b[p, lj])
    new_second = np.dot(a[:, i], b[p, lj]) + np.dot(a[:, j], b[p, li])
    # Subtract double-counted diagonal terms
    old_diag = a[i, i] * b[li, li] + a[i, j] * b[li, lj] + a[j, i] * b[lj, li] + a[j, j] * b[lj, lj]
    new_diag = a[i, i] * b[lj, lj] + a[i, j] * b[lj, li] + a[j, i] * b[li, lj] + a[j, j] * b[li, li]
    return (new_first + new_second - new_diag) - (old_first + old_second - old_diag)


def _local_search(a, b, p, time_limit):
    """First-improvement 2-opt local search with O(n) per-swap delta."""
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()

    start = time.time()
    while True:
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += d
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_p, best_cost


def _perturb(p, n_swaps):
    """Randomly swap n_swaps pairs in p."""
    p = p.copy()
    n = len(p)
    for _ in range(n_swaps):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 115

    # Step 1: MDS embedding + Procrustes alignment
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    k = min(40, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)
    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    # Step 2: Local search polish
    time_left = deadline - time.time()
    if time_left > 10:
        p, cost = _local_search(a, b, p, time_left)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # Step 3: Multiple restarts with perturbation
    time_left = deadline - time.time()
    while time_left > 20:
        n_swaps = max(5, n // 10)
        p_pert = _perturb(best_p, n_swaps)
        time_left = deadline - time.time()
        if time_left < 15:
            break
        p, cost = _local_search(a, b, p_pert, time_left - 5)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        time_left = deadline - time.time()

    return best_p.tolist()
# EVOLVE-BLOCK-END
