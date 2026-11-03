"""QAP tho150: local search with O(n) swap-delta, SA restarts, 2-opt polish, deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _qap_cost(a, b, p):
    pb = b[np.ix_(p, p)]
    return int(np.sum(a * pb))


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 118.0

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # initial permutation: heavy flow -> short distance
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    best_p = p.copy()
    best_c = _qap_cost(a, b, best_p)

    # maintain B[i,j] = b[p[i],p[j]]
    B = b[np.ix_(p, p)].copy()

    def swap_delta(i, j):
        # delta = 2*sum_k (a[i,k]-a[j,k])*(B[k,j]-B[k,i]) + (a[i,i]-a[i,j]-a[j,i]+a[j,j])*(B[i,i]-B[i,j]-B[j,i]+B[j,j])
        diff_a = a[i, :] - a[j, :]
        diff_B = B[:, j] - B[:, i]
        main = 2.0 * np.dot(diff_a, diff_B)
        diag = (a[i, i] - a[i, j] - a[j, i] + a[j, j])
        diag_B = (B[i, i] - B[i, j] - B[j, i] + B[j, j])
        return main + diag * diag_B

    def do_swap(i, j):
        pi, pj = p[i], p[j]
        p[i], p[j] = pj, pi
        # update B: B[i,:] and B[j,:] swap; B[:,i] and B[:,j] swap
        B[i, :], B[j, :] = B[j, :], B[i, :]
        B[:, i], B[:, j] = B[:, j], B[:, i]

    def two_opt(p, B, max_passes=60):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            # vectorized delta over all pairs
            # main[i,j] = 2*( a[i,:]@B[:,j] - a[i,:]@B[:,i] - a[j,:]@B[:,j] + a[j,:]@B[:,i] )
            Ab = a @ B  # (n,n)
            idx = np.arange(n)
            main = 2.0 * (Ab - Ab[idx, idx][:, None] - Ab.T[idx, idx][:, None] + Ab.T)
            # diag_a[i,j] = a[i,i]-a[i,j]-a[j,i]+a[j,j]
            diag_a = a[idx, idx][:, None] - a - a.T + a[idx, idx][None, :]
            diag_B = B[idx, idx][:, None] - B - B.T + B[idx, idx][None, :]
            delta = main + diag_a * diag_B
            np.fill_diagonal(delta, 0.0)
            k = int(np.argmin(delta))
            i, j = divmod(k, n)
            d = delta[i, j]
            if d < -1e-9:
                do_swap(i, j)
                improved = True
        return p, B

    # initial 2-opt
    p, B = two_opt(p, B)
    c = _qap_cost(a, b, p)
    if c < best_c:
        best_c = c
        best_p = p.copy()

    # SA restarts
    rng = np.random.default_rng(11)
    t0, t1 = 3000.0, 1.0
    iters = 25000
    while time.time() < deadline - 2.0:
        p = best_p.copy()
        B = b[np.ix_(p, p)].copy()
        # perturb
        for _ in range(4):
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i != j:
                do_swap(i, j)
        # SA
        T = t0
        cur_c = _qap_cost(a, b, p)
        for it in range(iters):
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i == j:
                continue
            d = swap_delta(i, j)
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                do_swap(i, j)
                cur_c += d
                if cur_c < best_c:
                    best_c = cur_c
                    best_p = p.copy()
            T *= 0.99995
        # 2-opt polish
        p, B = two_opt(p, B)
        c = _qap_cost(a, b, p)
        if c < best_c:
            best_c = c
            best_p = p.copy()

    # final polish
    p = best_p.copy()
    B = b[np.ix_(p, p)].copy()
    p, B = two_opt(p, B)
    c = _qap_cost(a, b, p)
    if c < best_c:
        best_c = c
        best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
