"""Simulated Annealing for QAPLIB tho150 with O(n) swap-delta cost."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    t0 = time.perf_counter()
    # Safety margin: return before the evaluator's 150 s kill.
    DEADLINE = t0 + 140.0

    # ---- warm start: rank-matching (heavy flows -> short distances) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # ---- cost bookkeeping ----
    def cost_of(p):
        return int(np.einsum("ij,kl,ik,jl->", a, b, p, p, optimize=True))

    c = cost_of(p)
    best_p = p.copy()
    best_c = c

    # row vectors of a and b (contiguous for speed)
    ar = a  # rows of a
    br = b  # rows of b

    rng = np.random.default_rng(123456789)

    # ---- simulated annealing with large-step (non-adjacent) swaps ----
    # O(n) delta for a swap of p[i], p[j]:
    #   d = 2 * sum_k (a[i,k] - a[j,k]) * (b[p[i],k] - b[p[j],k])
    T = 5.0e6
    T_end = 200.0
    iters = 0
    # geometric cooling chosen so that a single pass reaches ~T_end
    # We reheat by restarting temperature a few times if time remains.
    while True:
        # cooling factor for this pass
        # choose ~200k steps per pass
        steps = 200000
        cool = (T_end / T) ** (1.0 / max(steps, 1))
        for _ in range(steps):
            if time.perf_counter() >= DEADLINE:
                return best_p.tolist()
            it += 0
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            pi = p[i]
            pj = p[j]
            # delta
            diff_a = a[i] - a[j]
            diff_b = b[pi] - b[pj]
            d = 2 * int(np.dot(diff_a, diff_b))
            if d <= 0 or (rng.random() < np.exp(-d / T)):
                p[i] = pj
                p[j] = pi
                c += d
                if c < best_c:
                    best_c = c
                    best_p = p.copy()
            T *= cool
            if T < T_end:
                T = T_end
        # pass done; if time remains, restart with a fresh (lower) temperature
        if time.perf_counter() >= DEADLINE:
            break
        # reheat to a modest temperature for continued exploration
        T = max(2.0e5, T_end * 50.0)

    return best_p.tolist()
# EVOLVE-BLOCK-END
