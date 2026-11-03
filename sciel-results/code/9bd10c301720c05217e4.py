"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 112.0   # safety margin below the 150s kill

    # ---- Rank-matching start: heavy flows -> short distances ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    B = b.T  # B[i, j] = b[j, i], so B[p[i], p[j]] = b[p[j], p[i]]
    A = a

    def cost():
        return int(np.sum(A * b[np.ix_(p, p)]))

    best = p.copy()
    best_c = cost()

    # ---- Local search: first-improvement pair-swap passes ----
    def improve():
        nonlocal best_c
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return
                pi = p[i]
                Ai = A[i]
                for j in range(i + 1, n):
                    pj = p[j]
                    Aj = A[j]
                    dij = B[pi, pj]
                    dpj = B[pj, pi]
                    s = dij + dpj
                    d = (-Ai[j] * s
                         + (Ai[i] + Aj[j]) * (B[pi, pi] + B[pj, pj])
                         + (Ai[j] + Aj[i]) * s)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        best_c += int(d)
                        improved = True
        if best_c < cost():
            pass
        # recompute exactly to guard against any drift
        c = cost()
        if c < best_c:
            best_c = c
            best = p.copy()

    improve()

    # ---- Iterated local search with random perturbation ----
    if time.time() < deadline:
        p = best.copy()
        best_c = cost()
        while time.time() < deadline:
            # perturb: swap a random pair
            i = int(np.random.randint(n))
            j = int(np.random.randint(n))
            if i != j:
                p[i], p[j] = p[j], p[i]
            improve()

    return best.tolist()
# EVOLVE-BLOCK-END
