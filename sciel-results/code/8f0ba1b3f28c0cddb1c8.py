"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under the 120 s limit

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    bT = b.T.copy()

    # ---- initial solution: rank-matching ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)

    # ---- fast cost computation ----
    def full_cost():
        c = 0
        for i in range(n):
            c += int(np.dot(a[i], b[p[i]]))
        return c

    cur = full_cost()
    best = cur
    best_p = p.copy()

    # ---- precompute for fast delta ----
    a_rowsum = a.sum(axis=1)
    b_colsum = b.sum(axis=1)

    # ---- 2-opt local search with O(n) delta per swap ----
    order = np.argsort(-a_rowsum, kind="stable")

    no_improve = 0
    max_no_improve = 5

    while time.time() < deadline:
        improved = False

        for ii in range(n):
            if time.time() > deadline - 2.0:
                break
            i = order[ii]
            u = p[i]
            ai = a[i]

            for jj in range(ii + 1, n):
                j = order[jj]
                v = p[j]
                if u == v:
                    continue
                aj = a[j]

                diff_b = bT[v] - bT[u]
                diff_a = ai - aj
                cross = int(np.dot(diff_a, diff_b))
                delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross

                if delta < 0:
                    p[i], p[j] = v, u
                    inv[u], inv[v] = j, i
                    cur += delta
                    u, v = v, u
                    improved = True

                    if cur < best:
                        best = cur
                        best_p = p.copy()

        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                if time.time() > deadline - 5.0:
                    break

                # Perturb with random swaps
                for _ in range(min(15, n // 10)):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i == j:
                        continue
                    u = p[i]
                    v = p[j]
                    if u == v:
                        continue
                    aj = a[j]
                    ai = a[i]
                    diff_b = bT[v] - bT[u]
                    diff_a = ai - aj
                    cross = int(np.dot(diff_a, diff_b))
                    delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross

                    p[i], p[j] = v, u
                    inv[u], inv[v] = j, i
                    cur += delta

                    if cur < best:
                        best = cur
                        best_p = p.copy()

                no_improve = 0
        else:
            no_improve = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
