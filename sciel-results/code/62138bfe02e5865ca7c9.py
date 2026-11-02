"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition search.

Structural idea: build a valid decomposition via 2x2 block-Strassen recursion
(6x6 -> blocks of 3x3, Strassen rank-7 on the block level, schoolbook inside
each 3x3 block). This yields a rank-252 valid factorization with lots of
redundant terms. Refinement machinery: fast einsum verification + greedy
column removal to prune redundant terms toward a rank below the schoolbook 216.
"""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before the 150s kill

    def elapsed():
        return time.time() - start_time

    def check_decomposition(Um, Vm, Wm):
        """Exact check: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        R = Um.shape[1]
        if R == 0:
            return False
        Um_3d = Um.reshape(n, m, R)
        Vm_3d = Vm.reshape(m, p, R)
        Wm_3d = Wm.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', Um_3d, Vm_3d, Wm_3d, optimize=True)
        return np.allclose(T, 1.0, atol=1e-7)

    # ---- Schoolbook rank-216 factorization (always a valid fallback) ----
    rank0 = n * m * p  # 216
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    # ---- 2x2 block-Strassen factorization (rank 4*7*9 = 252, valid) ----
    # 6x6 matrix partitioned into 2x2 blocks of 3x3 submatrices.
    # Strassen: M1=M3=(A11+A22)(B12-B22), M2=(A21+A22)B11, M3=A11(B22-B12),
    # M4=A22(B11-B22), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7, C12=M2+M3, C21=M6+M5, C22=M1-M2+M3+M4
    STRASSEN = [
        (0, 1, 1, -1),  # M1 = M3
        (1, 2, 1, 0),   # M2
        (0, 0, 1, -1),  # M3
        (1, 1, 1, -1),  # M4
        (0, 0, 1, 0),   # M5
        (1, 0, -1, 0),  # M6
        (0, 1, -1, 1),  # M7
    ]
    APOS = [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 0), (0, 1)]
    BPOS = [(0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (0, 0), (1, 0)]
    CPOS = [(0, 0), (0, 1), (1, 0), (1, 1)]
    CCOEF = {
        (0, 0): [(1, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(1, 1), (2, 1)],
        (1, 0): [(5, 1), (4, 1)],
        (1, 1): [(1, 1), (1, -1), (2, 1), (3, 1)],
    }
    # Local (a,b) -> global (i,j) for a 3x3 block at block-row br, block-col bc
    def gij(a, b, br, bc):
        return (2 * br + a, 2 * bc + b)
    def gjk(a, b, bc, bk):
        return (2 * bc + a, 2 * bk + b)
    def gki(a, b, bk, br):
        return (2 * bk + a, 2 * br + b)

    def build_block_strassen():
        U = np.zeros((n * m, 252))
        V = np.zeros((m * p, 252))
        W = np.zeros((p * n, 252))
        col = 0
        for br in range(2):
            for bc in range(2):
                for bk in range(2):
                    for r in range(7):
                        aA, bA = APOS[r]
                        aB, bB = BPOS[r]
                        U[gij(aA, bA, br, bc), col] = 1.0
                        V[gjk(aB, bB, bc, bk), col] = 1.0
                        for (ci, cj), terms in CCOEF.items():
                            for (rr, coef) in terms:
                                if rr == r:
                                    W[gki(aB, bB, bk, ci), col] += float(coef)
                        col += 1
        return U, V, W

    U_bs, V_bs, W_bs = build_block_strassen()
    # Sanity: block-Strassen must be valid; if not, fall back to schoolbook.
    if not check_decomposition(U_bs, V_bs, W_bs):
        U_bs, V_bs, W_bs = U0, V0, W0

    # ---- Greedy column removal (structural refinement) ----
    def greedy_prune(Um, Vm, Wm, order, time_budget):
        R = Um.shape[1]
        U_cur, V_cur, W_cur = Um, Vm, Wm
        cur_rank = R
        # order is a permutation of [0, R); iterate, skipping stale indices
        for idx in order:
            if time.time() > time_budget:
                break
            if idx >= cur_rank:
                continue
            keep = [r for r in range(cur_rank) if r != idx]
            U_new = U_cur[:, keep]
            V_new = V_cur[:, keep]
            W_new = W_cur[:, keep]
            if check_decomposition(U_new, V_new, W_new):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                cur_rank = len(keep)
        return U_cur, V_cur, W_cur, cur_rank

    # ---- Search: multiple random orderings over the block-Strassen start ----
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        order = list(range(U_bs.shape[1]))
        random.shuffle(order)
        U_r, V_r, W_r, rank_r = greedy_prune(U_bs, V_bs, W_bs, order, deadline - 25)
        if rank_r < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r

    # Also try pruning the schoolbook start (cheap extra, may help if BS stalls)
    if time.time() < deadline - 25:
        order = list(range(rank0))
        random.shuffle(order)
        U_r, V_r, W_r, rank_r = greedy_prune(U0, V0, W0, order, deadline - 25)
        if rank_r < best_rank:
            best_U, best_V, W_r = U_r, V_r, W_r
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r

    # ---- Final verification; fall back to schoolbook if anything is off ----
    if not check_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist(),
    }
# EVOLVE-BLOCK-END
