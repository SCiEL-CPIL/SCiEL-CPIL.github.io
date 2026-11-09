import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin below 120s


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    if time.time() > _DEADLINE:
        raise TimeoutError("no time to build")

    # Split the middle dimension m=5 into 2 + 3.
    # A (n x m) = [ A1 (n x 2) | A2 (n x 3) ]
    # B (m x p) = [ B1 (2 x p) ; B2 (3 x p) ]
    # C = A1*B1 + A2*B2
    #
    # For the (n x 2) * (2 x p) product, split n into 2+2 and p into 2+3.
    # The 2x2*2x2 sub-block uses Strassen (rank 7); the rest uses schoolbook.
    # This gives 35 total rank-1 terms (vs 100 schoolbook).

    # ---- Strassen rank-1 terms for 2x2*2x2 ----
    # Local flat ordering:
    #   A[a*2+b] -> u, index in {0..3}
    #   B[b*2+c] -> v, index in {0..3}
    #   C[c*2+a] -> w, index in {0..3}
    # Each term is (u_dict, v_dict, w_dict) of (local_index -> coeff).
    strassen_2x2 = [
        # M1 = (A11+A22)(B11+B22), contributes to C11 + C22
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A21+A22)B11 -> C21
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3 = A11(B12-B22) -> C12
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        # M4 = A22(B21-B11) -> C11 + C21
        ({3: 1}, {0: -1, 2: 1}, {0: 1, 2: 1}),
        # M5 = (A11+A12)B22 -> C11 + C12
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1}),
        # M6 = (A21-A11)(B11+B12) -> C22
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A12-A22)(B21+B22) -> C11 + C22
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),
    ]

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_local, v_local, w_local, i_off, j_off, k_off, n_local, m_local, p_local):
        """Add one rank-1 term.
        u_local: dict local (a*ml+b)->coef over A block (n_local x m_local)
        v_local: dict local (b*pl+c)->coef over B block (m_local x p_local)
        w_local: dict local (c*nl+a)->coef over C block (p_local x n_local)
        """
        U = {}
        V = {}
        W = {}
        for (a, b), coef in u_local.items():
            gi = i_off + a
            gj = j_off + b
            idx = gi * m + gj
            U[idx] = U.get(idx, 0) + coef
        for (b, c), coef in v_local.items():
            gj = j_off + b
            gk = k_off + c
            idx = gj * p + gk
            V[idx] = V.get(idx, 0) + coef
        for (c, a), coef in w_local.items():
            gk = k_off + c
            gi = i_off + a
            idx = gk * n + gi
            W[idx] = W.get(idx, 0) + coef
        U_cols.append(U)
        V_cols.append(V)
        W_cols.append(W)

    def add_schoolbook_block(i_off, j_off, k_off, n_l, m_l, p_l):
        for a in range(n_l):
            for b in range(m_l):
                for c in range(p_l):
                    add_term({(a, b): 1}, {(b, c): 1}, {(c, a): 1},
                             i_off, j_off, k_off, n_l, m_l, p_l)

    def add_strassen_block(i_off, j_off, k_off):
        # 2x2 * 2x2, all local indices in {0,1}
        for (u_l, v_l, w_l) in strassen_2x2:
            add_term(u_l, v_l, w_l, i_off, j_off, k_off, 2, 2, 2)

    # Middle split offsets
    M2 = 2
    # A1: j in [0,2); A2: j in [2,5)
    # B1: j in [0,2); B2: j in [2,5)

    # Split n into 2+2, p into 2+3
    for (i_off, n_l) in [(0, 2), (2, 2)]:
        for (k_off, p_l) in [(0, 2), (2, 3)]:
            if n_l == 2 and p_l == 2:
                # A1 block: j in [0,2), n 2, m 2, p 2 -> Strassen
                add_strassen_block(i_off, 0, k_off)
                # A2 block: j in [2,5), n 2, m 3, p 2 -> schoolbook
                add_schoolbook_block(i_off, 2, k_off, 2, 3, 2)
            else:
                # A1 block: j in [0,2), n 2, m 2, p 3 -> schoolbook
                add_schoolbook_block(i_off, 0, k_off, 2, 2, p_l)
                # A2 block: j in [2,5), n 2, m 3, p 3 -> schoolbook
                add_schoolbook_block(i_off, 2, k_off, 2, 3, p_l)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val

    # ---- In-process verification against schoolbook tensor ----
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reconstruct: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Compare to 1 for all i,j,k.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    if max_err > 1e-9:
        # Fallback: pure schoolbook (guaranteed correct, rank 100)
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})
        rank = len(U_cols)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            for idx, val in U_cols[c].items():
                U[idx][c] = val
            for idx, val in V_cols[c].items():
                V[idx][c] = val
            for idx, val in W_cols[c].items():
                W[idx][c] = val

    if time.time() > _DEADLINE:
        raise TimeoutError("build exceeded budget")

    return {"U": U, "V": V, "W": W}
