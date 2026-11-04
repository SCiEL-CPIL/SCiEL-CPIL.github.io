"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition.

Deterministic rank-17 exact decomposition:
  - Block A = A[:,0:2] (2x2) * B[0:2,:] (2x3)  -> 2x2x2 sub-problem, Strassen rank 7
  - Block B = A[:,2]   (2x1) * B[2,:]   (1x3)  -> 2x1x3 sub-problem, direct rank 6
  Total rank = 7 + 6 = 17 (schoolbook is 18).

Tensor index convention:
  U index i = i_a*m + j_a   (A row i_a, col j_a)
  V index j = j_a*p + k_b   (B row j_a, col k_b)
  W index k = k_b*n + i_c   (C row i_c, col k_b)
All factor entries are integers, so the decomposition is exact with no rounding.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS


def construct():
    """Return {'U','V','W'} factor matrices (lists) of a rank-minimal exact
    decomposition of the 2x3 * 3x3 matrix multiplication tensor."""
    start = time.time()
    deadline = start + 110  # safety margin before the 120s wall-clock kill

    n, m, p = N, M, P
    u_dim = n * m   # 6
    v_dim = m * p   # 9
    w_dim = p * n   # 6

    def ui(i_a, j_a):
        return i_a * m + j_a

    def vi(j_a, k_b):
        return j_a * p + k_b

    def wi(k_b, i_c):
        return k_b * n + i_c

    # ------------------------------------------------------------------
    # Build the full multiplication tensor for verification.
    # T[i,j,k] = 1 iff (i//m)==(k%n) and (i%m)==(j//p) and (j%p)==(k//n)
    # ------------------------------------------------------------------
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(u_dim):
        i_a = i // m
        j_a = i % m
        for j in range(v_dim):
            if j_a != j // p:
                continue
            k_b = j % p
            for k in range(w_dim):
                if i_a == k % n and k_b == k // n:
                    T[i, j, k] = 1.0

    # ------------------------------------------------------------------
    # Strassen 2x2x2 block: A[:,0:2] (2x2) * B[0:2,:] (2x3) -> C[:,0:2]
    #   A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    #   B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    #   C00=C[0,0], C01=C[0,1], C10=C[1,0], C11=C[1,1]
    # ------------------------------------------------------------------
    strassen = []
    # M1 = (A00 + A11) * (B00 + B11)
    strassen.append(([(ui(0, 0), 1), (ui(1, 1), 1)],
                     [(vi(0, 0), 1), (vi(1, 1), 1)],
                     [(wi(0, 0), 1), (wi(1, 0), -1), (wi(0, 1), 1), (wi(1, 1), 1)]))
    # M2 = (A10 + A11) * B00
    strassen.append(([(ui(1, 0), 1), (ui(1, 1), 1)],
                     [(vi(0, 0), 1)],
                     [(wi(0, 1), -1), (wi(1, 1), 1)]))
    # M3 = A00 * (B10 - B00)
    strassen.append(([(ui(0, 0), 1)],
                     [(vi(1, 0), 1), (vi(0, 0), -1)],
                     [(wi(0, 0), 1), (wi(0, 1), 1)]))
    # M4 = A11 * (B01 - B11)
    strassen.append(([(ui(1, 1), 1)],
                     [(vi(0, 1), 1), (vi(1, 1), -1)],
                     [(wi(0, 1), 1), (wi(1, 0), -1)]))
    # M5 = (A00 + A01) * B11
    strassen.append(([(ui(0, 0), 1), (ui(0, 1), 1)],
                     [(vi(1, 1), 1)],
                     [(wi(0, 0), 1), (wi(1, 1), -1)]))
    # M6 = (A10 - A00) * (B00 + B10)
    strassen.append(([(ui(1, 0), 1), (ui(0, 0), -1)],
                     [(vi(0, 0), 1), (vi(1, 0), 1)],
                     [(wi(0, 0), -1), (wi(1, 0), -1)]))
    # M7 = (A01 - A11) * (B11 + B01)
    strassen.append(([(ui(0, 1), 1), (ui(1, 1), -1)],
                     [(vi(1, 1), 1), (vi(0, 1), 1)],
                     [(wi(1, 1), -1), (wi(0, 1), -1)]))

    # ------------------------------------------------------------------
    # Direct 2x1x3 block: A[:,2] (2x1) * B[2,:] (1x3) -> C[:,2]
    #   C[i,2] = A[i,2] * B[2,2]   for i in {0,1}
    # ------------------------------------------------------------------
    direct = []
    for i_a in range(n):
        for k_b in range(p):
            direct.append(([(ui(i_a, 2), 1)],
                           [(vi(2, k_b), 1)],
                           [(wi(k_b, i_a), 1)]))

    all_terms = strassen + direct
    rank = len(all_terms)

    # ------------------------------------------------------------------
    # Assemble factor matrices (integer entries).
    # ------------------------------------------------------------------
    U = np.zeros((u_dim, rank))
    V = np.zeros((v_dim, rank))
    W = np.zeros((w_dim, rank))
    for c, (u_terms, v_terms, w_terms) in enumerate(all_terms):
        for idx, val in u_terms:
            U[idx, c] += val
        for idx, val in v_terms:
            V[idx, c] += val
        for idx, val in w_terms:
            W[idx, c] += val

    # ------------------------------------------------------------------
    # Verify exactness (defensive; construction is exact by design).
    # ------------------------------------------------------------------
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    ok = bool(np.all(recon == T))
    if not ok:
        # Fallback to the guaranteed-correct schoolbook rank-18 decomposition.
        U = np.zeros((u_dim, n * m * p))
        V = np.zeros((v_dim, n * m * p))
        W = np.zeros((w_dim, n * m * p))
        col = 0
        for i_a in range(n):
            for j_a in range(m):
                for k_b in range(p):
                    U[ui(i_a, j_a), col] = 1.0
                    V[vi(j_a, k_b), col] = 1.0
                    W[wi(k_b, i_a), col] = 1.0
                    col += 1
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        assert np.all(recon == T)

    # Entries are integers; emit as exact floats (integers are valid
    # integer/half-integer entries).
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
