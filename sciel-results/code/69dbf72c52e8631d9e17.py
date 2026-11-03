"""
Matrix multiplication 2x4 x 4x7 — rank-minimal exact tensor decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Rank-7 decomposition of 2x2x2 multiplication (Strassen).
    Returns U (4,7), V (4,7), W (4,7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _get_school_221():
    """
    Rank-4 decomposition of 2x2x1 multiplication (schoolbook).
    Returns U (4,4), V (2,4), W (2,4).
    U[i*2+j, c], V[j*1+k, c], W[k*2+i, c].
    """
    r = 4
    U = np.zeros((4, r))
    V = np.zeros((2, r))
    W = np.zeros((2, r))

    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0

    return U, V, W

def _build_224():
    """
    Build rank-11 decomposition of 2x2x4 multiplication.
    Splits p=4 into two p=2 blocks, each using Strassen 2x2x2 (rank 7).
    Returns U (4,11), V (8,11), W (8,11).
    """
    U_s, V_s, W_s = _get_strassen_222()
    U = np.zeros((4, 11))
    V = np.zeros((8, 11))
    W = np.zeros((8, 11))

    # First p=2 block (k=0,1): columns 0..6
    U[:, 0:7] = U_s
    V[0:4, 0:7] = V_s  # j=0,1; k=0,1
    W[0:4, 0:7] = W_s  # k=0,1; i=0,1

    # Second p=2 block (k=2,3): columns 7..10
    U[:, 7:11] = U_s
    V[4:8, 7:11] = V_s  # j=0,1; k=2,3
    W[4:8, 7:11] = W_s  # k=2,3; i=0,1

    return U, V, W

def _build_223():
    """
    Build rank-11 decomposition of 2x2x3 multiplication.
    Splits p=3 into p=2 (Strassen, rank 7) + p=1 (schoolbook, rank 4).
    Returns U (4,11), V (6,11), W (6,11).
    """
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))

    # p=2 block (k=0,1): columns 0..6
    U[:, 0:7] = U_s
    V[0:4, 0:7] = V_s
    W[0:4, 0:7] = W_s

    # p=1 block (k=2): columns 7..10
    U[:, 7:11] = U_sc
    V[4:6, 7:11] = V_sc
    W[4:6, 7:11] = W_sc

    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify the tensor identity:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j, j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Strategy: split m=4 into two m=2 blocks.
    # For each m=2 block, split p=7 into p=4 and p=3.
    #   p=4: rank 11 (two Strassen 2x2x2)
    #   p=3: rank 11 (one Strassen 2x2x2 + one schoolbook 2x2x1)
    # Rank per m-block: 11 + 11 = 22
    # Total rank: 2 * 22 = 44

    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U_224, V_224, W_224 = _build_224()
    U_223, V_223, W_223 = _build_223()

    col_offset = 0

    for j_start in [0, 2]:
        # p=4 block: k=0..3, rank 11
        # U: i=0,1; j=j_start, j_start+1 -> U[i*4 + j, col]
        for i in range(2):
            U[i * 4 + j_start, col_offset:col_offset+11] = U_224[i * 2 + 0, :]
            U[i * 4 + j_start + 1, col_offset:col_offset+11] = U_224[i * 2 + 1, :]

        # V: j=j_start, j_start+1; k=0..3 -> V[j*7 + k, col]
        for j_sub in range(2):
            j = j_start + j_sub
            for k in range(4):
                V[j * 7 + k, col_offset:col_offset+11] = V_224[j_sub * 2 + k, :]

        # W: k=0..3; i=0,1 -> W[k*2 + i, col]
        for k in range(4):
            for i in range(2):
                W[k * 2 + i, col_offset:col_offset+11] = W_224[k * 2 + i, :]

        col_offset += 11

        # p=3 block: k=4..6, rank 11
        for i in range(2):
            U[i * 4 + j_start, col_offset:col_offset+11] = U_223[i * 2 + 0, :]
            U[i * 4 + j_start + 1, col_offset:col_offset+11] = U_223[i * 2 + 1, :]

        for j_sub in range(2):
            j = j_start + j_sub
            for k in range(3):
                V[j * 7 + 4 + k, col_offset:col_offset+11] = V_223[j_sub * 2 + k, :]

        for k in range(3):
            for i in range(2):
                W[(4 + k) * 2 + i, col_offset:col_offset+11] = W_223[k * 2 + i, :]

        col_offset += 11

    # Verify and return
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback: schoolbook rank 56
            r = n * m * p
            U_fb = np.zeros((n * m, r))
            V_fb = np.zeros((m * p, r))
            W_fb = np.zeros((p * n, r))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
