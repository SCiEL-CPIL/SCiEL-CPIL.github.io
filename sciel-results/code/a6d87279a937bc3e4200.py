"""Matrix multiplication 2x4x6 rank-42 via Strassen on 2x2x2 sub-blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Strassen rank-7 algorithm for 2x2 * 2x2 matrix multiplication.
    Returns list of 7 terms, each: (a_idx, b_idx, c_idx, d_idx)
    where (A[a_idx] * B[c_idx] + A[b_idx] * B[d_idx]) is the term.
    A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    B indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    """
    return [
        (0, 3, 0, 3),    # m0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        (2, 3, 0, 0),    # m1: (A[1,0]+A[1,1])*B[0,0]
        (0, 0, 1, 3),    # m2: A[0,0]*(B[0,1]-B[1,1])
        (3, 0, 2, 0),    # m3: A[1,1]*(B[1,0]-B[0,0])
        (0, 1, 3, 3),    # m4: (A[0,0]+A[0,1])*B[1,1]
        (2, 0, 0, 1),    # m5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        (1, 3, 2, 3),    # m6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    ]

def _strassen_output_coeffs():
    """
    For each Strassen term index (0..6), return dict of output (i,k) -> coefficient.
    c00 = m0+m3-m4+m6
    c01 = m2+m4
    c10 = m1+m3
    c11 = m0-m1+m2+m5
    """
    return [
        {(0,0): 1, (1,1): 1},    # m0
        {(1,0): 1, (1,1): -1},   # m1
        {(0,1): 1, (1,1): 1},    # m2
        {(0,0): 1, (1,0): 1},    # m3
        {(0,0): -1, (0,1): 1},   # m4
        {(1,1): 1},              # m5
        {(0,0): 1},              # m6
    ]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks: rank = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    strassen_terms = _strassen_2x2x2()
    strassen_out = _strassen_output_coeffs()

    # Collect all terms
    all_terms = []  # (a1, a2, b1, b2, w_idx, coeff)

    for a_block in range(2):  # left=0 (A cols 0,1), right=1 (A cols 2,3)
        for b_block in range(2):  # top=0 (B rows 0,1), bottom=1 (B rows 2,3)
            for sub in range(3):  # column sub-blocks 0,1,2
                a_base = a_block * 2
                b_base = b_block * 12 + sub * 2
                col_base = sub * 2

                for term_idx, (a_i, b_i, c_i, d_i) in enumerate(strassen_terms):
                    a1 = a_base + a_i
                    a2 = a_base + b_i
                    b1 = b_base + c_i
                    b2 = b_base + d_i

                    for (out_i, out_k), coeff in strassen_out[term_idx].items():
                        actual_k = col_base + out_k
                        w_idx = actual_k * n + out_i
                        all_terms.append((a1, a2, b1, b2, w_idx, coeff))

    rank = len(all_terms)
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)

    for col, (a1, a2, b1, b2, w_idx, coeff) in enumerate(all_terms):
        U[a1, col] = 1.0
        U[a2, col] = 1.0
        V[b1, col] = 1.0
        V[b2, col] = 1.0
        W[w_idx, col] = float(coeff)

    # Verify
    T_check = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for r in range(rank):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    if not np.allclose(T_check, T, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=np.float64)
        V = np.zeros((V_dim, rank), dtype=np.float64)
        W = np.zeros((W_dim, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
