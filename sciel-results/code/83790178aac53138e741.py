# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    tensor = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(tensor == 1)


def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_terms(i_off, j_off, k_off):
    """
    Return the 7 Strassen terms for a 2x2x2 block.
    Each term is (U_dict, V_dict, W_dict) where each dict maps global indices to coefficients.
    Indices are (i, j) for U, (j, k) for V, (k, i) for W (trace convention W[k*n+i]).
    """
    terms = []

    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # Contributes to C[0,0] with +1, C[1,1] with +1
    U1 = {(i_off + 0, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V1 = {(j_off + 0, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W1 = {(k_off + 0, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U1, V1, W1))

    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # Contributes to C[1,0] with +1, C[1,1] with -1
    U2 = {(i_off + 1, j_off + 0): 1, (i_off + 1, j_off + 1): 1}
    V2 = {(j_off + 0, k_off + 0): 1}
    W2 = {(k_off + 0, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
    terms.append((U2, V2, W2))

    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # Contributes to C[0,1] with +1, C[1,1] with +1
    U3 = {(i_off + 0, j_off + 0): 1}
    V3 = {(j_off + 0, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
    W3 = {(k_off + 1, i_off + 0): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U3, V3, W3))

    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # Contributes to C[1,0] with +1, C[0,0] with +1
    U4 = {(i_off + 1, j_off + 1): 1}
    V4 = {(j_off + 1, k_off + 0): 1, (j_off + 0, k_off + 0): -1}
    W4 = {(k_off + 0, i_off + 1): 1, (k_off + 0, i_off + 0): 1}
    terms.append((U4, V4, W4))

    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # Contributes to C[0,1] with +1, C[0,0] with -1
    U5 = {(i_off + 0, j_off + 0): 1, (i_off + 0, j_off + 1): 1}
    V5 = {(j_off + 1, k_off + 1): 1}
    W5 = {(k_off + 1, i_off + 0): 1, (k_off + 0, i_off + 0): -1}
    terms.append((U5, V5, W5))

    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # Contributes to C[1,1] with +1
    U6 = {(i_off + 1, j_off + 0): 1, (i_off + 0, j_off + 0): -1}
    V6 = {(j_off + 0, k_off + 0): 1, (j_off + 0, k_off + 1): 1}
    W6 = {(k_off + 1, i_off + 1): 1}
    terms.append((U6, V6, W6))

    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # Contributes to C[0,0] with +1
    U7 = {(i_off + 0, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
    V7 = {(j_off + 1, k_off + 0): 1, (j_off + 1, k_off + 1): 1}
    W7 = {(k_off + 0, i_off + 0): 1}
    terms.append((U7, V7, W7))

    return terms


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Uses four disjoint Strassen 2x2x2 blocks to reduce rank from 60 to 56.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = N, M, P

    # Four disjoint Strassen 2x2x2 blocks
    strassen_blocks = [
        (0, 0, 0),  # i in {0,1}, j in {0,1}, k in {0,1}
        (0, 2, 2),  # i in {0,1}, j in {2,3}, k in {2,3}
        (1, 0, 3),  # i in {1,2}, j in {0,1}, k in {3,4}
        (1, 2, 0),  # i in {1,2}, j in {2,3}, k in {0,1}
    ]

    # Identify which (i,j,k) tuples are covered by Strassen blocks.
    strassen_covered = set()
    for i_off, j_off, k_off in strassen_blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    strassen_covered.add((i_off + ii, j_off + jj, k_off + kk))

    # Build the full decomposition.
    all_terms = []

    # Add Strassen terms for each block
    for i_off, j_off, k_off in strassen_blocks:
        block_terms = _strassen_terms(i_off, j_off, k_off)
        for U_dict, V_dict, W_dict in block_terms:
            all_terms.append((U_dict, V_dict, W_dict))

    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_sb = {(i, j): 1}
                    V_sb = {(j, k): 1}
                    W_sb = {(k, i): 1}
                    all_terms.append((U_sb, V_sb, W_sb))

    rank = len(all_terms)

    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff

    result = {"U": U, "V": V, "W": W}

    # Verify
    if _verify(U, V, W):
        return result

    # Fallback to schoolbook if verification fails
    return _schoolbook()
# EVOLVE-BLOCK-END
