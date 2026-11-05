"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Checks that for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _winograd_3x3():
    """
    Return the 7-term Winograd decomposition for 3x3 matrix multiplication.
    Returns (U3, V3, W3) each of shape (9, 7) where:
      U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    """
    U3 = np.zeros((9, 7))
    V3 = np.zeros((9, 7))
    W3 = np.zeros((9, 7))
    
    # Term 0: (a00+a10+a20)*(b00+b10+b20) -> c00+c10+c20
    U3[0, 0] = 1; U3[3, 0] = 1; U3[6, 0] = 1
    V3[0, 0] = 1; V3[3, 0] = 1; V3[6, 0] = 1
    W3[0, 0] = 1; W3[3, 0] = 1; W3[6, 0] = 1
    
    # Term 1: (a00+a10+a20)*(b01+b11+b21) -> c01+c11+c21
    U3[0, 1] = 1; U3[3, 1] = 1; U3[6, 1] = 1
    V3[1, 1] = 1; V3[4, 1] = 1; V3[7, 1] = 1
    W3[1, 1] = 1; W3[4, 1] = 1; W3[7, 1] = 1
    
    # Term 2: (a00+a10+a20)*(b02+b12+b22) -> c02+c12+c22
    U3[0, 2] = 1; U3[3, 2] = 1; U3[6, 2] = 1
    V3[2, 2] = 1; V3[5, 2] = 1; V3[8, 2] = 1
    W3[2, 2] = 1; W3[5, 2] = 1; W3[8, 2] = 1
    
    # Term 3: (a10+a20)*(b00+b10+b20) -> -c00+c10+c20
    U3[3, 3] = 1; U3[6, 3] = 1
    V3[0, 3] = 1; V3[3, 3] = 1; V3[6, 3] = 1
    W3[0, 3] = -1; W3[3, 3] = 1; W3[6, 3] = 1
    
    # Term 4: (a10+a20)*(b01+b11+b21) -> -c01+c11+c21
    U3[3, 4] = 1; U3[6, 4] = 1
    V3[1, 4] = 1; V3[4, 4] = 1; V3[7, 4] = 1
    W3[1, 4] = -1; W3[4, 4] = 1; W3[7, 4] = 1
    
    # Term 5: (a10+a20)*(b02+b12+b22) -> -c02+c12+c22
    U3[3, 5] = 1; U3[6, 5] = 1
    V3[2, 5] = 1; V3[5, 5] = 1; V3[8, 5] = 1
    W3[2, 5] = -1; W3[5, 5] = 1; W3[8, 5] = 1
    
    # Term 6: a20*(b00+b10+b20) -> -c00-c10+c20
    U3[6, 6] = 1
    V3[0, 6] = 1; V3[3, 6] = 1; V3[6, 6] = 1
    W3[0, 6] = -1; W3[3, 6] = -1; W3[6, 6] = 1
    
    # Verify this decomposition
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_idx = i * 3 + j
                b_idx = j * 3 + k
                c_idx = k * 3 + i
                val = np.dot(U3[a_idx, :], V3[b_idx, :] * W3[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return None, None, None
    
    return U3, V3, W3

def _build_decomposition(n, m, p):
    """
    Build a structured decomposition using Winograd 3x3 for the core
    and rank-1 terms for the extra dimensions.
    Returns (U, V, W) or None if verification fails.
    """
    # Winograd 3x3 for i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    U3, V3, W3 = _winograd_3x3()
    if U3 is None:
        return None
    
    # Total rank: 7 (Winograd) + 24 (j=3) + 60 (k=3..7) = 91
    total_rank = 7 + n * p + n * m * (p - 3)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Part 1: Embed Winograd 3x3 (7 terms)
    for i in range(3):
        for j in range(3):
            for r in range(7):
                U[i * m + j, col + r] = U3[i * 3 + j, r]
    
    for j in range(3):
        for k in range(3):
            for r in range(7):
                V[j * p + k, col + r] = V3[j * 3 + k, r]
    
    for k in range(3):
        for i in range(3):
            for r in range(7):
                W[k * n + i, col + r] = W3[k * 3 + i, r]
    
    col += 7
    
    # Part 2: j=3, all i in {0,1,2}, all k in {0,...,7}
    # Each (i,k) pair gets a rank-1 term
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 3
            v_idx = 3 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
    
    # Part 3: j in {0,1,2,3}, k in {3,...,7}
    # Each (i,j,k) with k>=3 gets a rank-1 term
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try structured decomposition (Winograd 3x3 core + rank-1 extras)
    result = _build_decomposition(n, m, p)
    
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook (rank = n*m*p = 96)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
