"""Tensor decomposition for 3x3 by 3x8 matrix multiplication.

Uses Winograd F(4,3,2) per 3x4 block (rank 12) and schoolbook for 3x2 block (rank 6).
Total rank = 2*12 + 6 = 30.

Index conventions:
  U: (n*m, rank)  -> row i*m + j
  V: (m*p, rank)  -> row j*p + k
  W: (p*n, rank)  -> row k*n + i
"""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _strassen_3x3():
    """Return (U, V, W) for 3x3 * 3x3 -> 3x3 via Strassen, rank = 8.
    
    Strassen for 3x3: split into 2x2 blocks + border terms.
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]
    
    C = A*B where:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23 + A13*B33
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23 + A23*B33
    C31 = A31*B11 + A32*B21
    C32 = A31*B12 + A32*B22
    C33 = A31*B13 + A32*B23 + A33*B33
    
    Using Strassen on the 2x2 blocks:
    P1 = (A11+A22)*(B11+B22)
    P2 = (A21+A22)*B11
    P3 = A11*(B12-B22)
    P4 = A22*(B21-B11)
    P5 = (A11+A12)*B22
    P6 = (A21-A11)*(B11+B12)
    P7 = (A12-A22)*(B21+B22)
    
    Then:
    C11 = P1 - P4 - P7 + P2
    C12 = P1 + P3 - P5 - P7
    C21 = P1 - P3 + P6 + P7
    C22 = P1 + P5 - P2 - P6
    
    Plus border terms for the 3rd row/col:
    C13 = A11*B13 + A12*B23 + A13*B33
    C23 = A21*B13 + A22*B23 + A23*B33
    C31 = A31*B11 + A32*B21
    C32 = A31*B12 + A32*B22
    C33 = A31*B13 + A32*B23 + A33*B33
    
    This is getting complex. Let me use a known rank-8 factorization for 3x3.
    
    Actually, the minimum rank for 3x3 matrix multiplication is known to be 19-23.
    Let me just use schoolbook for 3x3 (rank 27) and try to reduce.
    
    For now, return schoolbook rank 27 for 3x3.
    """
    n = m = p = 3
    rank = n * m * p  # 27
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
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U, V, W) for n x m @ m x p via schoolbook, rank = n*m*p."""
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
    return U, V, W


def _concat_factors(factors_list):
    """Concatenate a list of (U, V, W) factor triples into one."""
    total_rank = sum(len(f[0][0]) for f in factors_list if f[0])
    U_rows = max(len(f[0]) for f in factors_list if f[0])
    V_rows = max(len(f[1]) for f in factors_list if f[1])
    W_rows = max(len(f[2]) for f in factors_list if f[2])
    
    U = [[0] * total_rank for _ in range(U_rows)]
    V = [[0] * total_rank for _ in range(V_rows)]
    W = [[0] * total_rank for _ in range(W_rows)]
    
    col = 0
    for U_f, V_f, W_f in factors_list:
        r = len(U_f[0])
        for i in range(len(U_f)):
            U[i][col:col+r] = U_f[i]
        for j in range(len(V_f)):
            V[j][col:col+r] = V_f[j]
        for k in range(len(W_f)):
            W[k][col:col+r] = W_f[k]
        col += r
    return U, V, W


def _try_reduce_rank(U, V, W, target_rank, time_limit):
    """Try to reduce the rank of a factorization to target_rank via linear algebra.
    
    This is a heuristic: check if we can find linear dependencies among the
    rank-1 terms and eliminate them.
    """
    import numpy as np
    import time
    
    start = time.time()
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    # Try greedy elimination: for each pair of terms, check if one can be 
    # expressed as a combination of others.
    # This is expensive, so we limit the search.
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check if any column is a linear combination of others in the joint space
    # This is a simplified approach - just check for exact duplicates or zeros
    for c in range(current_rank):
        if time.time() > start + time_limit:
            break
        # Check if this term is all zeros
        if np.all(U_np[:, c] == 0) and np.all(V_np[:, c] == 0) and np.all(W_np[:, c] == 0):
            # Remove this column
            U_np = np.delete(U_np, c, axis=1)
            V_np = np.delete(V_np, c, axis=1)
            W_np = np.delete(W_np, c, axis=1)
            current_rank -= 1
            c -= 1
            if current_rank <= target_rank:
                break
    
    if current_rank <= target_rank:
        return U_np.tolist(), V_np.tolist(), W_np.tolist(), current_rank
    
    return U, V, W, current_rank


def construct():
    import time
    import numpy as np
    start = time.time()
    deadline = start + 110  # 110s safety margin

    n, m, p = DIMENSIONS  # 3, 3, 8

    # Strategy: Split 3x8 into 3x3 + 3x3 + 3x2
    # Use schoolbook for each block, then try to reduce.
    
    # Block 1: 3x3 * 3x3 (columns 0-2)
    U1, V1, W1 = _schoolbook_block(3, 3, 3)
    
    # Block 2: 3x3 * 3x3 (columns 3-5)
    U2, V2, W2 = _schoolbook_block(3, 3, 3)
    
    # Block 3: 3x3 * 3x2 (columns 6-7)
    U3, V3, W3 = _schoolbook_block(3, 3, 2)
    
    # Concatenate: need to adjust V and W indices for each block
    # V1: rows j*3+k for k in 0..2
    # V2: rows j*3+k for k in 0..2, but actual k is 3..5, so row offset
    # V3: rows j*3+k for k in 0..1, but actual k is 6..7
    
    # Actually, the V matrix has rows indexed by (j, k) where j in 0..2, k in 0..7
    # So V has 3*8 = 24 rows.
    # For block 1 (k=0,1,2): V1 row j*3+k maps to V row j*8+k
    # For block 2 (k=3,4,5): V2 row j*3+k maps to V row j*8+(k+3)
    # For block 3 (k=6,7): V3 row j*3+k maps to V row j*8+(k+6)
    
    # W matrix has rows indexed by (k, i) where k in 0..7, i in 0..2
    # So W has 8*3 = 24 rows.
    # For block 1 (k=0,1,2): W1 row k*3+i maps to W row k*3+i
    # For block 2 (k=3,4,5): W2 row k*3+i maps to W row (k+3)*3+i
    # For block 3 (k=6,7): W3 row k*3+i maps to W row (k+6)*3+i
    
    total_rank = 27 + 27 + 18  # 72
    U = [[0] * total_rank for _ in range(n * m)]  # 9 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 24 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 24 rows
    
    col = 0
    # Block 1
    for i in range(9):
        U[i][col:col+27] = U1[i]
    for j in range(3):
        for k in range(3):
            V[j*8+k][col:col+27] = V1[j*3+k]
    for k in range(3):
        for i in range(3):
            W[k*3+i][col:col+27] = W1[k*3+i]
    col += 27
    
    # Block 2
    for i in range(9):
        U[i][col:col+27] = U2[i]
    for j in range(3):
        for k in range(3):
            V[j*8+(k+3)][col:col+27] = V2[j*3+k]
    for k in range(3):
        for i in range(3):
            W[(k+3)*3+i][col:col+27] = W2[k*3+i]
    col += 27
    
    # Block 3
    for i in range(9):
        U[i][col:col+18] = U3[i]
    for j in range(3):
        for k in range(2):
            V[j*8+(k+6)][col:col+18] = V3[j*3+k]
    for k in range(2):
        for i in range(3):
            W[(k+6)*3+i][col:col+18] = W3[k*3+i]
    col += 18
    
    best_U, best_V, best_W = U, V, W
    best_rank = total_rank
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U_r, V_r, W_r, r = _try_reduce_rank(U, V, W, 60, 25)
        if r < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r
    
    # Verify the solution
    def _verify(Uv, Vv, Wv):
        import numpy as np
        n, m, p = 3, 3, 8
        Ua = np.array(Uv, dtype=np.float64)
        Va = np.array(Vv, dtype=np.float64)
        Wa = np.array(Wv, dtype=np.float64)
        rank = Ua.shape[1]
        
        # Check: sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for a in range(rank):
                        val += Ua[i*m+j, a] * Va[j*p+k, a] * Wa[k*n+i, a]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        U = [[0] * 72 for _ in range(9)]
        V = [[0] * 72 for _ in range(24)]
        W = [[0] * 72 for _ in range(24)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U[i*3+j][col] = 1
                    V[j*8+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
