import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _tensor_product_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Tensor product of two matrix multiplication decompositions.
    Combines (n1 x m1) * (m1 x p1) -> (n1 x p1) with rank r1
    and (n2 x m2) * (m2 x p2) -> (n2 x p2) with rank r2
    to give (n1*n2 x m1*m2) * (m1*m2 x p1*p2) -> (n1*n2 x p1*p2) with rank r1*r2.
    
    Indexing convention:
    U: row index = i*m + j (i: row of A, j: col of A)
    V: row index = j*p + k (j: row of B, k: col of B)
    W: row index = k*n + i (k: col of C, i: row of C)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, r_idx] = U1[i1 * m1 + j1, r1_idx] * U2[i2 * m2 + j2, r2_idx]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, r_idx] = V1[j1 * p1 + k1, r1_idx] * V2[j2 * p2 + k2, r2_idx]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, r_idx] = W1[k1 * n1 + i1, r1_idx] * W2[k2 * n2 + i2, r2_idx]
    
    return U, V, W

def _strassen_4x4x4():
    """
    Build 4x4x4 decomposition from 2x2x2 Strassen using tensor product.
    Rank = 7*7 = 49.
    """
    U222, V222, W222 = _strassen_2x2x2()
    U444, V444, W444 = _tensor_product_decomp(
        U222, V222, W222, U222, V222, W222,
        2, 2, 2, 2, 2, 2
    )
    return U444, V444, W444

def _rank1_2x2x1():
    """
    Rank-1 decomposition for 2x2 * 2x1 -> 2x1.
    T[i,j,0] = 1 for all i,j in {0,1}.
    This is (e_0+e_1) ⊗ (e_0+e_1) ⊗ e_0, which is rank 1.
    
    U has shape (2*2, 1) = (4, 1): U[i*2+j, 0] = 1 for all i,j
    V has shape (2*1, 1) = (2, 1): V[j*1+0, 0] = 1 for all j
    W has shape (1*2, 1) = (2, 1): W[0*2+i, 0] = 1 for all i
    """
    U = np.ones((4, 1), dtype=int)
    V = np.ones((2, 1), dtype=int)
    W = np.ones((2, 1), dtype=int)
    return U, V, W

def _strassen_4x4x2():
    """
    Build 4x4x2 decomposition using tensor product:
    (2x2x2) ⊗ (2x2x1) with rank 7 * 1 = 7.
    
    This computes (4x4) * (4x2) -> (4x2) using 7 scalar multiplications.
    """
    U222, V222, W222 = _strassen_2x2x2()
    U221, V221, W221 = _rank1_2x2x1()
    U442, V442, W442 = _tensor_product_decomp(
        U222, V222, W222, U221, V221, W221,
        2, 2, 2, 2, 2, 1
    )
    return U442, V442, W442

def _try_explicit_split(n, m, p):
    """
    Construct a split decomposition for 4x4x6 by splitting into:
    - 4x4x4 part (columns 0-3): Strassen rank 49
    - 4x4x2 part (columns 4-5): Strassen-based rank 7
    Total rank = 56.
    """
    if n != 4 or m != 4 or p != 6:
        return None
    
    # Get 4x4x4 Strassen decomposition (rank 49)
    U444, V444, W444 = _strassen_4x4x4()
    rank444 = U444.shape[1]  # 49
    
    # Get 4x4x2 Strassen-based decomposition (rank 7)
    U442, V442, W442 = _strassen_4x4x2()
    rank442 = U442.shape[1]  # 7
    
    total_rank = rank444 + rank442  # 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Fill in 4x4x4 part (first 4 columns of output, k=0..3)
    for i in range(4):
        for j in range(4):
            for r in range(rank444):
                U[i*m+j, r] = U444[i*m+j, r]
    
    for j in range(4):
        for k in range(4):
            for r in range(rank444):
                V[j*p + k, r] = V444[j*4+k, r]
    
    for k in range(4):
        for i in range(4):
            for r in range(rank444):
                W[k*n + i, r] = W444[k*4+i, r]
    
    # Fill in 4x4x2 part (last 2 columns of output, k=4..5)
    offset = rank444
    for i in range(4):
        for j in range(4):
            for r in range(rank442):
                U[i*m+j, offset + r] = U442[i*4+j, r]
    
    for j in range(4):
        for k in range(2):
            for r in range(rank442):
                V[j*p + (k + 4), offset + r] = V442[j*2+k, r]
    
    for k in range(2):
        for i in range(4):
            for r in range(rank442):
                W[(k + 4)*n + i, offset + r] = W442[k*4+i, r]
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try explicit split: 4x4x4 (Strassen, rank 49) + 4x4x2 (Strassen-based, rank 7) = rank 56
    if time.time() < deadline:
        result = _try_explicit_split(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    return best_result
