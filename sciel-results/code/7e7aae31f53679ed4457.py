"""
Program for 5x5x5 matrix multiplication tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks to reduce rank.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen coefficients
    # M1 = (A00+A11)*(B00+B11), C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00, C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11), C01 += M3, C11 += M3
    # M4 = A11*(B10-B00), C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11, C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01), C11 += M6
    # M7 = (A01-A11)*(B10+B11), C00 += M7
    
    a = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[0, 0], [1, -1]],  # M6: A10 - A00
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]
    b = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[0, 0], [1, -1]],  # M4: B10 - B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 0], [1, 1]],   # M7: B10 + B11
    ]
    c = [
        [[1, 0], [0, 1]],   # M1: C00 + C11
        [[0, 0], [1, -1]],  # M2: C10 - C11
        [[0, 1], [0, 1]],   # M3: C01 + C11
        [[1, 0], [1, 0]],   # M4: C00 + C10
        [[-1, 1], [0, 0]],  # M5: -C00 + C01
        [[0, 0], [0, 1]],   # M6: C11
        [[1, 0], [0, 0]],   # M7: C00
    ]
    
    for q in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j, q] = a[q][i][j]
        for j in range(2):
            for k in range(2):
                V[j*2+k, q] = b[q][j][k]
        for k in range(2):
            for i in range(2):
                W[k*2+i, q] = c[q][i][k]
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """
    Return schoolbook decomposition for n x m @ m x p -> n x p.
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _block_decomposition_5x5x5():
    """
    Decompose 5x5x5 matmul using block structure with sizes [2, 3].
    """
    U00, V00, W00 = _strassen_2x2x2()
    rank00 = U00.shape[1]  # 7
    
    U01, V01, W01 = _schoolbook_block(2, 3, 2)
    rank01 = U01.shape[1]  # 12
    
    U10, V10, W10 = _schoolbook_block(3, 2, 3)
    rank10 = U10.shape[1]  # 18
    
    U11, V11, W11 = _schoolbook_block(3, 3, 3)
    rank11 = U11.shape[1]  # 27
    
    total_rank = rank00 + rank01 + rank10 + rank11  # 64
    
    U = np.zeros((25, total_rank), dtype=float)
    V = np.zeros((25, total_rank), dtype=float)
    W = np.zeros((25, total_rank), dtype=float)
    
    col = 0
    # Block (0,0): A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    U[0:10, col:col+rank00] = U00
    V[0:10, col:col+rank00] = V00
    W[0:10, col:col+rank00] = W00
    col += rank00
    
    # Block (0,1): A[0:2,2:5] @ B[2:5,0:2] -> C[0:2,0:2]
    for r in range(rank01):
        for i in range(2):
            for j in range(3):
                U[i*5 + (2+j), col+r] = U01[i*3+j, r]
        for j in range(3):
            for k in range(2):
                V[(2+j)*5 + k, col+r] = V01[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W[k*5 + i, col+r] = W01[k*2+i, r]
    col += rank01
    
    # Block (1,0): A[2:5,0:2] @ B[0:2,2:5] -> C[2:5,2:5]
    for r in range(rank10):
        for i in range(3):
            for j in range(2):
                U[(2+i)*5 + j, col+r] = U10[i*2+j, r]
        for j in range(2):
            for k in range(3):
                V[j*5 + (2+k), col+r] = V10[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[(2+k)*5 + (2+i), col+r] = W10[k*3+i, r]
    col += rank10
    
    # Block (1,1): A[2:5,2:5] @ B[2:5,2:5] -> C[2:5,2:5]
    for r in range(rank11):
        for i in range(3):
            for j in range(3):
                U[(2+i)*5 + (2+j), col+r] = U11[i*3+j, r]
        for j in range(3):
            for k in range(3):
                V[(2+j)*5 + (2+k), col+r] = V11[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[(2+k)*5 + (2+i), col+r] = W11[k*3+i, r]
    col += rank11
    
    return U, V, W


def _validate_decomposition(U, V, W, n, m, p):
    """Validate the decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use block decomposition with Strassen
    U, V, W = _block_decomposition_5x5x5()
    
    # Validate
    if _validate_decomposition(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if validation fails
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
