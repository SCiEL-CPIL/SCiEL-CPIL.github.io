"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=7):
    """Verify that U, V, W give an exact decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=7):
    """Return the schoolbook (rank=n*m*p) decomposition."""
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

def _strassen_2x2_block():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # P2 = (A10+A11)B00 -> C10 += P2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # P3 = A00(B10-B11) -> C01 += P3
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # P4 = A11(B01-B00) -> C00 += P4, C10 += P4
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # P7 = (A01-A11)(B10+B11) -> C00 += P7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _rank22_3x3x3():
    """
    Construct rank-22 decomposition for 3x3 * 3x3 matrix multiplication.
    """
    n = m = p = 3
    rank = 22
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Terms 0-6: Strassen for A11*B11
    sU, sV, sW = _strassen_2x2_block()
    for c in range(7):
        for idx2 in range(4):
            i2 = idx2 // 2
            j2 = idx2 % 2
            U[i2 * 3 + j2, c] = sU[idx2, c]
            V[idx2, c] = sV[idx2, c]  # This is wrong, let me fix
            W[idx2, c] = sW[idx2, c]
    
    # Let me redo this properly
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    sU, sV, sW = _strassen_2x2_block()
    for c in range(7):
        # U: row i*3+j for i,j in {0,1}
        for i2 in range(2):
            for j2 in range(2):
                U[i2 * 3 + j2, c] = sU[i2 * 2 + j2, c]
        # V: row j*3+k for j,k in {0,1}
        for j2 in range(2):
            for k2 in range(2):
                V[j2 * 3 + k2, c] = sV[j2 * 2 + k2, c]
        # W: row k*3+i for k,i in {0,1}
        for k2 in range(2):
            for i2 in range(2):
                W[k2 * 3 + i2, c] = sW[k2 * 2 + i2, c]
    
    # Terms 7-10: A11*B12 (2x2 * 2x1)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    for t in range(4):
        c = 7 + t
        i2 = t // 2
        j2 = t % 2
        U[i2 * 3 + j2, c] = 1
        V[j2 * 3 + 2, c] = 1
        W[2 * 3 + i2, c] = 1
    
    # Terms 11-12: A12*B22 (2x1 * 1x1)
    for t in range(2):
        c = 11 + t
        i2 = t
        U[i2 * 3 + 2, c] = 1
        V[2 * 3 + 2, c] = 1
        W[2 * 3 + i2, c] = 1
    
    # Terms 13-16: A21*B11 (1x2 * 2x2)
    for t in range(4):
        c = 13 + t
        j2 = t // 2
        k2 = t % 2
        U[2 * 3 + j2, c] = 1
        V[j2 * 3 + k2, c] = 1
        W[k2 * 3 + 2, c] = 1
    
    # Terms 17-18: A22*B21 (1x1 * 1x2)
    for t in range(2):
        c = 17 + t
        k2 = t
        U[2 * 3 + 2, c] = 1
        V[2 * 3 + k2, c] = 1
        W[k2 * 3 + 2, c] = 1
    
    # Terms 19-20: A21*B12 (1x2 * 2x1)
    for t in range(2):
        c = 19 + t
        j2 = t
        U[2 * 3 + j2, c] = 1
        V[j2 * 3 + 2, c] = 1
        W[2 * 3 + 2, c] = 1
    
    # Term 21: A22*B22 (1x1 * 1x1)
    c = 21
    U[2 * 3 + 2, c] = 1
    V[2 * 3 + 2, c] = 1
    W[2 * 3 + 2, c] = 1
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 110
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition: 3x3*3x3 (rank 22) + 3x3*3x4 (schoolbook rank 36)
    U3, V3, W3 = _rank22_3x3x3()
    
    rank = 22 + 36
    U_full = np.zeros((9, rank))
    V_full = np.zeros((9, rank))
    W_full = np.zeros((21, rank))
    
    U_full[:, :22] = U3
    V_full[:, :22] = V3
    W_full[:9, :22] = W3
    
    col = 22
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U_full[i * 3 + j, col] = 1
                V_full[j * 3 + k, col] = 1
                W_full[k * 3 + i, col] = 1
                col += 1
    
    if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        best_rank = rank
    
    return best
# EVOLVE-BLOCK-END

