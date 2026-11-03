import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    The decomposition uses 4 independent 2x2*2x2 Strassen blocks.
    Each block handles a 2x2 submatrix of A multiplied by a 2x2 submatrix of B,
    contributing to a 2x2 submatrix of C.
    
    U: shape (n*m, rank) = (8, 28)
    V: shape (m*p, rank) = (16, 28)
    W: shape (p*n, rank) = (8, 28)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i, j]
    V[i*p + j, r] corresponds to B[i, j]
    W[k*n + i, r] corresponds to C[i, k] (row i, column k)
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_r):
        """
        Adds Strassen decomposition for a 2x2 * 2x2 -> 2x2 block product.
        
        A_block: rows a_r0..a_r0+1, cols a_c0..a_c0+1
        B_block: rows b_r0..b_r0+1, cols b_c0..b_c0+1
        C_block: rows c_r0..c_r0+1, cols c_c0..c_c0+1
        """
        # A element indices
        a11 = a_r0 * m + a_c0
        a12 = a_r0 * m + a_c0 + 1
        a21 = (a_r0 + 1) * m + a_c0
        a22 = (a_r0 + 1) * m + a_c0 + 1
        
        # B element indices
        b11 = b_r0 * p + b_c0
        b12 = b_r0 * p + b_c0 + 1
        b21 = (b_r0 + 1) * p + b_c0
        b22 = (b_r0 + 1) * p + b_c0 + 1
        
        # C element indices (W[k*n + i, r] for C[i, k])
        c11_idx = c_c0 * n + c_r0
        c12_idx = (c_c0 + 1) * n + c_r0
        c21_idx = c_c0 * n + (c_r0 + 1)
        c22_idx = (c_c0 + 1) * n + (c_r0 + 1)
        
        r = start_r
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21][r] += 1
        U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1
        W[c22_idx][r] += -1
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11][r] += 1
        V[b12][r] += 1
        V[b22][r] += -1
        W[c12_idx][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22][r] += 1
        V[b21][r] += 1
        V[b11][r] += -1
        W[c11_idx][r] += 1
        W[c21_idx][r] += 1
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11][r] += 1
        U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1
        W[c12_idx][r] += 1
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21][r] += 1
        U[a11][r] += -1
        V[b11][r] += 1
        V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12][r] += 1
        U[a22][r] += -1
        V[b21][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        
        return r
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    r = add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    r = add_strassen_block(0, 2, 2, 0, 0, 0, r)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    r = add_strassen_block(0, 0, 0, 2, 0, 2, r)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    r = add_strassen_block(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Expected rank 28, got {r}"
    
    return {"U": U, "V": V, "W": W}
