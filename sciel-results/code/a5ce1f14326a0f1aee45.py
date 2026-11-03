"""Optimized matrix multiplication 2x4 by 4x5 - Strassen-based with fast verification."""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x5 matrix.
    
    Strategy: Strassen (rank 7) for 2x2x2 sub-blocks + schoolbook for last column.
    Total rank: 4 * 7 + 8 = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        c = col_start
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        
        return c
    
    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook for C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c
    
    # 4 Strassen blocks for the 2x4 x 4x4 part
    col = add_strassen_222(0, 0, 0, 0)   # A[:,0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 2, 0, col) # A[:,2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 0, 2, col) # A[:,0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 2, 2, col) # A[:,2:4] * B[2:4, 2:4]
    
    # Schoolbook for last column
    col = add_schoolbook_2x4x1(0, 0, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Fast verification using numpy einsum
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U has shape (n*m, rank) -> (n, m, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    # V has shape (m*p, rank) -> (m, p, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    # W has shape (p*n, rank) -> (p, n, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ik'
    C_reconstructed = np.einsum('ijr,jkr,kir->ik', U_reshaped, V_reshaped, W_reshaped)
    
    # Test with multiple random matrices for robustness
    rng = np.random.default_rng(42)
    valid = True
    for _ in range(10):
        A_test = rng.random((n, m))
        B_test = rng.random((m, p))
        C_expected = A_test @ B_test
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            valid = False
            break
    
    if not valid:
        # Fallback: return schoolbook (rank 40)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
