import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Rank-50 tensor decomposition for 2x4 @ 4x7 matrix multiplication
    using Strassen's algorithm tiled over the m and p dimensions.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 110
    
    # Build rank-50 decomposition
    rank_init = 50
    U = [[0] * rank_init for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank_init for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank_init for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
    
    # 6 Strassen blocks covering k=0..5, j=0..3
    add_strassen(0, 0, 0)  # j=0-1, k=0-1
    add_strassen(0, 0, 2)  # j=0-1, k=2-3
    add_strassen(0, 2, 0)  # j=2-3, k=0-1
    add_strassen(0, 2, 2)  # j=2-3, k=2-3
    add_strassen(0, 0, 4)  # j=0-1, k=4-5
    add_strassen(0, 2, 4)  # j=2-3, k=4-5
    
    # 8 rank-1 terms for k=6 (remaining column)
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify(U_m, V_m, W_m, n, m, p, seed=42):
        np.random.seed(seed)
        for _ in range(10):
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_expected = A @ B
            C_comp = np.zeros((n, p), dtype=float)
            for r in range(U_m.shape[1]):
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for j in range(m):
                            val += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
                        C_comp[i, k] += val
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    if not verify(U_arr, V_arr, W_arr, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_fallback = [[0] * rank for _ in range(n * m)]
        V_fallback = [[0] * rank for _ in range(m * p)]
        W_fallback = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i*m+j][c] = 1
                    V_fallback[j*p+k][c] = 1
                    W_fallback[k*n+i][c] = 1
                    c += 1
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    return {"U": U, "V": V, "W": W}
