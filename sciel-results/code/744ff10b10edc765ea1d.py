"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 18
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 18
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 18
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        A00 = (i0+0)*m + (j0+0)
        A01 = (i0+0)*m + (j0+1)
        A10 = (i0+1)*m + (j0+0)
        A11 = (i0+1)*m + (j0+1)
        
        B00 = (j0+0)*p + (k0+0)
        B01 = (j0+0)*p + (k0+1)
        B10 = (j0+1)*p + (k0+0)
        B11 = (j0+1)*p + (k0+1)
        
        C00 = (k0+0)*n + (i0+0)
        C10 = (k0+0)*n + (i0+1)
        C01 = (k0+1)*n + (i0+0)
        C11 = (k0+1)*n + (i0+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] += -1
        W[C01][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] += -1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += -1; W[C01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] += -1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_221(i0, j0, k0):
        nonlocal col
        A00 = (i0+0)*m + (j0+0)
        A01 = (i0+0)*m + (j0+1)
        A10 = (i0+1)*m + (j0+0)
        A11 = (i0+1)*m + (j0+1)
        
        B0 = (j0+0)*p + k0
        B1 = (j0+1)*p + k0
        
        C0 = k0*n + (i0+0)
        C1 = k0*n + (i0+1)
        
        U[A00][col] += 1
        U[A10][col] += 1
        V[B0][col] += 1
        W[C0][col] += 1
        W[C1][col] += 1
        col += 1
        
        U[A01][col] += 1
        U[A11][col] += 1
        V[B1][col] += 1
        W[C0][col] += 1
        W[C1][col] += 1
        col += 1
    
    def add_215(i0, j0, k0_start, k0_end):
        nonlocal col
        A0 = (i0+0)*m + j0
        A1 = (i0+1)*m + j0
        
        U[A0][col] += 1
        U[A1][col] += 1
        for k in range(k0_start, k0_end):
            V[j0*p + k][col] += 1
            W[k*n + (i0+0)][col] += 1
            W[k*n + (i0+1)][col] += 1
        col += 1
    
    # Build decomposition
    add_strassen(0, 0, 0)   # 2×2×2, rank 7
    add_strassen(0, 0, 2)   # 2×2×2, rank 7
    add_221(0, 0, 4)        # 2×2×1, rank 2
    add_215(0, 2, 0, 5)     # 2×1×5, rank 1
    add_215(0, 3, 0, 5)     # 2×1×5, rank 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify with a random test
    if time.time() - start_time < 10:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_np = np.array(U).T  # rank x n*m
        V_np = np.array(V).T  # rank x m*p
        W_np = np.array(W).T  # rank x p*n
        
        # C[k,i] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        # Actually: C = sum_r W[:,r] * (U[:,r]^T @ V[:,r]) reshaped
        # Let me compute: for each r, the contribution is outer(U[:,r], V[:,r]) contracted
        # C[k,i] = sum_r W[k*n+i, r] * sum_{j} U[i*m+j, r] * V[j*p+k, r]
        
        C_computed = np.zeros((p, n))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    inner = 0
                    for j in range(m):
                        inner += U[i*m+j][r] * V[j*p+k][r]
                    C_computed[k, i] += W[k*n+i][r] * inner
        
        # Compare
        # C_expected[i,k] = C_computed[k,i]
        max_err = np.max(np.abs(C_expected - C_computed.T))
        if max_err > 1e-10:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_s = [[0.0] * rank_school for _ in range(n * m)]
            V_s = [[0.0] * rank_school for _ in range(m * p)]
            W_s = [[0.0] * rank_school for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m+j][c] = 1
                        V_s[j*p+k][c] = 1
                        W_s[k*n+i][c] = 1
                        c += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
