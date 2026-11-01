import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Trivial decomposition fallback (Rank 36)
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # Rank 32 decomposition: 4 Strassen blocks (cols 0-1, 2-3, 4-5, 6-7) + 1 schoolbook (col 8)
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen(k0, k1):
        nonlocal col
        # M1 = (A00+A11)*(B0k0+B1k1) -> C0k0 += 1, C1k1 += 1
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[p+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B0k0 -> C1k0 += 1, C1k1 -= 1
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        
        # M3 = A00*(B0k1-B1k1) -> C0k1 += 1, C1k1 += 1
        U[0][col] = 1
        V[k1][col] = 1; V[p+k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M4 = A11*(B1k0-B0k0) -> C0k0 += 1, C1k0 += 1
        U[3][col] = 1
        V[p+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B1k1 -> C0k0 -= 1, C0k1 += 1
        U[0][col] = 1; U[1][col] = 1
        V[p+k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B0k0+B0k1) -> C1k1 += 1
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B1k0+B1k1) -> C0k0 += 1
        U[1][col] = 1; U[3][col] = -1
        V[p+k0][col] = 1; V[p+k1][col] = 1
        W[2*k0][col] = 1
        col += 1

    # Helper to add a schoolbook block for column k
    def add_schoolbook(k):
        nonlocal col
        # S1 = A00*B0k -> C0k
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        # S2 = A01*B1k -> C0k
        U[1][col] = 1
        V[p+k][col] = 1
        W[2*k][col] = 1
        col += 1
        # S3 = A10*B0k -> C1k
        U[2][col] = 1
        V[k][col] = 1
        W[2*k+1][col] = 1
        col += 1
        # S4 = A11*B1k -> C1k
        U[3][col] = 1
        V[p+k][col] = 1
        W[2*k+1][col] = 1
        col += 1

    # Build the decomposition
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    if time.time() > deadline:
        return trivial_decomposition()

    # Verification using tensor contraction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return trivial_decomposition()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
