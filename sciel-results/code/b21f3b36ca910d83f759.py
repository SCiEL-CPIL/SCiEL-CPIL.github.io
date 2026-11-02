"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 Strassen blocks for the 2x2 (A rows 0,1; cols 0,1) x 2x2 (B rows 0,1; cols pairs) part
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # B indices
        B00_idx = 0 * p + k0      # 2t
        B01_idx = 0 * p + k1      # 2t+1
        B10_idx = 1 * p + k0      # 10+2t
        B11_idx = 1 * p + k1      # 11+2t
        # C indices (W)
        C00_idx = k0 * n + 0      # 4t
        C01_idx = k1 * n + 0      # 4t+2
        C10_idx = k0 * n + 1      # 4t+1
        C11_idx = k1 * n + 1      # 4t+3
        # A indices (U)
        A00_idx = 0 * m + 0       # 0
        A01_idx = 0 * m + 1       # 1
        A10_idx = 1 * m + 0       # 3
        A11_idx = 1 * m + 1       # 4
        
        # M1 = (A00+A11)*(B00+B11)
        U[A00_idx][col] = 1
        U[A11_idx][col] = 1
        V[B00_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = 1
        W[C11_idx][col] = -1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[A10_idx][col] = 1
        U[A11_idx][col] = 1
        V[B00_idx][col] = 1
        W[C10_idx][col] = 1
        W[C11_idx][col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[A00_idx][col] = 1
        V[B01_idx][col] = 1
        V[B11_idx][col] = -1
        W[C01_idx][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[A11_idx][col] = 1
        V[B10_idx][col] = 1
        V[B00_idx][col] = -1
        W[C00_idx][col] = -1
        W[C01_idx][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[A00_idx][col] = 1
        U[A01_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = 1
        W[C10_idx][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[A10_idx][col] = 1
        U[A00_idx][col] = -1
        V[B00_idx][col] = 1
        V[B01_idx][col] = 1
        W[C11_idx][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U[A01_idx][col] = 1
        U[A11_idx][col] = -1
        V[B01_idx][col] = 1
        V[B11_idx][col] = 1
        W[C00_idx][col] = -1
        W[C01_idx][col] = 1
        col += 1
    
    # 20 terms for A[*,2] * B[2,*]
    for k in range(10):
        # A[0,2] * B[2,k] -> C[0,k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # A[1,2] * B[2,k] -> C[1,k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
