"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using 3 Strassen blocks
    rank = 21
    
    # Initialize matrices
    # U: (n*m) x rank = 4 x 21
    # V: (m*p) x rank = 12 x 21
    # W: (p*n) x rank = 12 x 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U: i*m + j
    # u00 = 0, u01 = 1, u10 = 2, u11 = 3
    u00, u01, u10, u11 = 0, 1, 2, 3
    
    # Indices for V: j*p + k
    # For block b, k in {2b, 2b+1}
    # v0_k = 0*6 + k = k
    # v1_k = 1*6 + k = 6 + k
    
    # Indices for W: k*n + i
    # For block b, output columns are 2b, 2b+1
    # C[2b, 0] -> w = 2b*2 + 0 = 4b
    # C[2b, 1] -> w = 2b*2 + 1 = 4b + 1
    # C[2b+1, 0] -> w = (2b+1)*2 + 0 = 4b + 2
    # C[2b+1, 1] -> w = (2b+1)*2 + 1 = 4b + 3
    
    for b in range(3):
        base_rank = b * 7
        
        # V indices for this block
        v11 = 2 * b       # j=0, k=2b
        v12 = 2 * b + 1   # j=0, k=2b+1
        v21 = 6 + 2 * b   # j=1, k=2b
        v22 = 6 + 2 * b + 1 # j=1, k=2b+1
        
        # W indices for this block
        w11 = 4 * b       # k=2b, i=0
        w12 = 4 * b + 1   # k=2b, i=1
        w21 = 4 * b + 2   # k=2b+1, i=0
        w22 = 4 * b + 3   # k=2b+1, i=1
        
        # M1: (A11 + A22) * (B11 + B22)
        r = base_rank
        U[u00][r] = 1
        U[u11][r] = 1
        V[v11][r] = 1
        V[v22][r] = 1
        W[w11][r] = 1
        W[w22][r] = 1
        
        # M2: (A21 + A22) * B11
        r = base_rank + 1
        U[u10][r] = 1
        U[u11][r] = 1
        V[v11][r] = 1
        W[w12][r] = 1
        W[w22][r] = -1
        
        # M3: A11 * (B12 - B22)
        r = base_rank + 2
        U[u00][r] = 1
        V[v12][r] = 1
        V[v22][r] = -1
        W[w21][r] = 1
        W[w22][r] = 1
        
        # M4: A22 * (B21 - B11)
        r = base_rank + 3
        U[u11][r] = 1
        V[v21][r] = 1
        V[v11][r] = -1
        W[w11][r] = 1
        W[w12][r] = 1
        
        # M5: (A11 + A12) * B22
        r = base_rank + 4
        U[u00][r] = 1
        U[u01][r] = 1
        V[v22][r] = 1
        W[w11][r] = -1
        W[w21][r] = 1
        
        # M6: (A21 - A11) * (B11 + B12)
        r = base_rank + 5
        U[u10][r] = 1
        U[u00][r] = -1
        V[v11][r] = 1
        V[v12][r] = 1
        W[w22][r] = 1
        
        # M7: (A12 - A22) * (B21 + B22)
        r = base_rank + 6
        U[u01][r] = 1
        U[u11][r] = -1
        V[v21][r] = 1
        V[v22][r] = 1
        W[w11][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
