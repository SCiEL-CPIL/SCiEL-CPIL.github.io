"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    # Split m=3 into 1+2.
    # Term 1: A[:, 0] * B[0, :] (2x1x7, Rank 14)
    # Term 2: A[:, 1:3] * B[1:3, :] (2x2x7)
    #   Split p=7 into 2+2+2+1.
    #   Three 2x2x2 blocks (Rank 7 each, total 21)
    #   One 2x2x1 block (Rank 4)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # i in 0..1, k in 0..6
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # A_sub indices in U: 
    # A[0,1] -> U[1], A[0,2] -> U[2]
    # A[1,1] -> U[4], A[1,2] -> U[5]
    # B_sub indices in V:
    # B[1,k] -> V[7+k], B[2,k] -> V[14+k]
    # C indices in W:
    # C[i,k] -> W[2k+i]
    
    # Three 2x2x2 Strassen blocks for k in {0,1}, {2,3}, {4,5}
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # W indices for this block
        w00 = k0 * n + 0  # C[0, k0]
        w01 = k1 * n + 0  # C[0, k1]
        w10 = k0 * n + 1  # C[1, k0]
        w11 = k1 * n + 1  # C[1, k1]
        
        # V indices
        v00 = 1 * p + k0  # B[1, k0]
        v01 = 1 * p + k1  # B[1, k1]
        v10 = 2 * p + k0  # B[2, k0]
        v11 = 2 * p + k1  # B[2, k1]
        
        # U indices
        u00 = 0 * m + 1  # A[0, 1]
        u01 = 0 * m + 2  # A[0, 2]
        u10 = 1 * m + 1  # A[1, 1]
        u11 = 1 * m + 2  # A[1, 2]
        
        # Strassen Terms
        # 1. M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # 2. M2 = (A10+A11)B00
        U[u10][col] = 1
        U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # 3. M3 = A00(B10-B11)
        U[u00][col] = 1
        V[v10][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # 4. M4 = A11(B00-B01)
        U[u11][col] = 1
        V[v00][col] = 1
        V[v01][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # 5. M5 = (A00+A01)B11
        U[u00][col] = 1
        U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # 6. M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1
        U[u00][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # 7. M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1
        U[u11][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # One 2x2x1 block for k=6
    k0 = 6
    w0 = k0 * n + 0  # C[0, 6]
    w1 = k0 * n + 1  # C[1, 6]
    
    v0 = 1 * p + k0  # B[1, 6]
    v1 = 2 * p + k0  # B[2, 6]
    
    # 1. A00 B0 -> C0
    U[u00][col] = 1
    V[v0][col] = 1
    W[w0][col] = 1
    col += 1
    
    # 2. A01 B1 -> C0
    U[u01][col] = 1
    V[v1][col] = 1
    W[w0][col] = 1
    col += 1
    
    # 3. A10 B0 -> C1
    U[u10][col] = 1
    V[v0][col] = 1
    W[w1][col] = 1
    col += 1
    
    # 4. A11 B1 -> C1
    U[u11][col] = 1
    V[v1][col] = 1
    W[w1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
