"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Rank 33 decomposition:
    # 12 terms for A[:,2] * B[2,:] (outer product)
    # 21 terms for A[:,:2] * B[:2,:] (3 blocks of 2x2x2 Strassen)
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # Part 1: A[:,2] * B[2,:]
    # i in 0..1, k in 0..5
    col = 0
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            U[i * m + 2][col] = 1
            # V index: 2*p + k
            V[2 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:,:2] * B[:2,:]
    # Split B[:2,:] into 3 blocks of 2x2: cols (0,1), (2,3), (4,5)
    for block in range(3):
        k_start = block * 2
        # Strassen for 2x2x2
        # U indices: 
        # A00 -> 0, A01 -> 1, A10 -> 3, A11 -> 4
        # V indices for this block:
        # B0, k_start -> 0*p + k_start
        # B0, k_start+1 -> 0*p + k_start + 1
        # B1, k_start -> 1*p + k_start
        # B1, k_start+1 -> 1*p + k_start + 1
        
        # W indices for this block:
        # C0, k_start -> k_start*n + 0
        # C0, k_start+1 -> (k_start+1)*n + 0
        # C1, k_start -> k_start*n + 1
        # C1, k_start+1 -> (k_start+1)*n + 1
        
        # Strassen Products M1..M7
        # M1 = A00 * (B0,ks + B0,ks+1)
        c = col
        U[0][c] = 1
        V[k_start][c] = 1
        V[k_start + 1][c] = 1
        W[k_start * n + 0][c] = 1
        W[(k_start + 1) * n + 0][c] = 1
        col += 1
        
        # M2 = (A00 + A01) * B1,ks
        c = col
        U[0][c] = 1
        U[1][c] = 1
        V[p + k_start][c] = 1
        W[(k_start + 1) * n + 1][c] = 1
        col += 1
        
        # M3 = (A10 + A11) * (B0,ks + B1,ks)
        c = col
        U[3][c] = 1
        U[4][c] = 1
        V[k_start][c] = 1
        V[p + k_start][c] = 1
        W[(k_start + 1) * n + 1][c] = -1
        col += 1
        
        # M4 = A11 * (B1,ks+1 - B1,ks)
        c = col
        U[4][c] = 1
        V[p + k_start + 1][c] = 1
        V[p + k_start][c] = -1
        W[k_start * n + 0][c] = 1
        W[(k_start + 1) * n + 0][c] = 1
        W[(k_start + 1) * n + 1][c] = 1
        col += 1
        
        # M5 = (A00 + A10) * B0,ks+1
        c = col
        U[0][c] = 1
        U[3][c] = 1
        V[k_start + 1][c] = 1
        W[k_start * n + 1][c] = 1
        W[(k_start + 1) * n + 1][c] = 1
        col += 1
        
        # M6 = (A01 - A11) * (B1,ks+1 + B1,ks)
        c = col
        U[1][c] = 1
        U[4][c] = -1
        V[p + k_start + 1][c] = 1
        V[p + k_start][c] = 1
        W[k_start * n + 0][c] = -1
        W[k_start * n + 1][c] = 1
        col += 1
        
        # M7 = (A01 + A11) * (B0,ks + B0,ks+1)
        c = col
        U[1][c] = 1
        U[4][c] = 1
        V[k_start][c] = 1
        V[k_start + 1][c] = 1
        W[k_start * n + 0][c] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
