"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # X is 2x2, Y is 2x2
    # X indices in A: (i, j) -> U[i*m+j]
    # Y indices in B: (j, k) -> V[j*p+k]
    # Z indices in C: (i, k) -> W[k*n+i]
    # Strassen products M1..M7
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)(y00+y11)
    # M3 = x00(y11-y01)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)(y11-y10)
    # M6 = (x10+x00)(y00-y10)
    # M7 = (x01+x11)(y01-y00)
    # Z00 = M1 - M4 + M5 - M7
    # Z01 = M3 + M1 - M5 - M6
    # Z10 = M2 + M4 - M6 - M7
    # Z11 = M1 - M2 + M3 + M4
    
    def add_strassen_block(x_i0, x_j0, x_i1, x_j1, y_j0, y_k0, y_j1, y_k1, z_i0, z_k0, z_i1, z_k1, base_col):
        # x_i0, x_j0 is top-left of X (i0, j0)
        # x_i1, x_j1 is bottom-right of X (i1, j1)
        # Actually X is 2x2.
        # X = [[A[i0][j0], A[i0][j1]], [A[i1][j0], A[i1][j1]]]
        # Y = [[B[j0][k0], B[j0][k1]], [B[j1][k0], B[j1][k1]]]
        
        # M1
        U[x_i0*m + x_j0][base_col + 0] += 1
        U[x_i1*m + x_j1][base_col + 0] += 1
        V[y_j0*p + y_k0][base_col + 0] += 1
        V[y_j1*p + y_k1][base_col + 0] += 1
        # M1 contributes to Z00 (+1), Z01 (+1), Z11 (+1)
        W[z_k0*n + z_i0][base_col + 0] += 1
        W[z_k1*n + z_i0][base_col + 0] += 1
        W[z_k1*n + z_i1][base_col + 0] += 1
        
        # M2
        U[x_i1*m + x_j0][base_col + 1] += 1
        U[x_i1*m + x_j1][base_col + 1] += 1
        V[y_j0*p + y_k0][base_col + 1] += 1
        V[y_j1*p + y_k1][base_col + 1] += 1
        # M2 contributes to Z10 (+1), Z11 (-1)
        W[z_k0*n + z_i1][base_col + 1] += 1
        W[z_k1*n + z_i1][base_col + 1] -= 1
        
        # M3
        U[x_i0*m + x_j0][base_col + 2] += 1
        V[y_j1*p + y_k1][base_col + 2] += 1
        V[y_j0*p + y_k1][base_col + 2] -= 1
        # M3 contributes to Z01 (+1), Z11 (+1)
        W[z_k1*n + z_i0][base_col + 2] += 1
        W[z_k1*n + z_i1][base_col + 2] += 1
        
        # M4
        U[x_i1*m + x_j1][base_col + 3] += 1
        V[y_j1*p + y_k0][base_col + 3] += 1
        V[y_j0*p + y_k0][base_col + 3] -= 1
        # M4 contributes to Z00 (-1), Z10 (+1), Z11 (+1)
        W[z_k0*n + z_i0][base_col + 3] -= 1
        W[z_k0*n + z_i1][base_col + 3] += 1
        W[z_k1*n + z_i1][base_col + 3] += 1
        
        # M5
        U[x_i0*m + x_j0][base_col + 4] += 1
        U[x_i0*m + x_j1][base_col + 4] += 1
        V[y_j1*p + y_k1][base_col + 4] += 1
        V[y_j1*p + y_k0][base_col + 4] -= 1
        # M5 contributes to Z00 (+1), Z01 (-1)
        W[z_k0*n + z_i0][base_col + 4] += 1
        W[z_k1*n + z_i0][base_col + 4] -= 1
        
        # M6
        U[x_i1*m + x_j0][base_col + 5] += 1
        U[x_i0*m + x_j0][base_col + 5] += 1
        V[y_j0*p + y_k0][base_col + 5] += 1
        V[y_j1*p + y_k0][base_col + 5] -= 1
        # M6 contributes to Z01 (-1), Z10 (-1)
        W[z_k1*n + z_i0][base_col + 5] -= 1
        W[z_k0*n + z_i1][base_col + 5] -= 1
        
        # M7
        U[x_i0*m + x_j1][base_col + 6] += 1
        U[x_i1*m + x_j1][base_col + 6] += 1
        V[y_j0*p + y_k1][base_col + 6] += 1
        V[y_j0*p + y_k0][base_col + 6] -= 1
        # M7 contributes to Z00 (-1), Z10 (-1)
        W[z_k0*n + z_i0][base_col + 6] -= 1
        W[z_k0*n + z_i1][base_col + 6] -= 1

    # Block 1: A1 (i=0,1; j=0,1) * B1a (j=0,1; k=0,1) -> C_left (i=0,1; k=0,1)
    # A1: i0=0, i1=1, j0=0, j1=1
    # B1a: j0=0, j1=1, k0=0, k1=1
    # C_left: i0=0, i1=1, k0=0, k1=1
    add_strassen_block(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0)
    
    # Block 2: A1 (i=0,1; j=0,1) * B1b (j=2,3; k=0,1) -> C_left (i=0,1; k=0,1)
    # B1b: j0=2, j1=3, k0=0, k1=1
    add_strassen_block(0, 0, 1, 1, 2, 0, 3, 1, 0, 0, 1, 1, 7)
    
    # Block 3: A2 (i=0,1; j=2,3) * B2a (j=0,1; k=2,3) -> C_right (i=0,1; k=2,3)
    # A2: i0=0, i1=1, j0=2, j1=3
    # B2a: j0=0, j1=1, k0=2, k1=3
    # C_right: i0=0, i1=1, k0=2, k1=3
    add_strassen_block(0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 14)
    
    # Block 4: A2 (i=0,1; j=2,3) * B2b (j=2,3; k=2,3) -> C_right (i=0,1; k=2,3)
    # B2b: j0=2, j1=3, k0=2, k1=3
    add_strassen_block(0, 2, 1, 3, 2, 2, 3, 3, 0, 2, 1, 3, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
