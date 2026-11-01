"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x6)
    # A_L is A[0:2, 0:2]
    # B_T is B[0:2, 0:6]
    # Split B_T into 3 blocks of 2x2
    
    # Strassen coefficients for 2x2x2
    # M1: (x11+x22)(y11+y22)
    # M2: (x21+x22)y11
    # M3: x11(y12-y22)
    # M4: x22(y21-y11)
    # M5: (x11+x12)y22
    # M6: (x21-x11)(y11+y12)
    # M7: (x12-x22)(y21+y22)
    
    # Output Z
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    for block_idx in range(3):
        v_base = block_idx * 4
        w_z11 = 4 * block_idx
        w_z12 = 4 * block_idx + 2
        w_z21 = 4 * block_idx + 1
        w_z22 = 4 * block_idx + 3
        
        # M1
        U[0][col] += 1
        U[3][col] += 1
        V[v_base][col] += 1
        V[v_base + 3][col] += 1
        W[w_z11][col] += 1
        W[w_z22][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1
        U[3][col] += 1
        V[v_base][col] += 1
        W[w_z21][col] += 1
        W[w_z22][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[v_base + 1][col] += 1
        V[v_base + 3][col] -= 1
        W[w_z12][col] += 1
        W[w_z22][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[v_base + 2][col] += 1
        V[v_base][col] -= 1
        W[w_z11][col] += 1
        W[w_z21][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1
        U[1][col] += 1
        V[v_base + 3][col] += 1
        W[w_z11][col] -= 1
        W[w_z12][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1
        U[0][col] -= 1
        V[v_base][col] += 1
        V[v_base + 1][col] += 1
        W[w_z22][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1
        U[3][col] -= 1
        V[v_base + 2][col] += 1
        V[v_base + 3][col] += 1
        W[w_z11][col] += 1
        col += 1

    # Part 2: A_R (2x1) * B_B (1x6)
    for k in range(6):
        U[4][col] += 1
        V[12 + k][col] += 1
        W[2*k][col] += 1
        col += 1
        
        U[5][col] += 1
        V[12 + k][col] += 1
        W[2*k + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
