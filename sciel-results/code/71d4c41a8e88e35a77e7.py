"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for M1..M7
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = (x00+x01)(y10-y11)
    # M4 = (x10+x11)(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Define the 6 blocks
    # A_L (j=0,1), A_R (j=2,3)
    # B_T (j=0,1), B_B (j=2,3)
    # B cols: 0-1, 2-3, 4-5
    
    blocks = [
        # (j_offset_A, j_offset_B, k_offset)
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (2, 2, 0),
        (2, 2, 2),
        (2, 2, 4)
    ]
    
    for j_off_a, j_off_b, k_off in blocks:
        # Local indices for A: 0,1 -> Global j: j_off_a, j_off_a+1
        # Local indices for B: 0,1 -> Global j: j_off_b, j_off_b+1
        # Local indices for C: 0,1 -> Global k: k_off, k_off+1
        
        # M1
        # U: A(0, j_off_a) + A(1, j_off_a+1)
        # V: B(j_off_b, k_off) + B(j_off_b+1, k_off+1)
        # W: C(0, k_off) + C(0, k_off+1) + C(1, k_off) + C(1, k_off+1) ? No, W is for each M.
        # W_{k n + i, r} = coefficient of C_{i, k} in the sum of M's.
        # For M1, it contributes to Z00, Z11.
        # Z00 = M1 ... => W_{k_off, 0} += 1
        # Z11 = M1 ... => W_{k_off+1, 1} += 1
        
        # M1
        U[0 * m + j_off_a][col] += 1
        U[1 * m + j_off_a + 1][col] += 1
        V[j_off_b * p + k_off][col] += 1
        V[(j_off_b + 1) * p + k_off + 1][col] += 1
        W[k_off * n + 0][col] += 1
        W[(k_off + 1) * n + 1][col] += 1
        col += 1
        
        # M2
        # (x10+x11)y00
        U[1 * m + j_off_a][col] += 1
        U[1 * m + j_off_a + 1][col] += 1
        V[j_off_b * p + k_off][col] += 1
        # Z10 = M2 ... => W_{k_off, 1} += 1
        # Z11 = -M2 ... => W_{k_off+1, 1} -= 1
        W[k_off * n + 1][col] += 1
        W[(k_off + 1) * n + 1][col] -= 1
        col += 1
        
        # M3
        # (x00+x01)(y10-y11)
        U[0 * m + j_off_a][col] += 1
        U[0 * m + j_off_a + 1][col] += 1
        V[(j_off_b + 1) * p + k_off][col] += 1
        V[(j_off_b + 1) * p + k_off + 1][col] -= 1
        # Z01 = M3 ... => W_{k_off+1, 0} += 1
        # Z11 = M3 ... => W_{k_off+1, 1} += 1
        W[(k_off + 1) * n + 0][col] += 1
        W[(k_off + 1) * n + 1][col] += 1
        col += 1
        
        # M4
        # (x10+x11)(y01-y00)
        U[1 * m + j_off_a][col] += 1
        U[1 * m + j_off_a + 1][col] += 1
        V[j_off_b * p + k_off + 1][col] += 1
        V[j_off_b * p + k_off][col] -= 1
        # Z00 = M4 ... => W_{k_off, 0} += 1
        # Z10 = M4 ... => W_{k_off, 1} += 1
        W[k_off * n + 0][col] += 1
        W[k_off * n + 1][col] += 1
        col += 1
        
        # M5
        # (x00+x01)y11
        U[0 * m + j_off_a][col] += 1
        U[0 * m + j_off_a + 1][col] += 1
        V[(j_off_b + 1) * p + k_off + 1][col] += 1
        # Z00 = -M5 ... => W_{k_off, 0} -= 1
        # Z01 = M5 ... => W_{k_off+1, 0} += 1
        W[k_off * n + 0][col] -= 1
        W[(k_off + 1) * n + 0][col] += 1
        col += 1
        
        # M6
        # (x10-x00)(y00+y01)
        U[1 * m + j_off_a][col] += 1
        U[0 * m + j_off_a][col] -= 1
        V[j_off_b * p + k_off][col] += 1
        V[j_off_b * p + k_off + 1][col] += 1
        # Z11 = M6 ... => W_{k_off+1, 1} += 1
        W[(k_off + 1) * n + 1][col] += 1
        col += 1
        
        # M7
        # (x01-x11)(y10+y11)
        U[0 * m + j_off_a + 1][col] += 1
        U[1 * m + j_off_a + 1][col] -= 1
        V[(j_off_b + 1) * p + k_off][col] += 1
        V[(j_off_b + 1) * p + k_off + 1][col] += 1
        # Z00 = M7 ... => W_{k_off, 0} += 1
        W[k_off * n + 0][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
