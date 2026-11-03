"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 blocks * 14 per block = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the linear combinations for U and V for each of the 7 multiplications
    # U coefficients for A indices (0,1,2,3) corresponding to a00, a01, a10, a11
    # V coefficients for B indices (0,1,2,3) corresponding to b00, b01, b10, b11
    
    # For a 2x2 block, A indices are local 0..3, B indices are local 0..3
    # A local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen terms definitions
    # term: (u_coeffs, v_coeffs)
    # u_coeffs: list of 4 values for a00, a01, a10, a11
    # v_coeffs: list of 4 values for b00, b01, b10, b11
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1]),
    ]
    
    # W coefficients for C indices (0,1,2,3) corresponding to c00, c01, c10, c11
    # C local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    w_coeffs = [
        # M1
        [1, 0, 0, 1],
        # M2
        [0, 0, 1, -1],
        # M3
        [0, 1, 0, 1],
        # M4
        [1, 0, 1, 0],
        # M5
        [-1, 1, 0, 0],
        # M6
        [0, 0, 0, 1],
        # M7
        [1, 0, 0, 0],
    ]
    
    # 3 blocks of B columns
    for g in range(3):
        base_col = g * 14
        # Block 1: A1 (cols 0,1 of A) * B_g1 (rows 0,1 of B, cols 2g, 2g+1)
        # A1 global indices: 0, 1, 2, 3 (i=0,1; j=0,1)
        # B_g1 global indices: rows 0,1; cols 2g, 2g+1
        # B global index: row * p + col
        
        for s in range(7):
            col = base_col + s
            u_c, v_c = strassen_terms[s]
            w_c = w_coeffs[s]
            
            # Fill U
            # A1 indices: i=0,1; j=0,1
            # Local A index 0 -> Global (0,0) -> 0
            # Local A index 1 -> Global (0,1) -> 1
            # Local A index 2 -> Global (1,0) -> 2
            # Local A index 3 -> Global (1,1) -> 3
            
            U[0][col] += u_c[0]
            U[1][col] += u_c[1]
            U[2][col] += u_c[2]
            U[3][col] += u_c[3]
            
            # Fill V
            # B_g1 indices: i=0,1; j=2g, 2g+1
            # Local B index 0 -> Global (0, 2g) -> 0*6 + 2g
            # Local B index 1 -> Global (0, 2g+1) -> 0*6 + 2g+1
            # Local B index 2 -> Global (1, 2g) -> 1*6 + 2g
            # Local B index 3 -> Global (1, 2g+1) -> 1*6 + 2g+1
            
            V[0 * 6 + 2 * g][col] += v_c[0]
            V[0 * 6 + 2 * g + 1][col] += v_c[1]
            V[1 * 6 + 2 * g][col] += v_c[2]
            V[1 * 6 + 2 * g + 1][col] += v_c[3]
            
            # Fill W
            # C_g indices: i=0,1; k=2g, 2g+1
            # W index: k * n + i
            # Local C index 0 -> Global (0, 2g) -> 2g * 2 + 0
            # Local C index 1 -> Global (0, 2g+1) -> (2g+1) * 2 + 0
            # Local C index 2 -> Global (1, 2g) -> 2g * 2 + 1
            # Local C index 3 -> Global (1, 2g+1) -> (2g+1) * 2 + 1
            
            W[2 * g * 2 + 0][col] += w_c[0]
            W[(2 * g + 1) * 2 + 0][col] += w_c[1]
            W[2 * g * 2 + 1][col] += w_c[2]
            W[(2 * g + 1) * 2 + 1][col] += w_c[3]
            
        # Block 2: A2 (cols 2,3 of A) * B_g2 (rows 2,3 of B, cols 2g, 2g+1)
        # A2 global indices: 4, 5, 6, 7
        # B_g2 global indices: rows 2,3; cols 2g, 2g+1
        
        for s in range(7):
            col = base_col + 7 + s
            u_c, v_c = strassen_terms[s]
            w_c = w_coeffs[s]
            
            # Fill U
            # A2 indices: i=0,1; j=2,3
            # Local A index 0 -> Global (0,2) -> 4
            # Local A index 1 -> Global (0,3) -> 5
            # Local A index 2 -> Global (1,2) -> 6
            # Local A index 3 -> Global (1,3) -> 7
            
            U[4][col] += u_c[0]
            U[5][col] += u_c[1]
            U[6][col] += u_c[2]
            U[7][col] += u_c[3]
            
            # Fill V
            # B_g2 indices: i=2,3; j=2g, 2g+1
            # Local B index 0 -> Global (2, 2g) -> 2*6 + 2g
            # Local B index 1 -> Global (2, 2g+1) -> 2*6 + 2g+1
            # Local B index 2 -> Global (3, 2g) -> 3*6 + 2g
            # Local B index 3 -> Global (3, 2g+1) -> 3*6 + 2g+1
            
            V[2 * 6 + 2 * g][col] += v_c[0]
            V[2 * 6 + 2 * g + 1][col] += v_c[1]
            V[3 * 6 + 2 * g][col] += v_c[2]
            V[3 * 6 + 2 * g + 1][col] += v_c[3]
            
            # Fill W
            # Same C_g indices as before
            
            W[2 * g * 2 + 0][col] += w_c[0]
            W[(2 * g + 1) * 2 + 0][col] += w_c[1]
            W[2 * g * 2 + 1][col] += w_c[2]
            W[(2 * g + 1) * 2 + 1][col] += w_c[3]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
