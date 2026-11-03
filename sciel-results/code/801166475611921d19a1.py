"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Part 1: C1 (Row 0)
    # A1L (1x2) * BL (2x8) + A1R (1x2) * BR (2x8)
    # A1L: U[0], U[1]
    # A1R: U[2], U[3]
    # BL: V[0..15]
    # BR: V[16..31]
    
    for k in range(8):
        # A[0,0] * B[0,k] -> W[k*3 + 0]
        add_term(0, 1, k, 1, k * n, 1)
        # A[0,1] * B[1,k] -> W[k*3 + 0]
        add_term(1, 1, 8 + k, 1, k * n, 1)
        # A[0,2] * B[2,k] -> W[k*3 + 0]
        add_term(2, 1, 16 + k, 1, k * n, 1)
        # A[0,3] * B[3,k] -> W[k*3 + 0]
        add_term(3, 1, 24 + k, 1, k * n, 1)
        
    # Part 2: C2 (Rows 1, 2)
    # A2L (2x2) * BL (2x8) + A2R (2x2) * BR (2x8)
    # A2L: U[4], U[5], U[8], U[9]
    # A2R: U[6], U[7], U[10], U[11]
    # BL: V[0..15]
    # BR: V[16..31]
    
    # Process A2L * BL
    for b in range(4):
        # V block indices: 4b to 4b+3
        v_base = 4 * b
        # W block indices:
        # C[1, 2b] -> W[2b*3 + 1]
        # C[1, 2b+1] -> W[(2b+1)*3 + 1]
        # C[2, 2b] -> W[2b*3 + 2]
        # C[2, 2b+1] -> W[(2b+1)*3 + 2]
        w11 = (2 * b) * n + 1
        w12 = (2 * b + 1) * n + 1
        w21 = (2 * b) * n + 2
        w22 = (2 * b + 1) * n + 2
        
        # U indices for A2L
        u11 = 4
        u12 = 5
        u21 = 8
        u22 = 9
        
        # Strassen Terms
        # M1 = (U11 + U22)(V11 + V22)
        add_term(u11, 1, v_base, 1, w11, 1)
        add_term(u11, 1, v_base + 3, 1, w11, 1)
        add_term(u22, 1, v_base, 1, w11, 1)
        add_term(u22, 1, v_base + 3, 1, w11, 1)
        add_term(u11, 1, v_base, 1, w12, 1)
        add_term(u11, 1, v_base + 3, 1, w12, 1)
        add_term(u22, 1, v_base, 1, w12, 1)
        add_term(u22, 1, v_base + 3, 1, w12, 1)
        add_term(u11, 1, v_base, 1, w21, 1)
        add_term(u11, 1, v_base + 3, 1, w21, 1)
        add_term(u22, 1, v_base, 1, w21, 1)
        add_term(u22, 1, v_base + 3, 1, w21, 1)
        add_term(u11, 1, v_base, 1, w22, 1)
        add_term(u11, 1, v_base + 3, 1, w22, 1)
        add_term(u22, 1, v_base, 1, w22, 1)
        add_term(u22, 1, v_base + 3, 1, w22, 1)
        
        # M2 = (U21 + U22)V11
        add_term(u21, 1, v_base, 1, w21, 1)
        add_term(u21, 1, v_base, 1, w22, -1)
        add_term(u22, 1, v_base, 1, w21, 1)
        add_term(u22, 1, v_base, 1, w22, -1)
        
        # M3 = U11(V12 - V22)
        add_term(u11, 1, v_base + 1, 1, w12, 1)
        add_term(u11, 1, v_base + 3, -1, w12, 1)
        add_term(u11, 1, v_base + 1, 1, w22, 1)
        add_term(u11, 1, v_base + 3, -1, w22, 1)
        
        # M4 = U22(V21 - V11)
        add_term(u22, 1, v_base + 2, 1, w11, 1)
        add_term(u22, 1, v_base, -1, w11, 1)
        add_term(u22, 1, v_base + 2, 1, w21, 1)
        add_term(u22, 1, v_base, -1, w21, 1)
        
        # M5 = (U11 + U12)V22
        add_term(u11, 1, v_base + 3, 1, w11, -1)
        add_term(u11, 1, v_base + 3, 1, w12, 1)
        add_term(u12, 1, v_base + 3, 1, w11, -1)
        add_term(u12, 1, v_base + 3, 1, w12, 1)
        
        # M6 = (U21 - U11)(V11 + V12)
        add_term(u21, 1, v_base, 1, w22, 1)
        add_term(u21, 1, v_base + 1, 1, w22, 1)
        add_term(u11, -1, v_base, 1, w22, 1)
        add_term(u11, -1, v_base + 1, 1, w22, 1)
        
        # M7 = (U12 - U22)(V21 + V22)
        add_term(u12, 1, v_base + 2, 1, w11, 1)
        add_term(u12, 1, v_base + 3, 1, w11, 1)
        add_term(u22, -1, v_base + 2, 1, w11, 1)
        add_term(u22, -1, v_base + 3, 1, w11, 1)

    # Process A2R * BR
    for b in range(4):
        v_base = 16 + 4 * b
        w11 = (2 * b) * n + 1
        w12 = (2 * b + 1) * n + 1
        w21 = (2 * b) * n + 2
        w22 = (2 * b + 1) * n + 2
        
        u11 = 6
        u12 = 7
        u21 = 10
        u22 = 11
        
        # M1
        add_term(u11, 1, v_base, 1, w11, 1)
        add_term(u11, 1, v_base + 3, 1, w11, 1)
        add_term(u22, 1, v_base, 1, w11, 1)
        add_term(u22, 1, v_base + 3, 1, w11, 1)
        add_term(u11, 1, v_base, 1, w12, 1)
        add_term(u11, 1, v_base + 3, 1, w12, 1)
        add_term(u22, 1, v_base, 1, w12, 1)
        add_term(u22, 1, v_base + 3, 1, w12, 1)
        add_term(u11, 1, v_base, 1, w21, 1)
        add_term(u11, 1, v_base + 3, 1, w21, 1)
        add_term(u22, 1, v_base, 1, w21, 1)
        add_term(u22, 1, v_base + 3, 1, w21, 1)
        add_term(u11, 1, v_base, 1, w22, 1)
        add_term(u11, 1, v_base + 3, 1, w22, 1)
        add_term(u22, 1, v_base, 1, w22, 1)
        add_term(u22, 1, v_base + 3, 1, w22, 1)
        
        # M2
        add_term(u21, 1, v_base, 1, w21, 1)
        add_term(u21, 1, v_base, 1, w22, -1)
        add_term(u22, 1, v_base, 1, w21, 1)
        add_term(u22, 1, v_base, 1, w22, -1)
        
        # M3
        add_term(u11, 1, v_base + 1, 1, w12, 1)
        add_term(u11, 1, v_base + 3, -1, w12, 1)
        add_term(u11, 1, v_base + 1, 1, w22, 1)
        add_term(u11, 1, v_base + 3, -1, w22, 1)
        
        # M4
        add_term(u22, 1, v_base + 2, 1, w11, 1)
        add_term(u22, 1, v_base, -1, w11, 1)
        add_term(u22, 1, v_base + 2, 1, w21, 1)
        add_term(u22, 1, v_base, -1, w21, 1)
        
        # M5
        add_term(u11, 1, v_base + 3, 1, w11, -1)
        add_term(u11, 1, v_base + 3, 1, w12, 1)
        add_term(u12, 1, v_base + 3, 1, w11, -1)
        add_term(u12, 1, v_base + 3, 1, w12, 1)
        
        # M6
        add_term(u21, 1, v_base, 1, w22, 1)
        add_term(u21, 1, v_base + 1, 1, w22, 1)
        add_term(u11, -1, v_base, 1, w22, 1)
        add_term(u11, -1, v_base + 1, 1, w22, 1)
        
        # M7
        add_term(u12, 1, v_base + 2, 1, w11, 1)
        add_term(u12, 1, v_base + 3, 1, w11, 1)
        add_term(u22, -1, v_base + 2, 1, w11, 1)
        add_term(u22, -1, v_base + 3, 1, w11, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
