"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks (28 mults)
    # Part 2: A2 (2x1) * B2 (1x8) using outer product (16 mults)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 6 x 44
    V = [[0] * rank for _ in range(m * p)] # 24 x 44
    W = [[0] * rank for _ in range(p * n)] # 16 x 44
    
    # Helper to add to U, V, W
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    col = 0
    
    # Part 1: Strassen for A1 * B1
    # A1 is A[0:2, 0:2] -> indices 0,1,3,4
    # B1 is B[0:2, 0:8] -> indices 0..7, 8..15
    # C1 is C[0:2, 0:8] -> indices 0..15 (k*n+i)
    
    for b in range(4):
        # Block b corresponds to B columns 2b, 2b+1
        # B indices for this block:
        # B_00: (0, 2b) -> 0*8 + 2b
        # B_01: (0, 2b+1) -> 0*8 + 2b + 1
        # B_10: (1, 2b) -> 1*8 + 2b
        # B_11: (1, 2b+1) -> 1*8 + 2b + 1
        
        b00_idx = 0 * 8 + 2 * b
        b01_idx = 0 * 8 + 2 * b + 1
        b10_idx = 1 * 8 + 2 * b
        b11_idx = 1 * 8 + 2 * b + 1
        
        # C indices for this block:
        # C_00: (i=0, k=2b) -> 2b*2 + 0
        # C_01: (i=0, k=2b+1) -> (2b+1)*2 + 0
        # C_10: (i=1, k=2b) -> 2b*2 + 1
        # C_11: (i=1, k=2b+1) -> (2b+1)*2 + 1
        
        c00_idx = 2 * b * 2 + 0
        c01_idx = (2 * b + 1) * 2 + 0
        c10_idx = 2 * b * 2 + 1
        c11_idx = (2 * b + 1) * 2 + 1
        
        # A indices for A1:
        # A_00: (0,0) -> 0
        # A_01: (0,1) -> 1
        # A_10: (1,0) -> 3
        # A_11: (1,1) -> 4
        
        a00_idx = 0
        a01_idx = 1
        a10_idx = 3
        a11_idx = 4
        
        # Strassen M1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00(+1), C11(+1)
        U[a00_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # Strassen M2: (A10+A11)B00
        # U: A10, A11
        # V: B00
        # W: C10(+1), C11(-1)
        U[a10_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # Strassen M3: A00(B01-B11)
        # U: A00
        # V: B01, -B11
        # W: C01(+1), C11(+1)
        U[a00_idx][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # Strassen M4: A11(B10-B00)
        # U: A11
        # V: B10, -B00
        # W: C00(+1), C10(+1)
        U[a11_idx][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # Strassen M5: (A00+A01)B11
        # U: A00, A01
        # V: B11
        # W: C00(-1), C01(+1)
        U[a00_idx][col] += 1
        U[a01_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # Strassen M6: (A10-A00)(B00+B01)
        # U: A10, -A00
        # V: B00, B01
        # W: C11(+1)
        U[a10_idx][col] += 1
        U[a00_idx][col] -= 1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # Strassen M7: (A01-A11)(B10+B11)
        # U: A01, -A11
        # V: B10, B11
        # W: C00(+1)
        U[a01_idx][col] += 1
        U[a11_idx][col] -= 1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    # Part 2: Outer product for A2 * B2
    # A2 is A[0:2, 2] -> indices 2, 5
    # B2 is B[2, 0:8] -> indices 16..23
    # C2 is C[0:2, 0:8] -> indices 0..15
    
    for i in range(2):
        a_idx = i * 3 + 2
        for k in range(8):
            b_idx = 2 * 8 + k
            c_idx = k * 2 + i
            
            U[a_idx][col] += 1
            V[b_idx][col] += 1
            W[c_idx][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
