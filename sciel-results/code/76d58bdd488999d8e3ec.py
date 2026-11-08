"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 16 (for 2x1x8) + 14 (for 2x2x4) + 14 (for 2x2x4) = 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # local_i, local_j for A (0..1, 0..1)
    # local_j, local_k for B (0..1, 0..1)
    # local_k, local_i for C (0..1, 0..1)
    # global offsets:
    # A: i_off, j_off
    # B: j_off, k_off
    # C: k_off, i_off
    def add_strassen(i_off, j_off, j_b_off, k_off, k_c_off, i_c_off):
        nonlocal col
        # Strassen products definition
        # Each entry is (u_val, v_val, w_val) for the 4 components?
        # No, U has 4 entries (2x2), V has 4, W has 4.
        # We define the 7 patterns.
        
        # Pattern 1: (a+d)(e+h) -> C00, C11
        # U: a=1, d=1
        # V: e=1, h=1
        # W: C00=1, C11=1
        U[i_off*2 + 0 + j_off*2 + 0][col] += 1 # a (i=0, j=0)
        U[i_off*2 + 1 + j_off*2 + 1][col] += 1 # d (i=1, j=1)
        V[j_b_off*2 + 0 + k_off*2 + 0][col] += 1 # e (j=0, k=0)
        V[j_b_off*2 + 1 + k_off*2 + 1][col] += 1 # h (j=1, k=1)
        W[k_c_off*2 + 0 + i_c_off*2 + 0][col] += 1 # C00 (k=0, i=0)
        W[k_c_off*2 + 1 + i_c_off*2 + 1][col] += 1 # C11 (k=1, i=1)
        col += 1
        
        # Pattern 2: (c+d)e -> C10, C11
        # U: c=1, d=1
        # V: e=1
        # W: C10=1, C11=-1
        U[i_off*2 + 1 + j_off*2 + 0][col] += 1 # c (i=1, j=0)
        U[i_off*2 + 1 + j_off*2 + 1][col] += 1 # d (i=1, j=1)
        V[j_b_off*2 + 0 + k_off*2 + 0][col] += 1 # e
        W[k_c_off*2 + 0 + i_c_off*2 + 1][col] += 1 # C10 (k=0, i=1)
        W[k_c_off*2 + 1 + i_c_off*2 + 1][col] += -1 # C11 (k=1, i=1)
        col += 1
        
        # Pattern 3: a(f-h) -> C01, C11
        # U: a=1
        # V: f=1, h=-1
        # W: C01=1, C11=1
        U[i_off*2 + 0 + j_off*2 + 0][col] += 1 # a
        V[j_b_off*2 + 0 + k_off*2 + 1][col] += 1 # f (j=0, k=1)
        V[j_b_off*2 + 1 + k_off*2 + 1][col] += -1 # h
        W[k_c_off*2 + 1 + i_c_off*2 + 0][col] += 1 # C01 (k=1, i=0)
        W[k_c_off*2 + 1 + i_c_off*2 + 1][col] += 1 # C11
        col += 1
        
        # Pattern 4: d(g-e) -> C00, C10
        # U: d=1
        # V: g=1, e=-1
        # W: C00=1, C10=1
        U[i_off*2 + 1 + j_off*2 + 1][col] += 1 # d
        V[j_b_off*2 + 1 + k_off*2 + 0][col] += 1 # g (j=1, k=0)
        V[j_b_off*2 + 0 + k_off*2 + 0][col] += -1 # e
        W[k_c_off*2 + 0 + i_c_off*2 + 0][col] += 1 # C00
        W[k_c_off*2 + 0 + i_c_off*2 + 1][col] += 1 # C10
        col += 1
        
        # Pattern 5: (a+b)h -> C00, C01
        # U: a=1, b=1
        # V: h=1
        # W: C00=-1, C01=1
        U[i_off*2 + 0 + j_off*2 + 0][col] += 1 # a
        U[i_off*2 + 0 + j_off*2 + 1][col] += 1 # b (i=0, j=1)
        V[j_b_off*2 + 1 + k_off*2 + 1][col] += 1 # h
        W[k_c_off*2 + 0 + i_c_off*2 + 0][col] += -1 # C00
        W[k_c_off*2 + 1 + i_c_off*2 + 0][col] += 1 # C01
        col += 1
        
        # Pattern 6: (c-a)(e+f) -> C11
        # U: c=1, a=-1
        # V: e=1, f=1
        # W: C11=1
        U[i_off*2 + 1 + j_off*2 + 0][col] += 1 # c
        U[i_off*2 + 0 + j_off*2 + 0][col] += -1 # a
        V[j_b_off*2 + 0 + k_off*2 + 0][col] += 1 # e
        V[j_b_off*2 + 0 + k_off*2 + 1][col] += 1 # f
        W[k_c_off*2 + 1 + i_c_off*2 + 1][col] += 1 # C11
        col += 1
        
        # Pattern 7: (b-d)(g+h) -> C00, C11
        # U: b=1, d=-1
        # V: g=1, h=1
        # W: C00=1, C11=1
        U[i_off*2 + 0 + j_off*2 + 1][col] += 1 # b
        U[i_off*2 + 1 + j_off*2 + 1][col] += -1 # d
        V[j_b_off*2 + 1 + k_off*2 + 0][col] += 1 # g
        V[j_b_off*2 + 1 + k_off*2 + 1][col] += 1 # h
        W[k_c_off*2 + 0 + i_c_off*2 + 0][col] += 1 # C00
        W[k_c_off*2 + 1 + i_c_off*2 + 1][col] += 1 # C11
        col += 1

    # Part 1: 2x1x8 multiplication (A col 2, B row 2)
    # A: i in 0..1, j=2. Global index i*3 + 2.
    # B: j=2, k in 0..7. Global index 2*8 + k.
    # C: i in 0..1, k in 0..7. Global index k*2 + i.
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col] = 1
            V[2 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: 2x2x4 multiplication (A cols 0,1; B rows 0,1; C cols 0..3)
    # Split into two 2x2x2 blocks.
    # Block 2a: C cols 0,1. B cols 0,1.
    # A: i_off=0, j_off=0.
    # B: j_off=0, k_off=0.
    # C: k_off=0, i_off=0.
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2b: C cols 2,3. B cols 2,3.
    # A: i_off=0, j_off=0.
    # B: j_off=0, k_off=2.
    # C: k_off=2, i_off=0.
    add_strassen(0, 0, 0, 2, 2, 0)

    # Part 3: 2x2x4 multiplication (A cols 0,1; B rows 0,1; C cols 4..7)
    # Split into two 2x2x2 blocks.
    # Block 3a: C cols 4,5. B cols 4,5.
    # A: i_off=0, j_off=0.
    # B: j_off=0, k_off=4.
    # C: k_off=4, i_off=0.
    add_strassen(0, 0, 0, 4, 4, 0)
    
    # Block 3b: C cols 6,7. B cols 6,7.
    # A: i_off=0, j_off=0.
    # B: j_off=0, k_off=6.
    # C: k_off=6, i_off=0.
    add_strassen(0, 0, 0, 6, 6, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
