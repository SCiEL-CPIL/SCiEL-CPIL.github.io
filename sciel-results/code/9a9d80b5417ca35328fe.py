"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 16 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_U(i, j, r, val):
        U[i * m + j][r] = val
        
    def set_V(j, k, r, val):
        V[j * p + k][r] = val
        
    def set_W(k, i, r, val):
        W[k * n + i][r] = val

    # Part 1: A_2 B_2 (2x1x4)
    # r=0,1
    # r=0: i=0
    set_U(0, 2, 0, 1)
    for k in range(4):
        set_V(2, k, 0, 1)
        set_W(k, 0, 0, 1)
        
    # r=1: i=1
    set_U(1, 2, 1, 1)
    for k in range(4):
        set_V(2, k, 1, 1)
        set_W(k, 1, 1, 1)

    # Part 2: A_1 B_1 (2x2x4)
    # Split into B_1L (k=0,1) and B_1R (k=2,3)
    
    # Strassen factors for 2x2x2
    # s=1..7
    
    # Block 1: k_offset = 0
    k_off = 0
    r_base = 2
    
    # s=1: (A00+A11)(B00+B11)
    r = r_base + 0
    set_U(0, 0, r, 1); set_U(1, 1, r, 1)
    set_V(0, 0+k_off, r, 1); set_V(1, 1+k_off, r, 1)
    set_W(0+k_off, 0, r, 1); set_W(1+k_off, 1, r, 1)
    
    # s=2: (A10+A11)B00
    r = r_base + 1
    set_U(1, 0, r, 1); set_U(1, 1, r, 1)
    set_V(0, 0+k_off, r, 1)
    set_W(0+k_off, 1, r, 1); set_W(1+k_off, 1, r, 1)
    
    # s=3: A00(B01-B11)
    r = r_base + 2
    set_U(0, 0, r, 1)
    set_V(0, 1+k_off, r, 1); set_V(1, 1+k_off, r, -1)
    set_W(1+k_off, 0, r, 1); set_W(1+k_off, 1, r, -1)
    
    # s=4: A11(B10-B00)
    r = r_base + 3
    set_U(1, 1, r, 1)
    set_V(1, 0+k_off, r, 1); set_V(0, 0+k_off, r, -1)
    set_W(0+k_off, 0, r, -1); set_W(0+k_off, 1, r, 1)
    
    # s=5: (A00+A01)B11
    r = r_base + 4
    set_U(0, 0, r, 1); set_U(0, 1, r, 1)
    set_V(1, 1+k_off, r, 1)
    set_W(1+k_off, 0, r, 1); set_W(1+k_off, 1, r, 1)
    
    # s=6: (A10-A00)(B00+B01)
    r = r_base + 5
    set_U(1, 0, r, 1); set_U(0, 0, r, -1)
    set_V(0, 0+k_off, r, 1); set_V(0, 1+k_off, r, 1)
    set_W(0+k_off, 0, r, 1); set_W(0+k_off, 1, r, 1)
    
    # s=7: (A01-A11)(B10+B11)
    r = r_base + 6
    set_U(0, 1, r, 1); set_U(1, 1, r, -1)
    set_V(1, 0+k_off, r, 1); set_V(1, 1+k_off, r, 1)
    set_W(0+k_off, 0, r, 1); set_W(1+k_off, 0, r, 1)

    # Block 2: k_offset = 2
    k_off = 2
    r_base = 9
    
    # s=1
    r = r_base + 0
    set_U(0, 0, r, 1); set_U(1, 1, r, 1)
    set_V(0, 0+k_off, r, 1); set_V(1, 1+k_off, r, 1)
    set_W(0+k_off, 0, r, 1); set_W(1+k_off, 1, r, 1)
    
    # s=2
    r = r_base + 1
    set_U(1, 0, r, 1); set_U(1, 1, r, 1)
    set_V(0, 0+k_off, r, 1)
    set_W(0+k_off, 1, r, 1); set_W(1+k_off, 1, r, 1)
    
    # s=3
    r = r_base + 2
    set_U(0, 0, r, 1)
    set_V(0, 1+k_off, r, 1); set_V(1, 1+k_off, r, -1)
    set_W(1+k_off, 0, r, 1); set_W(1+k_off, 1, r, -1)
    
    # s=4
    r = r_base + 3
    set_U(1, 1, r, 1)
    set_V(1, 0+k_off, r, 1); set_V(0, 0+k_off, r, -1)
    set_W(0+k_off, 0, r, -1); set_W(0+k_off, 1, r, 1)
    
    # s=5
    r = r_base + 4
    set_U(0, 0, r, 1); set_U(0, 1, r, 1)
    set_V(1, 1+k_off, r, 1)
    set_W(1+k_off, 0, r, 1); set_W(1+k_off, 1, r, 1)
    
    # s=6
    r = r_base + 5
    set_U(1, 0, r, 1); set_U(0, 0, r, -1)
    set_V(0, 0+k_off, r, 1); set_V(0, 1+k_off, r, 1)
    set_W(0+k_off, 0, r, 1); set_W(0+k_off, 1, r, 1)
    
    # s=7
    r = r_base + 6
    set_U(0, 1, r, 1); set_U(1, 1, r, -1)
    set_V(1, 0+k_off, r, 1); set_V(1, 1+k_off, r, 1)
    set_W(0+k_off, 0, r, 1); set_W(1+k_off, 0, r, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
