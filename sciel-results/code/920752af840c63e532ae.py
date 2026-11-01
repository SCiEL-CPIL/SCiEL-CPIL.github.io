"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_u(i, j, c, val):
        U[i * m + j][c] = val
        
    def set_v(j, k, c, val):
        V[j * p + k][c] = val
        
    def set_w(k, i, c, val):
        W[k * n + i][c] = val

    # Strassen decomposition for 2x2x2 (columns 0-6)
    # M1 = A00 * (B01 - B11)
    set_u(0, 0, 0, 1)
    set_v(0, 1, 0, 1)
    set_v(1, 1, 0, -1)
    set_w(1, 0, 0, 1)
    
    # M2 = (A00 + A01) * B11
    set_u(0, 0, 1, 1)
    set_u(0, 1, 1, 1)
    set_v(1, 1, 1, 1)
    set_w(0, 0, 1, -1)
    set_w(1, 0, 1, 1)
    
    # M3 = (A10 + A11) * B00
    set_u(1, 0, 2, 1)
    set_u(1, 1, 2, 1)
    set_v(1, 0, 2, 1)
    set_w(0, 1, 2, 1)
    set_w(1, 1, 2, -1)
    
    # M4 = A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1)
    set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)
    
    # M5 = (A00 + A11) * (B00 + B11)
    set_u(0, 0, 4, 1)
    set_u(1, 1, 4, 1)
    set_v(0, 0, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, 1)
    set_w(1, 1, 4, 1)
    
    # M6 = (A01 - A11) * (B10 + B11)
    set_u(0, 1, 5, 1)
    set_u(1, 1, 5, -1)
    set_v(1, 0, 5, 1)
    set_v(1, 1, 5, 1)
    set_w(0, 0, 5, 1)
    
    # M7 = (A00 + A10) * (B00 - B10)
    set_u(0, 0, 6, 1)
    set_u(1, 0, 6, 1)
    set_v(0, 0, 6, 1)
    set_v(1, 0, 6, -1)
    set_w(1, 1, 6, 1)

    # Matrix-vector multiplication for the third column (columns 7-10)
    # C02 = A00*B02 + A01*B12
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)
    
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)
    
    # C12 = A10*B02 + A11*B12
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)
    
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
