"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_U(i, j, r, val):
        U[i * m + j][r] += val
        
    def set_V(j, k, r, val):
        V[j * p + k][r] += val
        
    def set_W(k, i, r, val):
        W[k * n + i][r] += val

    # 1. Strassen 2x2x2 (Top-Left)
    # M1
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M2
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(1, 0, 1, 1) # C10
    
    # M3
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(0, 1, 2, 1) # C01
    
    # M4
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(1, 0, 3, 1) # C00, C10
    
    # M5
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(0, 1, 4, 1) # -C00, C01
    
    # M6
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1) # C11
    
    # M7
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1); set_W(1, 1, 6, -1) # C00, -C11

    # 2. Mat-Vec A[0:2,0:2] * B[0:2,2]
    # M8: A00 * B02 -> C02
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # M9: A01 * B12 -> C02
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # M10: A10 * B02 -> C12
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # M11: A11 * B12 -> C12
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)

    # 3. Outer Product A[0:2,2] * B[2,0:2]
    # M12: A02 * B20 -> C00
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    # M13: A02 * B21 -> C01
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(0, 1, 12, 1)
    
    # M14: A12 * B20 -> C10
    set_U(1, 2, 13, 1)
    set_V(2, 0, 13, 1)
    set_W(1, 0, 13, 1)
    
    # M15: A12 * B21 -> C11
    set_U(1, 2, 14, 1)
    set_V(2, 1, 14, 1)
    set_W(1, 1, 14, 1)

    # 4. Vec-Scalar A[0:2,2] * B[2,2]
    # M16: A02 * B22 -> C02
    set_U(0, 2, 15, 1)
    set_V(2, 2, 15, 1)
    set_W(2, 0, 15, 1)
    
    # M17: A12 * B22 -> C12
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
