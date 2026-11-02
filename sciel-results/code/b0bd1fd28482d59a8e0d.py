"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 15
    
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

    # Term 1
    set_u(0, 0, 0, 1); set_v(0, 0, 0, 1); set_w(2, 0, 0, 1)
    # Term 2
    set_u(1, 0, 1, 1); set_v(0, 1, 1, 1); set_w(0, 1, 1, 1)
    # Term 3
    set_u(0, 1, 2, 1); set_v(1, 0, 2, 1); set_w(2, 0, 2, 1)
    # Term 4
    set_u(1, 1, 3, 1); set_v(1, 1, 3, 1); set_w(0, 1, 3, 1)
    # Term 5
    set_u(0, 2, 4, 1); set_v(2, 0, 4, 1); set_w(2, 0, 4, 1)
    # Term 6
    set_u(1, 2, 5, 1); set_v(2, 1, 5, 1); set_w(0, 1, 5, 1)
    
    # Term 7
    set_u(0, 0, 6, 1); set_u(1, 0, 6, -1)
    set_v(0, 1, 6, 1); set_v(1, 0, 6, -1)
    set_w(2, 0, 6, 1); set_w(2, 1, 6, -1)
    
    # Term 8
    set_u(0, 0, 7, 1); set_u(1, 0, 7, -1)
    set_v(0, 1, 7, 1); set_v(1, 1, 7, -1)
    set_w(0, 0, 7, 1); set_w(0, 1, 7, -1)
    
    # Term 9
    set_u(0, 0, 8, 1); set_u(1, 0, 8, -1)
    set_v(0, 1, 8, 1); set_v(1, 2, 8, -1)
    set_w(1, 0, 8, 1); set_w(1, 1, 8, -1)
    
    # Term 10
    set_u(0, 1, 9, 1); set_u(1, 1, 9, -1)
    set_v(1, 1, 9, 1); set_v(2, 0, 9, -1)
    set_w(2, 0, 9, 1); set_w(2, 1, 9, -1)
    
    # Term 11
    set_u(0, 1, 10, 1); set_u(1, 1, 10, -1)
    set_v(1, 1, 10, 1); set_v(2, 1, 10, -1)
    set_w(0, 0, 10, 1); set_w(0, 1, 10, -1)
    
    # Term 12
    set_u(0, 1, 11, 1); set_u(1, 1, 11, -1)
    set_v(1, 1, 11, 1); set_v(2, 2, 11, -1)
    set_w(1, 0, 11, 1); set_w(1, 1, 11, -1)
    
    # Term 13
    set_u(0, 2, 12, 1); set_u(1, 2, 12, -1)
    set_v(2, 0, 12, 1); set_v(2, 1, 12, -1)
    set_w(2, 0, 12, 1); set_w(2, 1, 12, -1)
    
    # Term 14
    set_u(0, 2, 13, 1); set_u(1, 2, 13, -1)
    set_v(2, 0, 13, 1); set_v(2, 2, 13, -1)
    set_w(0, 0, 13, 1); set_w(0, 1, 13, -1)
    
    # Term 15
    set_u(0, 2, 14, 1); set_u(1, 2, 14, -1)
    set_v(2, 1, 14, 1); set_v(2, 2, 14, -1)
    set_w(1, 0, 14, 1); set_w(1, 1, 14, -1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
