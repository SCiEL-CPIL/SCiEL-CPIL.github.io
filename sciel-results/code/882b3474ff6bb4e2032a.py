"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set entries
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def set_u(i, j, col, val):
        U[i * m + j][col] = val
        
    def set_v(j, k, col, val):
        V[j * p + k][col] = val
        
    def set_w(k, i, col, val):
        W[k * n + i][col] = val

    # Term 1: (a00 + a11) * (b00 + b11)
    # C00 += 1, C11 += 1
    col = 0
    set_u(0, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)
    set_w(1, 1, col, 1)

    # Term 2: (a10 + a11) * b00
    # C10 += 1, C11 += -1
    col = 1
    set_u(1, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_w(0, 1, col, 1)
    set_w(1, 1, col, -1)

    # Term 3: a00 * (b01 - b11)
    # C01 += 1, C11 += 1
    col = 2
    set_u(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_v(1, 1, col, -1)
    set_w(1, 0, col, 1)
    set_w(1, 1, col, 1)

    # Term 4: a11 * (b10 - b00)
    # C00 += 1, C10 += 1
    col = 3
    set_u(1, 1, col, 1)
    set_v(1, 0, col, 1)
    set_v(0, 0, col, -1)
    set_w(0, 0, col, 1)
    set_w(0, 1, col, 1)

    # Term 5: (a00 + a01) * b11
    # C00 += -1, C01 += 1
    col = 4
    set_u(0, 0, col, 1)
    set_u(0, 1, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, -1)
    set_w(1, 0, col, 1)

    # Term 6: (a10 - a00) * (b00 + b01)
    # C11 += 1
    col = 5
    set_u(1, 0, col, 1)
    set_u(0, 0, col, -1)
    set_v(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_w(1, 1, col, 1)

    # Term 7: (a01 - a11) * (b10 + b11)
    # C00 += 1
    col = 6
    set_u(0, 1, col, 1)
    set_u(1, 1, col, -1)
    set_v(1, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)

    # Term 8: a00 * b02
    # C02 += 1
    col = 7
    set_u(0, 0, col, 1)
    set_v(0, 2, col, 1)
    set_w(2, 0, col, 1)

    # Term 9: a01 * b12
    # C02 += 1
    col = 8
    set_u(0, 1, col, 1)
    set_v(1, 2, col, 1)
    set_w(2, 0, col, 1)

    # Term 10: a10 * b02
    # C12 += 1
    col = 9
    set_u(1, 0, col, 1)
    set_v(0, 2, col, 1)
    set_w(2, 1, col, 1)

    # Term 11: a11 * b12
    # C12 += 1
    col = 10
    set_u(1, 1, col, 1)
    set_v(1, 2, col, 1)
    set_w(2, 1, col, 1)

    # Term 12: a02 * b20
    # C00 += 1
    col = 11
    set_u(0, 2, col, 1)
    set_v(2, 0, col, 1)
    set_w(0, 0, col, 1)

    # Term 13: a02 * b21
    # C01 += 1
    col = 12
    set_u(0, 2, col, 1)
    set_v(2, 1, col, 1)
    set_w(1, 0, col, 1)

    # Term 14: a02 * b22
    # C02 += 1
    col = 13
    set_u(0, 2, col, 1)
    set_v(2, 2, col, 1)
    set_w(2, 0, col, 1)

    # Term 15: a12 * b20
    # C10 += 1
    col = 14
    set_u(1, 2, col, 1)
    set_v(2, 0, col, 1)
    set_w(0, 1, col, 1)

    # Term 16: a12 * b21
    # C11 += 1
    col = 15
    set_u(1, 2, col, 1)
    set_v(2, 1, col, 1)
    set_w(1, 1, col, 1)

    # Term 17: a12 * b22
    # C12 += 1
    col = 16
    set_u(1, 2, col, 1)
    set_v(2, 2, col, 1)
    set_w(2, 1, col, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
