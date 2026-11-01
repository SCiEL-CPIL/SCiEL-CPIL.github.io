import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to set values
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Strassen decomposition for the 2x2 part (k=0,1)
    # M1 = (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1) # C00
    set_w(1, 1, 0, 1) # C11

    # M2 = (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1) # C10
    set_w(1, 1, 1, -1) # C11

    # M3 = A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1) # C01
    set_w(1, 1, 2, 1) # C11

    # M4 = A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1) # C00
    set_w(0, 1, 3, 1) # C10

    # M5 = (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1) # C00
    set_w(1, 0, 4, 1) # C01

    # M6 = (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1) # C11

    # M7 = (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1) # C00

    # Factor 7 for the 3rd column (k=2)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # U7 = [1, 1, 1, 1]
    set_u(0, 0, 7, 1); set_u(0, 1, 7, 1)
    set_u(1, 0, 7, 1); set_u(1, 1, 7, 1)
    
    # V7 has B02=1, B12=1
    set_v(0, 2, 7, 1); set_v(1, 2, 7, 1)
    
    # W7 has C02=1, C12=1
    set_w(2, 0, 7, 1)
    set_w(2, 1, 7, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
