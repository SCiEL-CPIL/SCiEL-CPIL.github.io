"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm coefficients for 2x2x2
    # M1: (a00+a11)(b00+b11) -> C11 += 1, C22 += 1
    # M2: (a10+a11)b00 -> C21 += 1, C22 -= 1
    # M3: a00(b01-b11) -> C12 += 1, C22 += 1
    # M4: a11(b10-b00) -> C11 += 1, C21 += 1
    # M5: (a00+a01)b11 -> C11 -= 1, C12 += 1
    # M6: (a10-a00)(b00+b01) -> C22 += 1
    # M7: (a01-a11)(b10+b11) -> C11 += 1
    
    # Note: b indices in Strassen are b00, b01, b10, b11
    # In our block b, b00 is b_{0, k1}, b01 is b_{0, k2}, b10 is b_{1, k1}, b11 is b_{1, k2}
    # V index for b_{j, k} is j*p + k.
    # W index for c_{i, k} is k*n + i.
    # U index for a_{i, j} is i*m + j.

    for block in range(5):
        k1 = 2 * block
        k2 = 2 * block + 1
        
        # M1
        r = 7 * block + 0
        U[0][r] += 1
        U[3][r] += 1
        V[k1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] += 1
        W[2 * k2 + 1][r] += 1
        
        # M2
        r = 7 * block + 1
        U[2][r] += 1
        U[3][r] += 1
        V[k1][r] += 1
        W[2 * k1 + 1][r] += 1
        W[2 * k2 + 1][r] -= 1
        
        # M3
        r = 7 * block + 2
        U[0][r] += 1
        V[k2][r] += 1
        V[10 + k2][r] -= 1
        W[2 * k2][r] += 1
        W[2 * k2 + 1][r] += 1
        
        # M4
        r = 7 * block + 3
        U[3][r] += 1
        V[10 + k1][r] += 1
        V[k1][r] -= 1
        W[2 * k1][r] += 1
        W[2 * k1 + 1][r] += 1
        
        # M5
        r = 7 * block + 4
        U[0][r] += 1
        U[1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] -= 1
        W[2 * k2][r] += 1
        
        # M6
        r = 7 * block + 5
        U[2][r] += 1
        U[0][r] -= 1
        V[k1][r] += 1
        V[k2][r] += 1
        W[2 * k2 + 1][r] += 1
        
        # M7
        r = 7 * block + 6
        U[1][r] += 1
        U[3][r] -= 1
        V[10 + k1][r] += 1
        V[10 + k2][r] += 1
        W[2 * k1][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
