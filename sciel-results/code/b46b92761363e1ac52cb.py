"""Matrix multiplication 2x4 x 4x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(4):
        i0 = (block >> 1) & 1
        k0 = (block & 1) * 2
        
        # Strassen 7-term multiplication for 2x2 x 2x2
        # Local A indices: (i, j) for i in {0,1}, j in {0,1}
        # Local B indices: (j, k) for j in {0,1}, k in {0,1}
        # Local C indices: (k, i) for k in {0,1}, i in {0,1}
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[i0 * m + 0][col] += 1
        U[(i0 + 1) * m + 1][col] += 1
        V[0 * p + k0][col] += 1
        V[1 * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[i0 * m + 0][col] += 1
        U[(i0 + 1) * m + 1][col] += 1
        V[0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[i0 * m + 0][col] += 1
        V[0 * p + k0 + 1][col] += 1
        V[1 * p + k0 + 1][col] += -1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[(i0 + 1) * m + 1][col] += 1
        V[1 * p + k0][col] += 1
        V[0 * p + k0][col] += -1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[i0 * m + 0][col] += 1
        U[i0 * m + 1][col] += 1
        V[1 * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[i0 * m + 0][col] += -1
        U[(i0 + 1) * m + 0][col] += 1
        V[0 * p + k0][col] += 1
        V[0 * p + k0 + 1][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[i0 * m + 1][col] += 1
        U[(i0 + 1) * m + 1][col] += -1
        V[1 * p + k0][col] += 1
        V[1 * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
