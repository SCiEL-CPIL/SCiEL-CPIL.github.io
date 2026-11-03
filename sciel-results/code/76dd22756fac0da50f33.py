"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2 m-blocks * 4 p-blocks * (7 Strassen + 4 Schoolbook)
    rank = 2 * 4 * (7 + 4)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # m-blocks: 0 (j=0,1), 1 (j=2,3)
    for bm in range(2):
        j0 = bm * 2
        j1 = bm * 2 + 1
        
        # p-blocks: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5), 3 (k=6,7)
        for bp in range(4):
            k0 = bp * 2
            k1 = bp * 2 + 1
            
            # --- Strassen Part (i=0,1) ---
            # M1 = (A00 + A11) * (B00 + B11)
            U[0 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            # C00 += M1
            W[k0 * n + 0][col] += 1
            # C11 += M1
            W[k1 * n + 1][col] += 1
            col += 1
            
            # M2 = (A10 + A11) * B00
            U[1 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            # C10 += M2
            W[k0 * n + 1][col] += 1
            # C11 -= M2
            W[k1 * n + 1][col] -= 1
            col += 1
            
            # M3 = A00 * (B10 - B11)
            U[0 * m + j0][col] += 1
            V[j1 * p + k0][col] += 1
            V[j1 * p + k1][col] -= 1
            # C01 += M3
            W[k1 * n + 0][col] += 1
            # C11 += M3
            W[k1 * n + 1][col] += 1
            col += 1
            
            # M4 = A11 * (B01 - B00)
            U[1 * m + j1][col] += 1
            V[j0 * p + k1][col] += 1
            V[j0 * p + k0][col] -= 1
            # C00 += M4
            W[k0 * n + 0][col] += 1
            # C10 += M4
            W[k0 * n + 1][col] += 1
            col += 1
            
            # M5 = (A00 + A01) * B11
            U[0 * m + j0][col] += 1
            U[0 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            # C00 -= M5
            W[k0 * n + 0][col] -= 1
            # C01 += M5
            W[k1 * n + 0][col] += 1
            col += 1
            
            # M6 = (A01 - A10) * (B00 + B01)
            U[0 * m + j1][col] += 1
            U[1 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k1][col] += 1
            # C11 += M6
            W[k1 * n + 1][col] += 1
            col += 1
            
            # M7 = (A10 - A11) * (B10 + B11)
            U[1 * m + j0][col] += 1
            U[1 * m + j1][col] -= 1
            V[j1 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            # C00 += M7
            W[k0 * n + 0][col] += 1
            col += 1
            
            # --- Schoolbook Part (i=2) ---
            # S1 = A20 * B00 -> C20
            U[2 * m + j0][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 2][col] += 1
            col += 1
            
            # S2 = A21 * B10 -> C20
            U[2 * m + j1][col] += 1
            V[j1 * p + k0][col] += 1
            W[k0 * n + 2][col] += 1
            col += 1
            
            # S3 = A20 * B01 -> C21
            U[2 * m + j0][col] += 1
            V[j0 * p + k1][col] += 1
            W[k1 * n + 2][col] += 1
            col += 1
            
            # S4 = A21 * B11 -> C21
            U[2 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k1 * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
