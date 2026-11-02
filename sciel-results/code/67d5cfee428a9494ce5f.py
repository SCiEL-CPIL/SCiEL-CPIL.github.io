"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # Rows of U correspond to a_ij. Index = i*m + j
    # Terms:
    # 0: M1 = (a00+a11)(b00+b11) -> a: 1,0,0,1
    # 1: M2 = (a10+a11)b00       -> a: 0,0,1,1
    # 2: M3 = a00(b01-b11)       -> a: 1,0,0,0
    # 3: M4 = a11(b10-b00)       -> a: 0,0,0,1
    # 4: M5 = (a00+a01)b11       -> a: 1,1,0,0
    # 5: M6 = (a10-a00)(b00+b01) -> a: -1,0,1,0
    # 6: M7 = (a01-a11)(b10+b11) -> a: 0,1,0,-1
    
    # Indices for a_ij: i*m+j. m=2.
    # a00: 0, a01: 1, a10: 2, a11: 3
    
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # For each block b (0 to 3), we use 7 terms.
    # Block b corresponds to columns 2b and 2b+1 of B and C.
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for t in range(7):
            c = b * 7 + t
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    val = U_coeffs[t][i * 2 + j]
                    if val != 0:
                        U[i * m + j][c] = val
                        
            # Fill V
            # V rows: j*p + k. j in {0,1}, k in {k0, k1}
            # M1: b00+b11 -> V[0*p+k0]+=1, V[1*p+k1]+=1
            # M2: b00      -> V[0*p+k0]+=1
            # M3: b01-b11  -> V[0*p+k1]+=1, V[1*p+k1]+=-1
            # M4: b10-b00  -> V[1*p+k0]+=1, V[0*p+k0]+=-1
            # M5: b11      -> V[1*p+k1]+=1
            # M6: b00+b01  -> V[0*p+k0]+=1, V[0*p+k1]+=1
            # M7: b10+b11  -> V[1*p+k0]+=1, V[1*p+k1]+=1
            
            if t == 0:
                V[0 * p + k0][c] = 1
                V[1 * p + k1][c] = 1
            elif t == 1:
                V[0 * p + k0][c] = 1
            elif t == 2:
                V[0 * p + k1][c] = 1
                V[1 * p + k1][c] = -1
            elif t == 3:
                V[1 * p + k0][c] = 1
                V[0 * p + k0][c] = -1
            elif t == 4:
                V[1 * p + k1][c] = 1
            elif t == 5:
                V[0 * p + k0][c] = 1
                V[0 * p + k1][c] = 1
            elif t == 6:
                V[1 * p + k0][c] = 1
                V[1 * p + k1][c] = 1
                
            # Fill W
            # W rows: k*n + i. k in {k0, k1}, i in {0,1}
            # c00 (k0, 0): 2*k0 + 0
            # c01 (k1, 0): 2*k1 + 0
            # c10 (k0, 1): 2*k0 + 1
            # c11 (k1, 1): 2*k1 + 1
            
            # M1: c00+=1, c11+=1
            # M2: c10+=1, c11+=-1
            # M3: c01+=1, c11+=1
            # M4: c00+=1, c10+=1
            # M5: c00+=-1, c01+=1
            # M6: c11+=1
            # M7: c00+=1
            
            r_c00 = k0 * n + 0
            r_c01 = k1 * n + 0
            r_c10 = k0 * n + 1
            r_c11 = k1 * n + 1
            
            if t == 0:
                W[r_c00][c] = 1
                W[r_c11][c] = 1
            elif t == 1:
                W[r_c10][c] = 1
                W[r_c11][c] = -1
            elif t == 2:
                W[r_c01][c] = 1
                W[r_c11][c] = 1
            elif t == 3:
                W[r_c00][c] = 1
                W[r_c10][c] = 1
            elif t == 4:
                W[r_c00][c] = -1
                W[r_c01][c] = 1
            elif t == 5:
                W[r_c11][c] = 1
            elif t == 6:
                W[r_c00][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
