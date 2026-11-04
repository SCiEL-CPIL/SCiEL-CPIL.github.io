"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189
    rank = 189
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 case
    # U2[I, J, c2], V2[J, K, c2], W2[K, I, c2]
    # We store them as dictionaries or lists for quick access
    # Indices I, J, K in 0..1
    # c2 in 0..6
    
    U2 = {}
    V2 = {}
    W2 = {}
    
    # Initialize with 0
    for I in range(2):
        for J in range(2):
            for c2 in range(7):
                U2[(I, J, c2)] = 0
    for J in range(2):
        for K in range(2):
            for c2 in range(7):
                V2[(J, K, c2)] = 0
    for K in range(2):
        for I in range(2):
            for c2 in range(7):
                W2[(K, I, c2)] = 0
                
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2[(0, 0, 0)] = 1
    U2[(1, 1, 0)] = 1
    V2[(0, 0, 0)] = 1
    V2[(1, 1, 0)] = 1
    W2[(0, 0, 0)] = 1 # C00
    W2[(1, 1, 0)] = 1 # C11
    
    # M2: (A10+A11)B00 -> C10
    U2[(1, 0, 1)] = 1
    U2[(1, 1, 1)] = 1
    V2[(0, 0, 1)] = 1
    W2[(0, 1, 1)] = 1 # C10
    
    # M3: A00(B01-B11) -> C01
    U2[(0, 0, 2)] = 1
    V2[(0, 1, 2)] = 1
    V2[(1, 1, 2)] = -1
    W2[(1, 0, 2)] = 1 # C01
    
    # M4: A11(B10-B00) -> C00, C10
    U2[(1, 1, 3)] = 1
    V2[(1, 0, 3)] = 1
    V2[(0, 0, 3)] = -1
    W2[(0, 0, 3)] = 1 # C00
    W2[(0, 1, 3)] = 1 # C10
    
    # M5: (A00+A01)B11 -> C00(-), C01(+)
    U2[(0, 0, 4)] = 1
    U2[(0, 1, 4)] = 1
    V2[(1, 1, 4)] = 1
    W2[(0, 0, 4)] = -1 # C00
    W2[(1, 0, 4)] = 1 # C01
    
    # M6: (A10-A00)(B00+B10) -> C11
    U2[(1, 0, 5)] = 1
    U2[(0, 0, 5)] = -1
    V2[(0, 0, 5)] = 1
    V2[(1, 0, 5)] = 1
    W2[(1, 1, 5)] = 1 # C11
    
    # M7: (A01-A11)(B01+B11) -> C00
    U2[(0, 1, 6)] = 1
    U2[(1, 1, 6)] = -1
    V2[(0, 1, 6)] = 1
    V2[(1, 1, 6)] = 1
    W2[(0, 0, 6)] = 1 # C00
    
    # Define Schoolbook coefficients for 3x3 case
    # U3[i2, j2, c3] = 1 if c3 == i2*3 + j2
    # V3[j2, k2, c3] = 1 if c3 == j2*3 + k2
    # W3[k2, i2, c3] = 1 if c3 == k2*3 + i2
    
    # We can compute these on the fly to save space/time in setup, 
    # but explicit arrays are faster in the loop.
    U3 = {}
    V3 = {}
    W3 = {}
    for i2 in range(3):
        for j2 in range(3):
            c3 = i2 * 3 + j2
            U3[(i2, j2, c3)] = 1
    for j2 in range(3):
        for k2 in range(3):
            c3 = j2 * 3 + k2
            V3[(j2, k2, c3)] = 1
    for k2 in range(3):
        for i2 in range(3):
            c3 = k2 * 3 + i2
            W3[(k2, i2, c3)] = 1
    
    # Fill U, V, W
    for c in range(rank):
        c2 = c // 27
        c3 = c % 27
        
        for I in range(2):
            for J in range(2):
                u2_val = U2.get((I, J, c2), 0)
                if u2_val == 0:
                    continue
                for i2 in range(3):
                    for j2 in range(3):
                        u3_val = U3.get((i2, j2, c3), 0)
                        if u3_val == 0:
                            continue
                        val = u2_val * u3_val
                        i = 3 * I + i2
                        j = 3 * J + j2
                        U[i * 6 + j][c] += val
                        
        for J in range(2):
            for K in range(2):
                v2_val = V2.get((J, K, c2), 0)
                if v2_val == 0:
                    continue
                for j2 in range(3):
                    for k2 in range(3):
                        v3_val = V3.get((j2, k2, c3), 0)
                        if v3_val == 0:
                            continue
                        val = v2_val * v3_val
                        j = 3 * J + j2
                        k = 3 * K + k2
                        V[j * 6 + k][c] += val
                        
        for K in range(2):
            for I in range(2):
                w2_val = W2.get((K, I, c2), 0)
                if w2_val == 0:
                    continue
                for k2 in range(3):
                    for i2 in range(3):
                        w3_val = W3.get((k2, i2, c3), 0)
                        if w3_val == 0:
                            continue
                        val = w2_val * w3_val
                        k = 3 * K + k2
                        i = 3 * I + i2
                        W[k * 6 + i][c] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
