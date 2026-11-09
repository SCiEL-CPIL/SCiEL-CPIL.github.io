"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 55.
    # Part 1: j=2 (M_{2,1,10}) -> Rank 20
    # Part 2: j=0,1 (M_{2,2,10}) -> Rank 35 (5 Strassen blocks)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)] # 6 x 55
    V = [[0] * rank for _ in range(m * p)] # 30 x 55
    W = [[0] * rank for _ in range(p * n)] # 20 x 55
    
    col = 0
    
    # Part 1: j=2
    # C_{i,k} += A_{i,2} * B_{2,k}
    for i in range(n):
        for k in range(p):
            # U index: i*m + j = i*3 + 2
            # V index: j*p + k = 2*10 + k
            # W index: k*n + i
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j=0,1 using Strassen on 5 column pairs
    # Strassen coefficients for M_{2,2,2}
    # U (A) coefficients: 2x2
    # V (B) coefficients: 2x2
    # W (C) coefficients: 2x2
    
    # Strassen M1..M7
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_U = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    strassen_V = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01 - B11
        [[-1, 0], [1, 0]],# M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 1], [1, 1]]  # M7: B10 + B11
    ]
    
    strassen_W = [
        [[1, 0], [0, -1]], # C00: M1 - M5 + M7 ? No.
        # Let's re-verify Strassen output formula.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        [[1, 0], [0, 0]], # Placeholder, will fill correctly
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]]
    ]
    
    # Correct W coefficients
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    strassen_W[0] = [[1, 0], [0, 0]] # M1 contributes to C00, C11
    # It's easier to define the contribution of each M to each C entry.
    
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 -= 1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    # Let's iterate 5 times for column pairs
    for pair_idx in range(5):
        k_a = pair_idx * 2
        k_b = pair_idx * 2 + 1
        
        for s in range(7):
            # U coefficients for A (j=0,1)
            for i in range(2):
                for j in range(2):
                    u_val = strassen_U[s][i][j]
                    if u_val != 0:
                        U[i * m + j][col] += u_val
                        
            # V coefficients for B (j=0,1, k=k_a, k_b)
            for j in range(2):
                for dk in range(2):
                    k_val = k_a + dk
                    v_val = strassen_V[s][j][dk]
                    if v_val != 0:
                        V[j * p + k_val][col] += v_val
                        
            # W coefficients for C (i=0,1, k=k_a, k_b)
            # We need to add the contributions to W
            # M1: C00(1), C11(1)
            # M2: C10(1), C11(-1)
            # M3: C01(1), C11(1)
            # M4: C00(1), C10(1)
            # M5: C00(-1), C01(1)
            # M6: C11(1)
            # M7: C00(1)
            
            # i is row of C, k is col of C
            # W index: k*n + i
            
            # Define W contributions for each s
            if s == 0: # M1
                W[k_a * n + 0][col] += 1
                W[k_b * n + 1][col] += 1
            elif s == 1: # M2
                W[k_a * n + 1][col] += 1
                W[k_b * n + 1][col] += -1
            elif s == 2: # M3
                W[k_b * n + 0][col] += 1
                W[k_b * n + 1][col] += 1
            elif s == 3: # M4
                W[k_a * n + 0][col] += 1
                W[k_a * n + 1][col] += 1
            elif s == 4: # M5
                W[k_a * n + 0][col] += -1
                W[k_b * n + 0][col] += 1
            elif s == 5: # M6
                W[k_b * n + 1][col] += 1
            elif s == 6: # M7
                W[k_a * n + 0][col] += 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
