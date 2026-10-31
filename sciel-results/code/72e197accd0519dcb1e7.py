import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen Decomposition
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # strassen_A[s][i] coefficient of A[i] in s-th product
    # strassen_B[s][j] coefficient of B[j] in s-th product
    # strassen_C[i][s] coefficient of s-th product in C[i]
    
    sA = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [1, 0, -1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    sB = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    sC = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # 4x4 Strassen Decomposition
    # Blocks: 0:00, 1:01, 2:10, 3:11
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # outer_A[m][b] coefficient of block b in m-th product's A-side
    # outer_B[m][b] coefficient of block b in m-th product's B-side
    # outer_C[b][m] coefficient of m-th product in block b of C
    
    oA = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [1, 0, -1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    oB = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    oC = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for m in range(7):
        for s in range(7):
            # Compute linear combination for A
            # A_side is a 2x2 matrix of coefficients for A's blocks
            # But we need coefficients for A's scalar entries
            # A_scalar_coeffs[i,j]
            
            A_coeffs = [0] * (n * m)
            B_coeffs = [0] * (m * p)
            
            # For each entry u,v in the 2x2 block
            for u in range(2):
                for v in range(2):
                    # Coefficient of A'_m[u,v] in the s-th inner product
                    coeff_inner = sA[s][u * 2 + v]
                    if coeff_inner == 0:
                        continue
                    
                    # A'_m[u,v] is a linear combination of A's blocks
                    for b in range(4):
                        coeff_outer = oA[m][b]
                        if coeff_outer == 0:
                            continue
                        
                        total_coeff = coeff_inner * coeff_outer
                        if total_coeff == 0:
                            continue
                        
                        # Block b corresponds to rows 2*(b//2), 2*(b//2)+1
                        # and cols 2*(b%2), 2*(b%2)+1
                        br = b // 2
                        bc = b % 2
                        
                        # Entries in block b
                        # (2*br+u, 2*bc+v)
                        i = 2 * br + u
                        j = 2 * bc + v
                        
                        A_coeffs[i * m + j] += total_coeff
            
            # Similarly for B
            for u in range(2):
                for v in range(2):
                    coeff_inner = sB[s][u * 2 + v]
                    if coeff_inner == 0:
                        continue
                    
                    for b in range(4):
                        coeff_outer = oB[m][b]
                        if coeff_outer == 0:
                            continue
                        
                        total_coeff = coeff_inner * coeff_outer
                        if total_coeff == 0:
                            continue
                        
                        br = b // 2
                        bc = b % 2
                        
                        i = 2 * br + u
                        j = 2 * bc + v
                        
                        B_coeffs[i * p + j] += total_coeff
            
            # Fill U and V
            for i in range(n * m):
                if A_coeffs[i] != 0:
                    U[i][col] = A_coeffs[i]
            
            for i in range(m * p):
                if B_coeffs[i] != 0:
                    V[i][col] = B_coeffs[i]
            
            # Compute W coefficients
            # For each output entry i,k in C
            for i in range(n):
                for k in range(p):
                    # Block index
                    bi = i // 2
                    bk = k // 2
                    # Local index in block
                    ui = i % 2
                    uk = k % 2
                    
                    coeff_outer = oC[bi * 2 + bk][m]
                    if coeff_outer == 0:
                        continue
                    
                    coeff_inner = sC[ui * 2 + uk][s]
                    
                    total_coeff = coeff_outer * coeff_inner
                    if total_coeff != 0:
                        W[k * n + i][col] = total_coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
