import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # M = A * B
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # M11 = (a11+a21)(b11+b21) - (a21-a11)(b11+b21) ... no, standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The indices in U, V, W correspond to flattened matrices.
    # A is 2x4. B is 4x4. C is 2x4.
    # We split into 4 blocks of 2x2.
    # Block 0: A[:, 0:2], B[0:2, 0:2] -> C[:, 0:2] part 1?
    # Actually C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # This is not a simple block multiplication of 2x2 by 2x2.
    # C is 2x4.
    # C[:, 0:2] depends on A[:, 0:4] and B[0:4, 0:2].
    
    # The previous program used a specific decomposition.
    # It treated the 4x4 B as 4 blocks of 2x2.
    # And A as 2 blocks of 2x2?
    # A is 2x4.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # C = [A_L B11 + A_R B21, A_L B12 + A_R B22].
    
    # This requires 4 matrix multiplications of 2x2 matrices.
    # 1. A_L * B11
    # 2. A_R * B21
    # 3. A_L * B12
    # 4. A_R * B22
    
    # Each 2x2 multiplication uses 7 scalar multiplications (Strassen).
    # Total 28.
    
    # We define the Strassen coefficients for a single 2x2 multiplication.
    # Let A be 2x2, B be 2x2, C be 2x2.
    # U_A, U_B, U_C are vectors of length 4 (flattened).
    # We need 7 terms.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # A coeffs: [1, 0, 0, 1] (indices 0,1,2,3 for 11,12,21,22)
    # B coeffs: [1, 0, 0, 1]
    # C coeffs: [1, 0, 0, 1] (contributes to C11, C12, C21, C22? No, M1 contributes to C11, C22)
    # Wait, C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each term k, we have:
    # u_k: vector in R^4
    # v_k: vector in R^4
    # w_k: vector in R^4
    
    # Term 1:
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: [1, 0, 0, 1] (M1 adds to C11 and C22? No, M1 is a scalar.
    # C11 += M1, C22 += M1.
    # So w: [1, 0, 0, 1]
    
    # Term 2: M2 = (A21+A22)B11
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: [0, 0, 1, 0] (C21 += M2)
    
    # Term 3: M3 = A11(B12-B22)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: [0, 1, 0, 1] (C12 += M3, C22 += M3)
    
    # Term 4: M4 = A22(B21-B11)
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1]
    # w: [1, 0, 1, 0] (C11 += M4, C21 += M4)
    
    # Term 5: M5 = (A11+A12)B22
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # w: [-1, 1, 0, 0] (C11 -= M5, C12 += M5)
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # w: [0, 0, 0, 1] (C22 += M6)
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # w: [1, 0, 0, 0] (C11 += M7)
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (2, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        {'U': [(0, -1), (2, 1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    # We have 4 blocks of multiplication.
    # Block 0: A_L * B11. A_L is A[:, 0:2]. B11 is B[0:2, 0:2]. Result contributes to C[:, 0:2].
    # Block 1: A_L * B12. A_L is A[:, 0:2]. B12 is B[0:2, 2:4]. Result contributes to C[:, 2:4].
    # Block 2: A_R * B21. A_R is A[:, 2:4]. B21 is B[2:4, 0:2]. Result contributes to C[:, 0:2].
    # Block 3: A_R * B22. A_R is A[:, 2:4]. B22 is B[2:4, 2:4]. Result contributes to C[:, 2:4].
    
    # U index: i * m + j. i in 0..1, j in 0..3.
    # V index: j * p + k. j in 0..3, k in 0..3.
    # W index: k * n + i. k in 0..3, i in 0..1.
    
    # For a block with A offset jA (0 or 2) and B offset jB (0 or 2) and k offset kB (0 or 2):
    # A indices: i in 0..1, j in jA..jA+1.
    # B indices: j in jB..jB+1, k in kB..kB+1.
    # C indices: k in kB..kB+1, i in 0..1.
    
    col = 0
    for jA in [0, 2]:
        for jB in [0, 2]:
            for kB in [0, 2]:
                # This combination is not a simple block multiplication.
                # C[:, kB:kB+2] = A[:, 0:2] B[0:2, kB:kB+2] + A[:, 2:4] B[2:4, kB:kB+2]
                # So we have 4 terms:
                # 1. A_L * B_top_left (if kB=0) or B_top_right (if kB=2)
                # 2. A_R * B_bottom_left (if kB=0) or B_bottom_right (if kB=2)
                
                # Let's iterate over the 4 block multiplications explicitly.
                pass
    
    # Correct mapping:
    # Mult 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Mult 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # Mult 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # Mult 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    blocks = [
        {'jA': 0, 'jB': 0, 'kB': 0}, # A_L * B11 -> C[:, 0:2]
        {'jA': 0, 'jB': 0, 'kB': 2}, # A_L * B12 -> C[:, 2:4]
        {'jA': 2, 'jB': 2, 'kB': 0}, # A_R * B21 -> C[:, 0:2]
        {'jA': 2, 'jB': 2, 'kB': 2}  # A_R * B22 -> C[:, 2:4]
    ]
    
    for block in blocks:
        jA = block['jA']
        jB = block['jB']
        kB = block['kB']
        
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = jA + j
                U[global_i * m + global_j][col] += coeff
            
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = jB + j
                global_k = kB + k
                V[global_j * p + global_k][col] += coeff
            
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = kB + k
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
