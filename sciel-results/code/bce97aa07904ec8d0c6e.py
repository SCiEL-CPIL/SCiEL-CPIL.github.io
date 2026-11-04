import numpy as np

def construct():
    n, m, p = 2, 3, 6
    # Rank 33 decomposition
    # 12 terms for A_L B_L (2x1 * 1x6)
    # 21 terms for A_R B_R (2x2 * 2x6) using 3 Strassen (2x2*2x2)
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x1) * B_L (1x6)
    # A_L indices: i in {0,1}, j=0 -> idx i*3+0
    # B_L indices: i=0, k in {0..5} -> idx 0*6+k
    # C indices: k in {0..5}, i in {0,1} -> idx k*2+i
    for i in range(2):
        for k in range(6):
            U[i * 3 + 0][col] = 1
            V[0 * 6 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A_R (2x2) * B_R (2x6)
    # A_R indices: 
    # A11: i=0, j=1 -> 1
    # A12: i=0, j=2 -> 2
    # A21: i=1, j=1 -> 4
    # A22: i=1, j=2 -> 5
    
    # B_R split into 3 blocks of 2x2: cols (0,1), (2,3), (4,5)
    # Block b: k_start = 2*b
    # B11: i=1, k=2b -> 6+2b
    # B12: i=1, k=2b+1 -> 6+2b+1
    # B21: i=2, k=2b -> 12+2b
    # B22: i=2, k=2b+1 -> 12+2b+1
    
    # C indices for block b:
    # C11: i=0, k=2b -> 4b
    # C12: i=0, k=2b+1 -> 4b+1
    # C21: i=1, k=2b -> 4b+2
    # C22: i=1, k=2b+1 -> 4b+3
    
    # Strassen Terms M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define the 7 terms as list of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {a_idx: coeff}
    # V_coeffs: dict {b_idx: coeff}
    # W_coeffs: dict {c_idx: coeff}
    
    A11, A12, A21, A22 = 1, 2, 4, 5
    
    for b in range(3):
        k_start = 2 * b
        B11 = 6 + k_start
        B12 = 6 + k_start + 1
        B21 = 12 + k_start
        B22 = 12 + k_start + 1
        
        C11 = 4 * b
        C12 = 4 * b + 1
        C21 = 4 * b + 2
        C22 = 4 * b + 3
        
        # Term 1: M1
        U_t = {A11: 1, A22: 1}
        V_t = {B11: 1, B22: 1}
        W_t = {C11: 1, C22: 1}
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
        # Term 2: M2
        U_t = {A21: 1, A22: 1}
        V_t = {B11: 1}
        W_t = {C21: 1} # M2 contributes to C21 with +1
        # Wait, C22 = M1 - M2 + ... so M2 contributes -1 to C22.
        W_t = {C21: 1, C22: -1}
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
        # Term 3: M3
        U_t = {A11: 1}
        V_t = {B12: 1, B22: -1}
        W_t = {C12: 1, C22: 1} # C12=M3+M5, C22=M1-M2+M3+M6
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
        # Term 4: M4
        U_t = {A22: 1}
        V_t = {B21: 1, B11: -1}
        W_t = {C11: 1, C21: 1} # C11=M1+M4-M5+M7, C21=M2+M4
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
        # Term 5: M5
        U_t = {A11: 1, A12: 1}
        V_t = {B22: 1}
        W_t = {C11: -1, C12: 1} # C11=...-M5..., C12=M3+M5
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
        # Term 6: M6
        U_t = {A21: 1, A11: -1}
        V_t = {B11: 1, B12: 1}
        W_t = {C22: 1} # C22=...+M6
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
        # Term 7: M7
        U_t = {A12: 1, A22: -1}
        V_t = {B21: 1, B22: 1}
        W_t = {C11: 1} # C11=...+M7
        for idx, coeff in U_t.items(): U[idx][col] += coeff
        for idx, coeff in V_t.items(): V[idx][col] += coeff
        for idx, coeff in W_t.items(): W[idx][col] += coeff
        col += 1
        
    return {"U": U, "V": V, "W": W}
