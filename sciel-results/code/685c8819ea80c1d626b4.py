import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split p=6 into p1=4 and p2=2.
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x2 using Schoolbook (Rank 32)
    # Total Rank: 81
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4 ---
    # Strassen 2x2 factors
    # U_2, V_2, W_2 for 2x2x2
    # Indices: i,j in 0..1 for U, k in 0..1 for W, j,k in 0..1 for V
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # We build 4x4x4 by treating 4x4 as 2x2 blocks of 2x2.
    # This is a Kronecker product of the 2x2 Strassen decomposition with itself.
    
    # First, define the 2x2 Strassen factors explicitly.
    # r from 0 to 6
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # P1 = (A00+A11)(B00+B11)
    U_2[0][0] = 1; U_2[3][0] = 1
    V_2[0][0] = 1; V_2[3][0] = 1
    W_2[0][0] = 1; W_2[3][0] = 1 # C00 += P1, C11 += P1
    
    # P2 = (A10+A11)B00
    U_2[2][1] = 1; U_2[3][1] = 1
    V_2[0][1] = 1
    W_2[2][1] = 1; W_2[3][1] = 1 # C10 += P2, C11 -= P2
    
    # P3 = A00(B10-B11)
    U_2[0][2] = 1
    V_2[2][2] = 1; V_2[3][2] = -1
    W_2[1][2] = 1; W_2[3][2] = 1 # C01 += P3, C11 += P3
    
    # P4 = A11(B01-B00)
    U_2[3][3] = 1
    V_2[1][3] = 1; V_2[0][3] = -1
    W_2[0][3] = 1; W_2[2][3] = 1 # C00 += P4, C10 += P4
    
    # P5 = (A00+A01)B11
    U_2[0][4] = 1; U_2[1][4] = 1
    V_2[3][4] = 1
    W_2[0][4] = -1; W_2[1][4] = 1 # C00 -= P5, C01 += P5
    
    # P6 = (A10-A00)(B00+B01)
    U_2[2][5] = 1; U_2[0][5] = -1
    V_2[0][5] = 1; V_2[1][5] = 1
    W_2[3][5] = 1 # C11 += P6
    
    # P7 = (A01-A11)(B10+B11)
    U_2[1][6] = 1; U_2[3][6] = -1
    V_2[2][6] = 1; V_2[3][6] = 1
    W_2[0][6] = 1; W_2[1][6] = 1 # C00 += P7, C01 += P7 (Wait, C01 = P3+P5, C00 = P1+P4-P5+P7)
    # Check C01: P3+P5. W2[1] gets P3 and P5. Correct.
    # Check C00: P1+P4-P5+P7. W2[0] gets P1, P4, -P5, P7. Correct.
    
    # Now, construct 4x4x4 factors by Kronecker product.
    # U_4[i*4+j, r1*7+r2] = U_2[i*2+i2, r1] * U_2[j*2+j2, r2] ? No.
    # The structure is:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The "outer" Strassen combines Aij.
    # The "inner" Strassen multiplies the resulting 2x2 blocks.
    
    # Let's use the property that T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}.
    # Rank is 7*7=49.
    
    col_offset = 0
    for r_outer in range(7):
        for r_inner in range(7):
            r = col_offset + r_outer * 7 + r_inner
            # U_4 index: i*4+j. i,j in 0..3.
            # i = i_outer*2 + i_inner, j = j_outer*2 + j_inner.
            # U_4[i,j,r] = U_2[i_outer, j_outer, r_outer] * U_2[i_inner, j_inner, r_inner]
            for i_outer in range(2):
                for j_outer in range(2):
                    u_outer = U_2[i_outer * 2 + j_outer][r_outer]
                    if u_outer == 0: continue
                    for i_inner in range(2):
                        for j_inner in range(2):
                            u_inner = U_2[i_inner * 2 + j_inner][r_inner]
                            if u_inner == 0: continue
                            u_val = u_outer * u_inner
                            i = i_outer * 2 + i_inner
                            j = j_outer * 2 + j_inner
                            U[i * 4 + j][r] = u_val
                            
            # V_4 index: j*4+k.
            for j_outer in range(2):
                for k_outer in range(2):
                    v_outer = V_2[j_outer * 2 + k_outer][r_outer]
                    if v_outer == 0: continue
                    for j_inner in range(2):
                        for k_inner in range(2):
                            v_inner = V_2[j_inner * 2 + k_inner][r_inner]
                            if v_inner == 0: continue
                            v_val = v_outer * v_inner
                            j = j_outer * 2 + j_inner
                            k = k_outer * 2 + k_inner
                            V[j * 4 + k][r] = v_val
                            
            # W_4 index: k*4+i.
            for k_outer in range(2):
                for i_outer in range(2):
                    w_outer = W_2[k_outer * 2 + i_outer][r_outer]
                    if w_outer == 0: continue
                    for k_inner in range(2):
                        for i_inner in range(2):
                            w_inner = W_2[k_inner * 2 + i_inner][r_inner]
                            if w_inner == 0: continue
                            w_val = w_outer * w_inner
                            k = k_outer * 2 + k_inner
                            i = i_outer * 2 + i_inner
                            W[k * 4 + i][r] = w_val
                            
            col_offset += 1
            
    # --- Part 2: Schoolbook 4x4x2 ---
    # This part handles the last 2 columns of B (indices 4, 5).
    # A is 4x4. B_2 is 4x2.
    # C_2 is 4x2.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # i in 0..3, j in 0..3, k in 0..1.
    # Total 32 terms.
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(2):
                r = col_offset
                U[i * 4 + j][r] = 1
                # V global index: j * 6 + (4 + k)
                V[j * 6 + 4 + k][r] = 1
                # W global index: (4 + k) * 4 + i
                W[(4 + k) * 4 + i][r] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
