"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We construct a decomposition of rank 34.
    # Strategy:
    # 1. Decompose M_{2,3,2} with rank 11.
    # 2. Decompose M_{2,3,1} with rank 6.
    # 3. M_{2,3,3} = M_{2,3,2} + M_{2,3,1} (block diagonal in the k-index)
    #    Rank <= 11 + 6 = 17.
    # 4. M_{2,3,6} = M_{2,3,3} (for k=0,1,2) + M_{2,3,3} (for k=3,4,5)
    #    Rank <= 17 + 17 = 34.
    
    rank_total = 34
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 34
    V = [[0] * rank_total for _ in range(m * p)]  # 18 x 34
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 34
    
    def add_term(u_vec, v_vec, w_vec, col_idx):
        """
        u_vec: length n*m (6)
        v_vec: length m*p (18)
        w_vec: length p*n (12)
        """
        for i in range(n * m):
            U[i][col_idx] = u_vec[i]
        for i in range(m * p):
            V[i][col_idx] = v_vec[i]
        for i in range(p * n):
            W[i][col_idx] = w_vec[i]

    # --- Construction of M_{2,3,2} (rank 11) ---
    # A is 2x3, B is 3x2.
    # Split A into A_2 (2x2) and a_3 (2x1).
    # Split B into B_2 (2x2) and b_3 (1x2).
    # C = A_2 B_2 + a_3 b_3^T.
    # A_2 B_2 uses Strassen (rank 7) on indices i in {0,1}, j in {0,1}, k in {0,1}.
    # a_3 b_3^T uses 4 terms on j=2.
    
    # Strassen for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for Strassen part (j=0,1; k=0,1):
    # U indices: i*m+j. i in 0,1. j in 0,1.
    # V indices: j*p+k. j in 0,1. k in 0,1.
    # W indices: k*n+i. k in 0,1. i in 0,1.
    
    # We need to map these to the global indices.
    # For M_{2,3,2}, the k indices are 0,1.
    # The j indices are 0,1.
    
    # Let's generate the 7 Strassen terms.
    # Each term c has U, V, W vectors.
    # We will add them to the global U, V, W.
    
    col_offset = 0
    
    # Strassen terms for the 2x2 submatrix (j=0,1; k=0,1)
    # Note: The standard Strassen is for 2x2 * 2x2.
    # Here we are multiplying 2x2 (A[:,0:2]) by 2x2 (B[0:2,:]).
    # The output is 2x2 (C[:,0:2]).
    
    # Term 1: M1 = (A00 + A11) (B00 + B11)
    # U: A00=1, A11=1. Others 0.
    # V: B00=1, B11=1. Others 0.
    # W: C00 += 1, C11 += 1. (So W[0,0]=1, W[1,1]=1)
    u1 = [0]*6; u1[0*3+0] = 1; u1[1*3+1] = 1
    v1 = [0]*18; v1[0*6+0] = 1; v1[1*6+1] = 1
    w1 = [0]*12; w1[0*2+0] = 1; w1[1*2+1] = 1
    add_term(u1, v1, w1, col_offset); col_offset += 1
    
    # Term 2: M2 = (A10 + A11) B00
    u2 = [0]*6; u2[1*3+0] = 1; u2[1*3+1] = 1
    v2 = [0]*18; v2[0*6+0] = 1
    w2 = [0]*12; w2[0*2+1] = 1 # C10 += M2. W index k*n+i. k=0, i=1 -> 1.
    add_term(u2, v2, w2, col_offset); col_offset += 1
    
    # Term 3: M3 = A00 (B01 - B11)
    u3 = [0]*6; u3[0*3+0] = 1
    v3 = [0]*18; v3[0*6+1] = 1; v3[1*6+1] = -1
    w3 = [0]*12; w3[1*2+0] = 1 # C01 += M3. k=1, i=0 -> 2.
    add_term(u3, v3, w3, col_offset); col_offset += 1
    
    # Term 4: M4 = A11 (B10 - B00)
    u4 = [0]*6; u4[1*3+1] = 1
    v4 = [0]*18; v4[1*6+0] = 1; v4[0*6+0] = -1
    w4 = [0]*12; w4[0*2+0] = 1 # C00 += M4. k=0, i=0 -> 0.
    w4[0*2+1] = 1 # C10 += M4. k=0, i=1 -> 1.
    add_term(u4, v4, w4, col_offset); col_offset += 1
    
    # Term 5: M5 = (A00 + A01) B11
    u5 = [0]*6; u5[0*3+0] = 1; u5[0*3+1] = 1
    v5 = [0]*18; v5[1*6+1] = 1
    w5 = [0]*12; w5[1*2+0] = -1 # C01 -= M5? No, C01 = M3 + M5. So +1.
    # Wait, C01 = M3 + M5.
    # C00 = M1 + M4 - M5 + M7. So C00 has -M5.
    # So W for M5 should contribute to C01 (+1) and C00 (-1)?
    # No, M5 is a scalar.
    # M5 = (A00+A01)B11.
    # It contributes to C01 with +1 and C00 with -1?
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes +1 to C01 and -1 to C00.
    w5 = [0]*12; w5[1*2+0] = 1; w5[0*2+0] = -1
    add_term(u5, v5, w5, col_offset); col_offset += 1
    
    # Term 6: M6 = (A10 - A00) (B00 + B01)
    u6 = [0]*6; u6[1*3+0] = 1; u6[0*3+0] = -1
    v6 = [0]*18; v6[0*6+0] = 1; v6[0*6+1] = 1
    w6 = [0]*12; w6[1*2+1] = 1 # C11 += M6. k=1, i=1 -> 3.
    add_term(u6, v6, w6, col_offset); col_offset += 1
    
    # Term 7: M7 = (A01 - A11) (B10 + B11)
    u7 = [0]*6; u7[0*3+1] = 1; u7[1*3+1] = -1
    v7 = [0]*18; v7[1*6+0] = 1; v7[1*6+1] = 1
    w7 = [0]*12; w7[1*2+1] = 1 # C11 += M7? No, C11 = M1 - M2 + M3 + M6.
    # C00 = ... + M7.
    # So M7 contributes to C00 (+1) and C11? No.
    # Check C11 formula: M1 - M2 + M3 + M6. No M7.
    # Check C00 formula: M1 + M4 - M5 + M7.
    # So M7 contributes to C00 (+1).
    w7 = [0]*12; w7[0*2+0] = 1
    add_term(u7, v7, w7, col_offset); col_offset += 1
    
    # Now the outer product part for j=2.
    # a3 = A[:,2]. b3 = B[2,:].
    # Terms:
    # 1. A02 * B20 -> C00
    # 2. A02 * B21 -> C01
    # 3. A12 * B20 -> C10
    # 4. A12 * B21 -> C11
    
    u8 = [0]*6; u8[0*3+2] = 1
    v8 = [0]*18; v8[2*6+0] = 1
    w8 = [0]*12; w8[0*2+0] = 1
    add_term(u8, v8, w8, col_offset); col_offset += 1
    
    u9 = [0]*6; u9[0*3+2] = 1
    v9 = [0]*18; v9[2*6+1] = 1
    w9 = [0]*12; w9[1*2+0] = 1
    add_term(u9, v9, w9, col_offset); col_offset += 1
    
    u10 = [0]*6; u10[1*3+2] = 1
    v10 = [0]*18; v10[2*6+0] = 1
    w10 = [0]*12; w10[0*2+1] = 1
    add_term(u10, v10, w10, col_offset); col_offset += 1
    
    u11 = [0]*6; u11[1*3+2] = 1
    v11 = [0]*18; v11[2*6+1] = 1
    w11 = [0]*12; w11[1*2+1] = 1
    add_term(u11, v11, w11, col_offset); col_offset += 1
    
    # Total 11 terms for M_{2,3,2} (k=0,1).
    # We need to repeat this for the second block (k=2,3,4,5)?
    # No, we decomposed M_{2,3,2} which covers k=0,1.
    # We need M_{2,3,3} which covers k=0,1,2.
    # The k=2 part is M_{2,3,1} with j=0,1,2?
    # No, M_{2,3,3} = M_{2,3,2} (k=0,1) + M_{2,3,1} (k=2).
    # The M_{2,3,1} part uses all j=0,1,2.
    
    # Let's add the M_{2,3,1} part for k=2.
    # A is 2x3. B is 3x1 (column 2).
    # C is 2x1.
    # Terms:
    # j=0: A00 B02 -> C02. A10 B02 -> C12.
    # j=1: A01 B12 -> C02. A11 B12 -> C12.
    # j=2: A02 B22 -> C02. A12 B22 -> C12.
    
    # 6 terms.
    
    u12 = [0]*6; u12[0*3+0] = 1
    v12 = [0]*18; v12[0*6+2] = 1
    w12 = [0]*12; w12[2*2+0] = 1 # k=2, i=0 -> 4.
    add_term(u12, v12, w12, col_offset); col_offset += 1
    
    u13 = [0]*6; u13[1*3+0] = 1
    v13 = [0]*18; v13[0*6+2] = 1
    w13 = [0]*12; w13[2*2+1] = 1 # k=2, i=1 -> 5.
    add_term(u13, v13, w13, col_offset); col_offset += 1
    
    u14 = [0]*6; u14[0*3+1] = 1
    v14 = [0]*18; v14[1*6+2] = 1
    w14 = [0]*12; w14[2*2+0] = 1
    add_term(u14, v14, w14, col_offset); col_offset += 1
    
    u15 = [0]*6; u15[1*3+1] = 1
    v15 = [0]*18; v15[1*6+2] = 1
    w15 = [0]*12; w15[2*2+1] = 1
    add_term(u15, v15, w15, col_offset); col_offset += 1
    
    u16 = [0]*6; u16[0*3+2] = 1
    v16 = [0]*18; v16[2*6+2] = 1
    w16 = [0]*12; w16[2*2+0] = 1
    add_term(u16, v16, w16, col_offset); col_offset += 1
    
    u17 = [0]*6; u17[1*3+2] = 1
    v17 = [0]*18; v17[2*6+2] = 1
    w17 = [0]*12; w17[2*2+1] = 1
    add_term(u17, v17, w17, col_offset); col_offset += 1
    
    # Total 17 terms for M_{2,3,3} (k=0,1,2).
    
    # Now repeat for the second block k=3,4,5.
    # We need to shift the k indices by 3.
    # The U and V indices for the first 3 k-values used j=0,1,2.
    # The second block uses the same A and same B structure, but k=3,4,5.
    # So we can copy the terms, but shift the k indices in V and W.
    
    # V index: j*p + k.
    # Old k: 0,1,2. New k: 3,4,5.
    # Shift in V: +3.
    # W index: k*n + i.
    # Old k: 0,1,2. New k: 3,4,5.
    # Shift in W: +3*n = +6.
    
    # We have 17 terms. We add 17 more.
    for c in range(17):
        u_new = [row[c] for row in U]
        v_new = [0]*18
        for idx, val in enumerate(V[:, c]):
             if val != 0:
                 # idx = j*6 + k. k is 0,1,2.
                 j = idx // 6
                 k = idx % 6
                 new_k = k + 3
                 new_idx = j*6 + new_k
                 v_new[new_idx] = val
        w_new = [0]*12
        for idx, val in enumerate(W[:, c]):
             if val != 0:
                 # idx = k*2 + i. k is 0,1,2.
                 k = idx // 2
                 i = idx % 2
                 new_k = k + 3
                 new_idx = new_k*2 + i
                 w_new[new_idx] = val
                 
        add_term(u_new, v_new, w_new, col_offset)
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
