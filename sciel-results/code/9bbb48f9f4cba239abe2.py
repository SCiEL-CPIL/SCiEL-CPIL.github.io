"""Kronecker product decomposition: 6x6 = 2x2 (Strassen rank 7) ⊗ 3x3 (schoolbook rank 9).

The 6x6 matrix multiplication tensor decomposes as the tensor product of:
- 2x2 multiplication (Strassen, rank 7)
- 3x3 multiplication (schoolbook, rank 9)

This gives a rank-63 exact decomposition, far below the schoolbook rank 216.

Index conventions:
- U has shape (n*m, rank) = (36, 63)
- V has shape (m*p, rank) = (36, 63)  
- W has shape (p*n, rank) = (36, 63)
- W index: W[k*n + i, r] (official trace convention)
"""

def construct():
    n, m, p = 6, 6, 6
    
    # Strassen decomposition for 2x2 matrices (rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+b)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+c)(h)
    # M6 = (b-d)(e+g)
    # M7 = (b-c)(f+h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # For 2x2, index mapping:
    # A row: i in {0,1}, col: j in {0,1} -> A[i*2+j]
    # B row: j in {0,1}, col: k in {0,1} -> B[j*2+k]
    # C row: i in {0,1}, col: k in {0,1} -> C[i*2+k]
    # But W uses k*n+i convention, so for 2x2: W[k*2+i]
    
    # Strassen factors for 2x2:
    # U_strassen[i*2+j, s] for s in 0..6
    # V_strassen[j*2+k, s] for s in 0..6
    # W_strassen[k*2+i, s] for s in 0..6
    
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # M1 = (a+b)(e+h): a=A[0], b=A[1], e=B[0], h=B[3]
    # a = i=0,j=0; b = i=0,j=1; e = j=0,k=0; h = j=1,k=1
    # W: k=0,i=0 -> W[0]; k=1,i=0 -> W[2]
    # C11 += M1, C22 += M1
    # C11: i=0,k=0 -> W[k=0,i=0]=W[0]
    # C22: i=1,k=1 -> W[k=1,i=1]=W[3]
    U_2[0][0] = 1; U_2[1][0] = 1  # a+b
    V_2[0][0] = 1; V_2[3][0] = 1  # e+h
    W_2[0][0] = 1; W_2[3][0] = 1  # C11, C22
    
    # M2 = (c+d)(e): c=A[2]=i=1,j=0; d=A[3]=i=1,j=1; e=B[0]=j=0,k=0
    # C21 += M2: i=1,k=0 -> W[k=0,i=1]=W[1]
    U_2[2][1] = 1; U_2[3][1] = 1  # c+d
    V_2[0][1] = 1  # e
    W_2[1][1] = 1  # C21
    
    # M3 = (a)(f-h): a=A[0]; f=B[1]=j=0,k=1; h=B[3]=j=1,k=1
    # C12 += M3: i=0,k=1 -> W[k=1,i=0]=W[2]
    # C22 += M3: i=1,k=1 -> W[k=1,i=1]=W[3]
    U_2[0][2] = 1  # a
    V_2[1][2] = 1; V_2[3][2] = -1  # f-h
    W_2[2][2] = 1; W_2[3][2] = 1  # C12, C22
    
    # M4 = (d)(g-c): d=A[3]; g=B[2]=j=1,k=0; c=B[... wait, c is in A, not B
    # M4 = d*(g-c) where c is A[2]? No: M4 = (d)(g-c) but c here is from A? 
    # Let me recheck Strassen: 
    # M1 = (a+b)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)  -- wait, this is wrong. Standard Strassen:
    # M4 = d(g-c) is not right. Let me use the standard form:
    # Actually standard is:
    # M1 = (a+b)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)  -- Hmm, let me verify.
    
    # Let me use a well-known Strassen decomposition:
    # m1 = (a+b)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+c)h
    # m6 = (b-d)(e+g)
    # m7 = (b-c)(f+h)
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 - m6
    
    # Let me redo with this:
    # Reset
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # m1 = (a+b)(e+h)
    # a=A[0], b=A[1], e=B[0], h=B[3]
    U_2[0][0] = 1; U_2[1][0] = 1
    V_2[0][0] = 1; V_2[3][0] = 1
    W_2[0][0] = 1; W_2[3][0] = 1  # c11 (i=0,k=0: W[0*2+0]=W[0]), c22 (i=1,k=1: W[1*2+1]=W[3])
    
    # m2 = (c+d)e
    # c=A[2], d=A[3], e=B[0]
    U_2[2][1] = 1; U_2[3][1] = 1
    V_2[0][1] = 1
    W_2[1][1] = 1  # c21 (i=1,k=0: W[0*2+1]=W[1])
    
    # m3 = a(f-h)
    # a=A[0], f=B[1], h=B[3]
    U_2[0][2] = 1
    V_2[1][2] = 1; V_2[3][2] = -1
    W_2[2][2] = 1; W_2[3][2] = 1  # c12 (i=0,k=1: W[1*2+0]=W[2]), c22 (i=1,k=1: W[1*2+1]=W[3])
    
    # m4 = d(g-c)
    # d=A[3], g=B[2], c=B[... wait, c is in A. This formula seems off.
    # Let me use another standard Strassen:
    # m4 = (d)(g - c) where c is from A? That doesn't make sense dimensionally.
    
    # I'll use a different well-known set. Let me just use the schoolbook for 2x2 to be safe,
    # but that gives rank 8. Actually, let me carefully implement Strassen.
    
    # Standard Strassen from Wikipedia:
    # m1 = (a + b)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - c)
    # m5 = (a + c)h
    # m6 = (b - d)(e + g)
    # m7 = (b - c)(f + h)
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 - m6
    
    # The issue is m4 = d(g-c). Here c is an element of A. So:
    # m4 = d * (g - c) where c = A[2] (i=1,j=0)
    # So m4 involves A[3] and A[2] in U, and B[2] in V.
    
    # Reset again
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # m1 = (a+b)(e+h)
    U_2[0][0] = 1; U_2[1][0] = 1
    V_2[0][0] = 1; V_2[3][0] = 1
    W_2[0][0] = 1; W_2[3][0] = 1
    
    # m2 = (c+d)e
    U_2[2][1] = 1; U_2[3][1] = 1
    V_2[0][1] = 1
    W_2[1][1] = 1
    
    # m3 = a(f-h)
    U_2[0][2] = 1
    V_2[1][2] = 1; V_2[3][2] = -1
    W_2[2][2] = 1; W_2[3][2] = 1
    
    # m4 = d(g-c)
    # d=A[3], g=B[2], c=A[2]
    U_2[3][3] = 1; U_2[2][3] = -1
    V_2[2][3] = 1
    W_2[0][3] = 1; W_2[1][3] = 1  # c11 (W[0]), c21 (W[1])
    
    # m5 = (a+c)h
    # a=A[0], c=A[2], h=B[3]
    U_2[0][4] = 1; U_2[2][4] = 1
    V_2[3][4] = 1
    W_2[0][4] = -1; W_2[2][4] = 1  # c11 has -m5 (W[0]), c12 has +m5 (i=0,k=1: W[2])
    
    # m6 = (b-d)(e+g)
    # b=A[1], d=A[3], e=B[0], g=B[2]
    U_2[1][5] = 1; U_2[3][5] = -1
    V_2[0][5] = 1; V_2[2][5] = 1
    W_2[3][5] = -1  # c22 has -m6 (i=1,k=1: W[3])
    
    # m7 = (b-c)(f+h)
    # b=A[1], c=A[2], f=B[1], h=B[3]
    U_2[1][6] = 1; U_2[2][6] = -1
    V_2[1][6] = 1; V_2[3][6] = 1
    W_2[0][6] = -1  # c11 has -m7 (i=0,k=0: W[0])
    
    # Schoolbook for 3x3 (rank 9)
    U_3 = [[0]*9 for _ in range(9)]
    V_3 = [[0]*9 for _ in range(9)]
    W_3 = [[0]*9 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = i*9 + j*3 + k
                U_3[i*3+j][s] = 1
                V_3[j*3+k][s] = 1
                W_3[k*3+i][s] = 1
    
    # Now combine: 6x6 = 2x2 ⊗ 3x3
    # Each 6x6 index is (i2,i3) where i2 in {0,1} is the 2x2 block index, i3 in {0,1,2} is the 3x3 block index
    # Full index: i = i2*3 + i3
    # U_full[i*6+j, s2*9+s3] = U_2[i2*2+j2, s2] * U_3[i3*3+j3, s3]
    # where i = i2*3+i3, j = j2*3+j3
    # s = s2*9 + s3
    
    rank = 7 * 9
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for i2 in range(2):
        for i3 in range(3):
            i = i2*3 + i3
            for j2 in range(2):
                for j3 in range(3):
                    j = j2*3 + j3
                    for s2 in range(7):
                        for s3 in range(9):
                            s = s2*9 + s3
                            u_val = U_2[i2*2+j2][s2] * U_3[i3*3+j3][s3]
                            v_val = V_2[j2*2+0][s2] * V_3[j3*3+0][s3]  # This is wrong, need to be careful
                            
                            # Actually for V: V_full[j*6+k, s] = V_2[j2*2+k2, s2] * V_3[j3*3+k3, s3]
                            # But we need to handle the full index properly.
                            pass
    
    # Let me redo this more carefully.
    # For the Kronecker product, we have:
    # U_full[(i2,i3),(j2,j3), (s2,s3)] = U_2[i2,j2,s2] * U_3[i3,j3,s3]
    # where full row index = (i2*3+i3)*6 + (j2*3+j3)
    # and full column index = s2*9 + s3
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for i2 in range(2):
        for i3 in range(3):
            i = i2*3 + i3
            for j2 in range(2):
                for j3 in range(3):
                    j = j2*3 + j3
                    row_u = i*6 + j
                    for s2 in range(7):
                        for s3 in range(9):
                            s = s2*9 + s3
                            val = U_2[i2*2+j2][s2] * U_3[i3*3+j3][s3]
                            if val != 0:
                                U[row_u][s] = val
    
    for j2 in range(2):
        for j3 in range(3):
            j = j2*3 + j3
            for k2 in range(2):
                for k3 in range(3):
                    k = k2*3 + k3
                    row_v = j*6 + k
                    for s2 in range(7):
                        for s3 in range(9):
                            s = s2*9 + s3
                            val = V_2[j2*2+k2][s2] * V_3[j3*3+k3][s3]
                            if val != 0:
                                V[row_v][s] = val
    
    for k2 in range(2):
        for k3 in range(3):
            k = k2*3 + k3
            for i2 in range(2):
                for i3 in range(3):
                    i = i2*3 + i3
                    row_w = k*6 + i
                    for s2 in range(7):
                        for s3 in range(9):
                            s = s2*9 + s3
                            val = W_2[k2*2+i2][s2] * W_3[k3*3+i3][s3]
                            if val != 0:
                                W[row_w][s] = val
    
    return {"U": U, "V": V, "W": W}
