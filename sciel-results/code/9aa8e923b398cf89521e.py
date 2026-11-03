import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U size: n*m = 6
    # V size: m*p = 9
    # W size: p*n = 6
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # Indices:
    # U[i*m + j] for i in 0..1, j in 0..2
    # V[j*p + k] for j in 0..2, k in 0..2
    # W[k*n + i] for k in 0..2, i in 0..1
    
    # 1. Strassen for j in {0,1}, k in {0,1}
    # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # C indices: (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3
    
    # M1 = (A00 + A11) * B00
    u1 = [0]*6; u1[0]=1; u1[4]=1
    v1 = [0]*9; v1[0]=1
    w1 = [0]*6; w1[0]=1; w1[1]=1 # C00, C10
    add_term(u1, v1, w1)
    
    # M2 = (A10 + A11) * B11
    u2 = [0]*6; u2[3]=1; u2[4]=1
    v2 = [0]*9; v2[4]=1
    w2 = [0]*6; w2[3]=2 # C11 (from A10B11 and A11B11)
    add_term(u2, v2, w2)
    
    # M3 = A00 * (B01 - B11)
    u3 = [0]*6; u3[0]=1
    v3 = [0]*9; v3[1]=1; v3[4]=-1
    w3 = [0]*6; w3[2]=1 # C01
    add_term(u3, v3, w3)
    
    # M4 = A11 * (B10 + B00)
    u4 = [0]*6; u4[4]=1
    v4 = [0]*9; v4[3]=1; v4[0]=1
    w4 = [0]*6; w4[1]=1 # C10
    add_term(u4, v4, w4)
    
    # M5 = (A00 + A01) * B11
    u5 = [0]*6; u5[0]=1; u5[1]=1
    v5 = [0]*9; v5[4]=1
    w5 = [0]*6; w5[2]=1; w5[3]=1 # C01, C11
    add_term(u5, v5, w5)
    
    # M6 = (A10 - A00) * (B00 + B01)
    u6 = [0]*6; u6[3]=1; u6[0]=-1
    v6 = [0]*9; v6[0]=1; v6[1]=1
    w6 = [0]*6; w6[0]=-1; w6[2]=1 # -C00, C01
    # Wait, C00 gets -A00B00 - A00B01?
    # M6 = (A10 - A00)(B00 + B01) = A10B00 + A10B01 - A00B00 - A00B01
    # A10B00 contributes to C10.
    # A10B01 contributes to C11.
    # -A00B00 contributes to -C00.
    # -A00B01 contributes to -C01.
    # My w6 above is wrong.
    # Let's re-evaluate W for M6.
    # Terms: A10B00 (C10), A10B01 (C11), -A00B00 (C00), -A00B01 (C01).
    # C10 index 1. C11 index 3. C00 index 0. C01 index 2.
    w6_correct = [0]*6
    w6_correct[1] = 1  # A10B00
    w6_correct[3] = 1  # A10B01
    w6_correct[0] = -1 # -A00B00
    w6_correct[2] = -1 # -A00B01
    # Replace w6
    terms[-1] = (u6, v6, w6_correct)
    
    # M7 = (A01 - A11) * (B10 + B11)
    u7 = [0]*6; u7[1]=1; u7[4]=-1
    v7 = [0]*9; v7[3]=1; v7[4]=1
    # Terms: A01B10 (C01), A01B11 (C01? No C01 is i=0,k=1. A01 is i=0,j=1. B10 is j=1,k=0. C00? No.
    # C_ik = sum A_ij B_jk.
    # A01 (i=0,j=1) * B10 (j=1,k=0) -> C_00.
    # A01 (i=0,j=1) * B11 (j=1,k=1) -> C_01.
    # -A11 (i=1,j=1) * B10 (j=1,k=0) -> -C_10.
    # -A11 (i=1,j=1) * B11 (j=1,k=1) -> -C_11.
    w7_correct = [0]*6
    w7_correct[0] = 1  # A01B10 -> C00
    w7_correct[2] = 1  # A01B11 -> C01
    w7_correct[1] = -1 # -A11B10 -> -C10
    w7_correct[3] = -1 # -A11B11 -> -C11
    add_term(u7, v7, w7_correct)

    # 2. Schoolbook for j in {0,1}, k=2
    # A00 B02 -> C02 (k=2, i=0 -> idx 4)
    u8 = [0]*6; u8[0]=1
    v8 = [0]*9; v8[2]=1 # j=0, k=2 -> 0*3+2=2
    w8 = [0]*6; w8[4]=1
    add_term(u8, v8, w8)
    
    # A01 B12 -> C02 (k=2, i=0 -> idx 4)
    u9 = [0]*6; u9[1]=1
    v9 = [0]*9; v9[5]=1 # j=1, k=2 -> 1*3+2=5
    w9 = [0]*6; w9[4]=1
    add_term(u9, v9, w9)
    
    # A10 B02 -> C12 (k=2, i=1 -> idx 5)
    u10 = [0]*6; u10[3]=1
    v10 = [0]*9; v10[2]=1
    w10 = [0]*6; w10[5]=1
    add_term(u10, v10, w10)
    
    # A11 B12 -> C12 (k=2, i=1 -> idx 5)
    u11 = [0]*6; u11[4]=1
    v11 = [0]*9; v11[5]=1
    w11 = [0]*6; w11[5]=1
    add_term(u11, v11, w11)

    # 3. Schoolbook for j=2, k in {0,1,2}
    # A02 B20 -> C00 (k=0, i=0 -> idx 0)
    u12 = [0]*6; u12[2]=1 # i=0, j=2 -> 0*3+2=2
    v12 = [0]*9; v12[6]=1 # j=2, k=0 -> 2*3+0=6
    w12 = [0]*6; w12[0]=1
    add_term(u12, v12, w12)
    
    # A12 B20 -> C10 (k=0, i=1 -> idx 1)
    u13 = [0]*6; u13[5]=1 # i=1, j=2 -> 1*3+2=5
    v13 = [0]*9; v13[6]=1
    w13 = [0]*6; w13[1]=1
    add_term(u13, v13, w13)
    
    # A02 B21 -> C01 (k=1, i=0 -> idx 2)
    u14 = [0]*6; u14[2]=1
    v14 = [0]*9; v14[7]=1 # j=2, k=1 -> 2*3+1=7
    w14 = [0]*6; w14[2]=1
    add_term(u14, v14, w14)
    
    # A12 B21 -> C11 (k=1, i=1 -> idx 3)
    u15 = [0]*6; u15[5]=1
    v15 = [0]*9; v15[7]=1
    w15 = [0]*6; w15[3]=1
    add_term(u15, v15, w15)
    
    # A02 B22 -> C02 (k=2, i=0 -> idx 4)
    u16 = [0]*6; u16[2]=1
    v16 = [0]*9; v16[8]=1 # j=2, k=2 -> 2*3+2=8
    w16 = [0]*6; w16[4]=1
    add_term(u16, v16, w16)
    
    # A12 B22 -> C12 (k=2, i=1 -> idx 5)
    u17 = [0]*6; u17[5]=1
    v17 = [0]*9; v17[8]=1
    w17 = [0]*6; w17[5]=1
    add_term(u17, v17, w17)

    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        u, v, w = terms[c]
        for i in range(n*m):
            U[i][c] = u[i]
        for j in range(m*p):
            V[j][c] = v[j]
        for k in range(p*n):
            W[k][c] = w[k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
