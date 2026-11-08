"""Program for matrix multiplication 4x4x4 using rank-49 Kronecker product of Strassen decompositions."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # The 4x4 matrix multiplication tensor is the Kronecker product of two 2x2 matmul tensors.
    # T_4[i,j,k] = T_2[i_hi,j_hi,k_hi] * T_2[i_lo,j_lo,k_lo]
    # where i = 2*i_hi + i_lo, etc.
    #
    # The 2x2 matmul tensor has rank 7 (Strassen). The Kronecker product of two rank-7
    # decompositions gives a rank-49 decomposition of the 4x4 matmul tensor.
    #
    # For 2x2 Strassen:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y where:
    # z00 = m1 - m4 + m5 - m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m3 + m6 - m7
    # where:
    # m1 = (x00+x11)*(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y10)
    # m7 = (x01-x11)*(y01+y11)
    #
    # The 2x2 matmul tensor T_2[i,j,k] = 1 for all i,j,k in {0,1}^3 (it's the all-ones tensor
    # for 2x2 matmul: C[i,k] = sum_j A[i,j]*B[j,k]).
    #
    # Strassen decomposition: T_2[i,j,k] = sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r]
    #
    # Let me define the Strassen factors for 2x2 matmul.
    # U2 has shape (4, 7) where index is i*2+j
    # V2 has shape (4, 7) where index is j*2+k
    # W2 has shape (4, 7) where index is k*2+i
    
    # Strassen 2x2 matmul factors
    # m1 = (x00+x11)*(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y10)
    # m7 = (x01-x11)*(y01+y11)
    
    # U2[i*2+j, r]: coefficient of A[i,j] in the r-th Strassen product
    # V2[j*2+k, r]: coefficient of B[j,k] in the r-th Strassen product
    # W2[k*2+i, r]: coefficient of C[i,k] in the r-th Strassen product
    
    # Let me work out each:
    # r=0 (m1): (x00+x11)*(y00+y11)
    #   A terms: x00 (i=0,j=0), x11 (i=1,j=1)
    #   B terms: y00 (j=0,k=0), y11 (j=1,k=1)
    #   C terms: contributes to z00 and z11
    #   z00 = m1 - m4 + m5 - m7, so coefficient of m1 in z00 is +1
    #   z11 = m1 - m3 + m6 - m7, so coefficient of m1 in z11 is +1
    #   W2[0*2+0, 0] = 1 (C[0,0])
    #   W2[1*2+1, 0] = 1 (C[1,1])
    
    # r=1 (m2): (x10+x11)*y00
    #   A terms: x10 (i=1,j=0), x11 (i=1,j=1)
    #   B terms: y00 (j=0,k=0)
    #   C terms: z10 = m2 + m4, so coefficient of m2 in z10 is +1
    #   W2[0*2+1, 1] = 1 (C[1,0])
    
    # r=2 (m3): x00*(y01-y11)
    #   A terms: x00 (i=0,j=0)
    #   B terms: y01 (j=0,k=1), y11 (j=1,k=1) with signs +1, -1
    #   C terms: z01 = m3 + m5, z11 = m1 - m3 + m6 - m7
    #   W2[1*2+0, 2] = 1 (C[0,1])
    #   W2[1*2+1, 2] = -1 (C[1,1])
    
    # r=3 (m4): x11*(y10-y00)
    #   A terms: x11 (i=1,j=1)
    #   B terms: y10 (j=1,k=0), y00 (j=0,k=0) with signs +1, -1
    #   C terms: z00 = m1 - m4 + m5 - m7, z10 = m2 + m4
    #   W2[0*2+0, 3] = -1 (C[0,0])
    #   W2[0*2+1, 3] = 1 (C[1,0])
    
    # r=4 (m5): (x00+x01)*y11
    #   A terms: x00 (i=0,j=0), x01 (i=0,j=1)
    #   B terms: y11 (j=1,k=1)
    #   C terms: z00 = m1 - m4 + m5 - m7, z01 = m3 + m5
    #   W2[0*2+0, 4] = 1 (C[0,0])
    #   W2[1*2+0, 4] = 1 (C[0,1])
    
    # r=5 (m6): (x10-x00)*(y00+y10)
    #   A terms: x10 (i=1,j=0), x00 (i=0,j=0) with signs +1, -1
    #   B terms: y00 (j=0,k=0), y10 (j=1,k=0)
    #   C terms: z11 = m1 - m3 + m6 - m7
    #   W2[1*2+1, 5] = 1 (C[1,1])
    
    # r=6 (m7): (x01-x11)*(y01+y11)
    #   A terms: x01 (i=0,j=1), x11 (i=1,j=1) with signs +1, -1
    #   B terms: y01 (j=0,k=1), y11 (j=1,k=1)
    #   C terms: z00 = m1 - m4 + m5 - m7, z11 = m1 - m3 + m6 - m7
    #   W2[0*2+0, 6] = -1 (C[0,0])
    #   W2[1*2+1, 6] = -1 (C[1,1])
    
    # Now build the matrices:
    U2 = [[0]*7 for _ in range(4)]  # 4x7, index i*2+j
    V2 = [[0]*7 for _ in range(4)]  # 4x7, index j*2+k
    W2 = [[0]*7 for _ in range(4)]  # 4x7, index k*2+i
    
    # r=0: m1 = (x00+x11)*(y00+y11)
    U2[0*2+0][0] = 1  # x00
    U2[1*2+1][0] = 1  # x11
    V2[0*2+0][0] = 1  # y00
    V2[1*2+1][0] = 1  # y11
    W2[0*2+0][0] = 1  # C[0,0]
    W2[1*2+1][0] = 1  # C[1,1]
    
    # r=1: m2 = (x10+x11)*y00
    U2[1*2+0][1] = 1  # x10
    U2[1*2+1][1] = 1  # x11
    V2[0*2+0][1] = 1  # y00
    W2[0*2+1][1] = 1  # C[1,0]
    
    # r=2: m3 = x00*(y01-y11)
    U2[0*2+0][2] = 1  # x00
    V2[0*2+1][2] = 1  # y01
    V2[1*2+1][2] = -1  # y11
    W2[1*2+0][2] = 1  # C[0,1]
    W2[1*2+1][2] = -1  # C[1,1]
    
    # r=3: m4 = x11*(y10-y00)
    U2[1*2+1][3] = 1  # x11
    V2[1*2+0][3] = 1  # y10
    V2[0*2+0][3] = -1  # y00
    W2[0*2+0][3] = -1  # C[0,0]
    W2[0*2+1][3] = 1  # C[1,0]
    
    # r=4: m5 = (x00+x01)*y11
    U2[0*2+0][4] = 1  # x00
    U2[0*2+1][4] = 1  # x01
    V2[1*2+1][4] = 1  # y11
    W2[0*2+0][4] = 1  # C[0,0]
    W2[1*2+0][4] = 1  # C[0,1]
    
    # r=5: m6 = (x10-x00)*(y00+y10)
    U2[1*2+0][5] = 1  # x10
    U2[0*2+0][5] = -1  # x00
    V2[0*2+0][5] = 1  # y00
    V2[1*2+0][5] = 1  # y10
    W2[1*2+1][5] = 1  # C[1,1]
    
    # r=6: m7 = (x01-x11)*(y01+y11)
    U2[0*2+1][6] = 1  # x01
    U2[1*2+1][6] = -1  # x11
    V2[0*2+1][6] = 1  # y01
    V2[1*2+1][6] = 1  # y11
    W2[0*2+0][6] = -1  # C[0,0]
    W2[1*2+1][6] = -1  # C[1,1]
    
    # Verify 2x2 decomposition:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j][r] * V2[j*2+k][r] * W2[k*2+i][r] for r in range(7))
                assert s == 1, f"2x2 decomposition failed at i={i},j={j},k={k}: sum={s}"
    
    # Now construct the 4x4 decomposition via Kronecker product.
    # T_4[i,j,k] = T_2[i_hi,j_hi,k_hi] * T_2[i_lo,j_lo,k_lo]
    # where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo, k = 2*k_hi + k_lo
    #
    # U4[i*4+j, r1*7+r2] = U2[i_hi*2+j_hi, r1] * U2[i_lo*2+j_lo, r2]
    # V4[j*4+k, r1*7+r2] = V2[j_hi*2+k_hi, r1] * V2[j_lo*2+k_lo, r2]
    # W4[k*4+i, r1*7+r2] = W2[k_hi*2+i_hi, r1] * W2[k_lo*2+i_lo, r2]
    
    rank = 49
    U4 = [[0]*rank for _ in range(16)]
    V4 = [[0]*rank for _ in range(16)]
    W4 = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i in range(4):
                i_hi = i >> 1
                i_lo = i & 1
                for j in range(4):
                    j_hi = j >> 1
                    j_lo = j & 1
                    U4[i*4+j][col] = U2[i_hi*2+j_hi][r1] * U2[i_lo*2+j_lo][r2]
            for j in range(4):
                j_hi = j >> 1
                j_lo = j & 1
                for k in range(4):
                    k_hi = k >> 1
                    k_lo = k & 1
                    V4[j*4+k][col] = V2[j_hi*2+k_hi][r1] * V2[j_lo*2+k_lo][r2]
            for k in range(4):
                k_hi = k >> 1
                k_lo = k & 1
                for i in range(4):
                    i_hi = i >> 1
                    i_lo = i & 1
                    W4[k*4+i][col] = W2[k_hi*2+i_hi][r1] * W2[k_lo*2+i_lo][r2]
    
    # Verify 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U4[i*4+j][r] * V4[j*4+k][r] * W4[k*4+i][r] for r in range(49))
                assert s == 1, f"4x4 decomposition failed at i={i},j={j},k={k}: sum={s}"
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
