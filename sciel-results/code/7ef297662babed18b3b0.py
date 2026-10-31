"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition of 4x4 matrix multiplication via recursive Strassen.
    # 4x4 = 2x2 blocks of 2x2. Outer Strassen: 7 terms. Each term is a 2x2 matmul,
    # which uses inner Strassen: 7 terms. Total rank = 7*7 = 49.
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # U: shape (n*m, rank) = (16, 49)
    # Row index: i*n + j where i in [0,4), j in [0,4)
    # Column index: t1*7 + t2 where t1 in [0,7), t2 in [0,7)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen 2x2: for matrices X (rows x, cols y) and Y (rows y, cols z)
    # M_t for t=0..6
    # U_X[x,y,t], V_Y[y,z,t], W_C[z,x,t] such that
    # C[z,x] = sum_t W_C[z,x,t] * (sum_{x,y} U_X[x,y,t]*X[x,y]) * (sum_{y,z} V_Y[y,z,t]*Y[y,z])
    
    # Define Strassen factors for 2x2:
    # X indexed by (a,b) with a,b in {0,1}, Y by (b,c), C by (c,a)
    # U_X[a,b,t], V_Y[b,c,t], W_C[c,a,t]
    
    # t=0: M0 = (X00+X11)(Y00+Y11)
    # t=1: M1 = (X10+X11)Y00
    # t=2: M2 = X00(Y01-Y11)
    # t=3: M3 = X11(Y10-Y00)
    # t=4: M4 = (X00+X01)Y11
    # t=5: M5 = (X10-X00)(Y00+Y01)
    # t=6: M6 = (X01-X11)(Y10+Y11)
    
    # U_X[a,b,t]: coefficient for X[a,b] in the left factor of M_t
    U_X = [[0]*7 for _ in range(4)]  # indexed by a*2+b
    V_Y = [[0]*7 for _ in range(4)]  # indexed by b*2+c
    W_C = [[0]*7 for _ in range(4)]  # indexed by c*2+a
    
    # t=0: left = X00+X11, right = Y00+Y11
    U_X[0][0] = 1  # X00
    U_X[3][0] = 1  # X11
    V_Y[0][0] = 1  # Y00
    V_Y[3][0] = 1  # Y11
    W_C[0][0] = 1  # C00
    W_C[1][0] = 1  # C01
    W_C[2][0] = 1  # C10
    W_C[3][0] = 1  # C11
    # C00 += M0, C01 += M0, C10 += M0, C11 += M0? No!
    # Actually Strassen: C00 = M0+M3-M4+M6, C01=M2+M4, C10=M1+M3, C11=M0-M1+M2+M5
    # So W_C is the output combination, not per-term.
    # I need to be more careful.
    
    # Let me restart with a cleaner approach.
    # For each t in 0..6, define:
    # left_factors[t][a,b] for X
    # right_factors[t][b,c] for Y
    # And the output combination: for each output (c,a), how does M_t contribute?
    
    # Actually, the standard way:
    # C = sum_t w_t * (L_t * R_t) where L_t and R_t are linear combos of X and Y blocks.
    # But here we're doing scalar decomposition, so it's:
    # C[ga, gb] = sum_t W[gb*4+ga, t] * (sum U[ga'*4+gb', t] * X[ga',gb']) * (sum V[gb'*4+gc', t] * Y[gb',gc'])
    # where the sums are over the 2x2 block entries.
    
    # Let me define for 2x2 Strassen:
    # For each term t=0..6:
    #   L_t[a,b] for a,b in {0,1}
    #   R_t[b,c] for b,c in {0,1}
    #   And the contribution to C: C[c,a] += contrib_t[c,a] * L_t dot X * R_t dot Y
    #   But actually C[c,a] = sum_t W_out[c,a,t] * (L_t : X) * (R_t : Y)
    #   where W_out encodes the Strassen combination.
    
    # Standard Strassen:
    # M0 = (X00+X11)(Y00+Y11)
    # M1 = (X10+X11)Y00
    # M2 = X00(Y01-Y11)
    # M3 = X11(Y10-Y00)
    # M4 = (X00+X01)Y11
    # M5 = (X10-X00)(Y00+Y01)
    # M6 = (X01-X11)(Y10+Y11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # So W_out:
    # W_out[0,0,0]=1, W_out[0,0,3]=1, W_out[0,0,4]=-1, W_out[0,0,6]=1  # C00
    # W_out[1,0,2]=1, W_out[1,0,4]=1                                    # C01
    # W_out[2,0,1]=1, W_out[2,0,3]=1                                    # C10
    # W_out[3,0,0]=1, W_out[3,0,1]=-1, W_out[3,0,2]=1, W_out[3,0,5]=1  # C11
    
    # L_t:
    # t=0: L[0,0]=1, L[1,1]=1
    # t=1: L[1,0]=1, L[1,1]=1
    # t=2: L[0,0]=1
    # t=3: L[1,1]=1
    # t=4: L[0,0]=1, L[0,1]=1
    # t=5: L[1,0]=1, L[0,0]=-1
    # t=6: L[0,1]=1, L[1,1]=-1
    
    # R_t:
    # t=0: R[0,0]=1, R[1,1]=1
    # t=1: R[0,0]=1
    # t=2: R[0,1]=1, R[1,1]=-1
    # t=3: R[1,0]=1, R[0,0]=-1
    # t=4: R[1,1]=1
    # t=5: R[0,0]=1, R[0,1]=1
    # t=6: R[1,0]=1, R[1,1]=1
    
    # Now for the 4x4 case:
    # A is 4x4, indexed by (I,J) where I=2*Ib+ia, J=2*Ib+jb... wait no.
    # Let me use: A[i,j] with i,j in 0..3
    # Block (bi,bj) where bi=i//2, bj=j//2, and within block: ia=i%2, jb=j%2
    # So A[2*bi+ia, 2*bj+jb]
    
    # Similarly B[j,k] with j=2*bj+jb, k=2*bk+kb
    # C[i,k] with i=2*bi+ia, k=2*bk+kb
    
    # Outer Strassen on blocks:
    # M_outer_t = (sum_{bi,bj} L_outer[t][bi,bj] * A_block[bi,bj]) * (sum_{bj,bk} R_outer[t][bj,bk] * B_block[bj,bk])
    # where A_block[bi,bj] is the 2x2 matrix A[2*bi:2*bi+2, 2*bj:2*bj+2]
    
    # Each A_block[bi,bj] has entries A[2*bi+ia, 2*bj+jb] for ia,jb in {0,1}
    # Each B_block[bj,bk] has entries B[2*bj+jb, 2*bk+kb] for jb,kb in {0,1}
    
    # The product M_outer_t is a 2x2 matrix (indexed by ia, kb)
    # M_outer_t[ia,kb] = sum_{jb} (sum_{bi} L_outer[t][bi,bj]*A[2*bi+ia,2*bj+jb]) * (sum_{bk} R_outer[t][bj,bk]*B[2*bj+jb,2*bk+kb])
    
    # Wait, this is getting complex. Let me think of it as:
    # The "left factor" for term t is a 4x4 matrix L_t^{(4)} where:
    # L_t^{(4)}[2*bi+ia, 2*bj+jb] = L_outer[t][bi,bj] * L_inner[ia,jb] ??? No.
    
    # Actually, I think the cleaner way is:
    # For each outer term t1 and inner term t2:
    # U[2*bi+ia, 2*bj+jb, t1*7+t2] = L_outer[t1][bi,bj] * L_inner[t2][ia,jb]
    # V[2*bj+jb, 2*bk+kb, t1*7+t2] = R_outer[t1][bj,bk] * R_inner[t2][jb,kb]
    # W[2*bk+kb, 2*bi+ia, t1*7+t2] = W_outer[t1][bk,bi] * W_inner[t2][kb,ia]
    
    # where W_outer[t1][bk,bi] is the output combination from outer Strassen
    # and W_inner[t2][kb,ia] is the output combination from inner Strassen
    
    # But wait, that's not quite right either. Let me think more carefully.
    
    # The full tensor: C[i,k] = sum_{j} A[i,j] B[j,k]
    # With i=2*bi+ia, j=2*bj+jb, k=2*bk+kb:
    # C[2*bi+ia, 2*bk+kb] = sum_{bj} sum_{jb} A[2*bi+ia, 2*bj+jb] B[2*bj+jb, 2*bk+kb]
    
    # Strassen decomposition says:
    # C_block[bk,bi] (2x2) = sum_{t1} W_outer[t1][bk,bi] * M_outer[t1]
    # where M_outer[t1] is the product of two 2x2 matrices:
    # M_outer[t1][ia,kb] = sum_{jb} L_outer[t1]_{ia,jb} * R_outer[t1]_{jb,kb}
    # where L_outer[t1] and R_outer[t1] are 2x2 matrices that are linear combos of A_blocks and B_blocks.
    
    # More precisely:
    # L_outer[t1] is a 2x2 matrix where (L_outer[t1])_{ia,jb} = sum_{bi'} L_outer_scalar[t1][bi',bj] * delta_{ia,?} ... 
    # This is getting confusing. Let me just brute force it.
    
    # Alternative clean approach:
    # Define the 49 rank-1 terms explicitly.
    # For each (t1, t2) in 0..6 x 0..6:
    #   U[row, col=t1*7+t2] where row = i*4+j
    #   V[row, col] where row = j*4+k
    #   W[row, col] where row = k*4+i
    
    # The key realization: the 4x4 matmul tensor is the Kronecker product of the 2x2 matmul tensor with itself, composed with Strassen.
    
    # Let me just directly construct it by thinking of the 4x4 as 2x2 of 2x2 blocks.
    
    # Outer Strassen terms (t1=0..6):
    # Define outer_L[t1][bi,bj] and outer_R[t1][bj,bk] and outer_W[t1][bk,bi]
    
    outer_L = [[0]*4 for _ in range(7)]  # [t][bi*2+bj]
    outer_R = [[0]*4 for _ in range(7)]  # [t][bj*2+bk]
    outer_W = [[0]*4 for _ in range(7)]  # [t][bk*2+bi]
    
    # t1=0: (A00+A11)(B00+B11)
    outer_L[0][0] = 1  # bi=0,bj=0 -> A00
    outer_L[0][3] = 1  # bi=1,bj=1 -> A11
    outer_R[0][0] = 1  # bj=0,bk=0 -> B00
    outer_R[0][3] = 1  # bj=1,bk=1 -> B11
    # C00 += M0, C11 += M0
    outer_W[0][0] = 1  # bk=0,bi=0 -> C00
    outer_W[0][3] = 1  # bk=1,bi=1 -> C11
    
    # t1=1: (A10+A11)B00
    outer_L[1][2] = 1  # bi=1,bj=0 -> A10
    outer_L[1][3] = 1  # bi=1,bj=1 -> A11
    outer_R[1][0] = 1  # bj=0,bk=0 -> B00
    # C10 += M1
    outer_W[1][2] = 1  # bk=0,bi=1 -> C10
    
    # t1=2: A00(B01-B11)
    outer_L[2][0] = 1  # A00
    outer_R[2][1] = 1  # bj=0,bk=1 -> B01
    outer_R[2][3] = -1 # bj=1,bk=1 -> B11
    # C01 += M2
    outer_W[2][1] = 1  # bk=1,bi=0 -> C01
    
    # t1=3: A11(B10-B00)
    outer_L[3][3] = 1  # A11
    outer_R[3][2] = 1  # bj=1,bk=0 -> B10
    outer_R[3][0] = -1 # bj=0,bk=0 -> B00
    # C10 += M3, C00 += M3
    outer_W[3][2] = 1  # C10
    outer_W[3][0] = 1  # C00
    
    # t1=4: (A00+A01)B11
    outer_L[4][0] = 1  # A00
    outer_L[4][1] = 1  # A01
    outer_R[4][3] = 1  # B11
    # C01 += M4, C00 -= M4
    outer_W[4][1] = 1  # C01
    outer_W[4][0] = -1 # C00
    
    # t1=5: (A10-A00)(B00+B01)
    outer_L[5][2] = 1  # A10
    outer_L[5][0] = -1 # A00
    outer_R[5][0] = 1  # B00
    outer_R[5][1] = 1  # B01
    # C11 += M5
    outer_W[5][3] = 1  # C11
    
    # t1=6: (A01-A11)(B10+B11)
    outer_L[6][1] = 1  # A01
    outer_L[6][3] = -1 # A11
    outer_R[6][2] = 1  # B10
    outer_R[6][3] = 1  # B11
    # C00 += M6
    outer_W[6][0] = 1  # C00
    
    # Inner Strassen for 2x2:
    inner_L = [[0]*4 for _ in range(7)]  # [t][ia*2+jb]
    inner_R = [[0]*4 for _ in range(7)]  # [t][jb*2+kb]
    inner_W = [[0]*4 for _ in range(7)]  # [t][kb*2+ia]
    
    # t2=0: (X00+X11)(Y00+Y11)
    inner_L[0][0] = 1  # ia=0,jb=0
    inner_L[0][3] = 1  # ia=1,jb=1
    inner_R[0][0] = 1  # jb=0,kb=0
    inner_R[0][3] = 1  # jb=1,kb=1
    inner_W[0][0] = 1  # kb=0,ia=0
    inner_W[0][3] = 1  # kb=1,ia=1
    inner_W[0][1] = 1  # kb=0,ia=1? No.
    # C00=M0+M3-M4+M6, C01=M2+M4, C10=M1+M3, C11=M0-M1+M2+M5
    # inner_W[t][kb*2+ia]:
    # t=0: C00+=1, C11+=1 -> inner_W[0][0]=1 (kb=0,ia=0), inner_W[0][3]=1 (kb=1,ia=1)
    inner_W[0][0] = 1
    inner_W[0][3] = 1
    
    # t2=1: (X10+X11)Y00
    inner_L[1][2] = 1  # ia=1,jb=0
    inner_L[1][3] = 1  # ia=1,jb=1
    inner_R[1][0] = 1  # jb=0,kb=0
    inner_W[1][2] = 1  # C10: kb=0,ia=1 -> index 0*2+1=1? No, kb*2+ia = 0*2+1=1
    # Wait, C10 means output row=1,col=0, so kb=0,ia=1, index=kb*2+ia=0*2+1=1
    inner_W[1][1] = 1
    
    # t2=2: X00(Y01-Y11)
    inner_L[2][0] = 1  # ia=0,jb=0
    inner_R[2][1] = 1  # jb=0,kb=1
    inner_R[2][3] = -1 # jb=1,kb=1
    inner_W[2][3] = 1  # C01: kb=1,ia=0 -> 1*2+0=2
    inner_W[2][2] = 1
    
    # t2=3: X11(Y10-Y00)
    inner_L[3][3] = 1  # ia=1,jb=1
    inner_R[3][2] = 1  # jb=1,kb=0
    inner_R[3][0] = -1 # jb=0,kb=0
    inner_W[3][1] = 1  # C10: kb=0,ia=1 -> 1
    inner_W[3][0] = 1  # C00: kb=0,ia=0 -> 0
    
    # t2=4: (X00+X01)Y11
    inner_L[4][0] = 1  # ia=0,jb=0
    inner_L[4][1] = 1  # ia=0,jb=1
    inner_R[4][3] = 1  # jb=1,kb=1
    inner_W[4][2] = 1  # C01: kb=1,ia=0 -> 2
    inner_W[4][0] = -1 # C00: kb=0,ia=0 -> 0
    
    # t2=5: (X10-X00)(Y00+Y01)
    inner_L[5][2] = 1  # ia=1,jb=0
    inner_L[5][0] = -1 # ia=0,jb=0
    inner_R[5][0] = 1  # jb=0,kb=0
    inner_R[5][1] = 1  # jb=0,kb=1
    inner_W[5][3] = 1  # C11: kb=1,ia=1 -> 3
    
    # t2=6: (X01-X11)(Y10+Y11)
    inner_L[6][1] = 1  # ia=0,jb=1
    inner_L[6][3] = -1 # ia=1,jb=1
    inner_R[6][2] = 1  # jb=1,kb=0
    inner_R[6][3] = 1  # jb=1,kb=1
    inner_W[6][0] = 1  # C00: kb=0,ia=0 -> 0
    
    # Now build the full 4x4 factors:
    # For each (t1,t2), column c = t1*7+t2
    # U[i,j,c] where i=2*bi+ia, j=2*bj+jb:
    #   U[2*bi+ia, 2*bj+jb, c] = outer_L[t1][bi*2+bj] * inner_L[t2][ia*2+jb]
    # V[j,k,c] where j=2*bj+jb, k=2*bk+kb:
    #   V[2*bj+jb, 2*bk+kb, c] = outer_R[t1][bj*2+bk] * inner_R[t2][jb*2+kb]
    # W[k,i,c] where k=2*bk+kb, i=2*bi+ia:
    #   W[2*bk+kb, 2*bi+ia, c] = outer_W[t1][bk*2+bi] * inner_W[t2][kb*2+ia]
    
    for t1 in range(7):
        for t2 in range(7):
            c = t1 * 7 + t2
            for bi in range(2):
                for bj in range(2):
                    for ia in range(2):
                        for jb in range(2):
                            i = 2 * bi + ia
                            j = 2 * bj + jb
                            val = outer_L[t1][bi * 2 + bj] * inner_L[t2][ia * 2 + jb]
                            if val != 0:
                                U[i * 4 + j][c] = val
            for bj in range(2):
                for bk in range(2):
                    for jb in range(2):
                        for kb in range(2):
                            j = 2 * bj + jb
                            k = 2 * bk + kb
                            val = outer_R[t1][bj * 2 + bk] * inner_R[t2][jb * 2 + kb]
                            if val != 0:
                                V[j * 4 + k][c] = val
            for bk in range(2):
                for bi in range(2):
                    for kb in range(2):
                        for ia in range(2):
                            k = 2 * bk + kb
                            i = 2 * bi + ia
                            val = outer_W[t1][bk * 2 + bi] * inner_W[t2][kb * 2 + ia]
                            if val != 0:
                                W[k * 4 + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
