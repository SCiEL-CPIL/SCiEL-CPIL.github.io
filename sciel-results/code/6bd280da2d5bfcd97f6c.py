import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # Build the rank-22 decomposition analytically
    # Strategy: Use Strassen (rank 7) for multiplying 2x2 blocks
    # For 2x3 * 3x4:
    #   A = [A_top(2x3); A_bottom(2x3)]
    #   B = [B_left(3x2); B_right(3x2)]
    #   Actually, let's think of it as:
    #   A is 2x3, B is 3x4
    #   Split A into [A_1(2x1), A_2(2x1), A_3(2x1)]
    #   Split B into [B_1(1x4), B_2(1x4), B_3(1x4)]
    #   C = A_1*B_1 + A_2*B_2 + A_3*B_3 (rank 12 if done naively, but we can do better)
    
    # Better approach: Group columns of A and rows of B
    # A = [A_L (2x2), A_R (2x1)]
    # B = [B_L (2x4); B_R (1x4)]
    # C = A_L * B_L + A_R * B_R
    # A_L (2x2) * B_L (2x4): Split B_L = [B_LL (2x2), B_LR (2x2)]
    # A_L * B_LL uses Strassen (rank 7)
    # A_L * B_LR uses Strassen (rank 7)
    # A_R (2x1) * B_R (1x4): rank 8 (outer product)
    # Total: 7 + 7 + 8 = 22

    # Let's define the decomposition more carefully.
    # A is 2x3, B is 3x4.
    # Let A = [[a11, a12, a13], [a21, a22, a23]]
    # Let B = [[b11, b12, b13, b14],
    #          [b21, b22, b23, b24],
    #          [b31, b32, b33, b34]]
    # C = A * B is 2x4.
    # C[0,0] = a11*b11 + a12*b21 + a13*b31
    # C[0,1] = a11*b12 + a12*b22 + a13*b32
    # C[0,2] = a11*b13 + a12*b23 + a13*b33
    # C[0,3] = a11*b14 + a12*b24 + a13*b34
    # C[1,0] = a21*b11 + a22*b21 + a23*b31
    # C[1,1] = a21*b12 + a22*b22 + a23*b32
    # C[1,2] = a21*b13 + a22*b23 + a23*b33
    # C[1,3] = a21*b14 + a22*b24 + a23*b34

    # Strassen for 2x2 * 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For A_L (2x2) * B_LL (2x2):
    # A_L = [[a11, a12], [a21, a22]]
    # B_LL = [[b11, b12], [b21, b22]]
    # This gives C[0,0], C[0,1], C[1,0], C[1,1]

    # For A_L (2x2) * B_LR (2x2):
    # B_LR = [[b13, b14], [b23, b24]]
    # This gives C[0,2], C[0,3], C[1,2], C[1,3]

    # For A_R (2x1) * B_R (1x4):
    # A_R = [[a13], [a23]]
    # B_R = [[b31, b32, b33, b34]]
    # This contributes a13*b3j to C[0,j] and a23*b3j to C[1,j] for j=1,2,3,4

    # Let's build the factors.
    # U has shape (6, 22), V has shape (12, 22), W has shape (8, 22)
    # U[i*m+j, r] where i in 0..1, j in 0..2 -> U[0..5, :]
    # V[j*p+k, r] where j in 0..2, k in 0..3 -> V[0..11, :]
    # W[k*n+i, r] where k in 0..3, i in 0..1 -> W[0..7, :]
    # Output C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j

    # Wait, the tensor is T_{i,j,k} = U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # C[i,k] = sum_j T_{i,j,k} = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Let's assign the 22 terms to the 3 groups.
    # Group 1: Strassen for A_L * B_LL (terms 0-6)
    # Group 2: Strassen for A_L * B_LR (terms 7-13)
    # Group 3: Outer product for A_R * B_R (terms 14-21)

    U = [[0] * 22 for _ in range(6)]
    V = [[0] * 22 for _ in range(12)]
    W = [[0] * 22 for _ in range(8)]

    # Group 1: A_L * B_LL
    # A_L = [[a11, a12], [a21, a22]] -> U[0]=a11, U[1]=a12, U[3]=a21, U[4]=a22
    # B_LL = [[b11, b12], [b21, b22]] -> V[0]=b11, V[1]=b12, V[4]=b21, V[5]=b22
    # Output C[0,0]=sum_j a1j*bj1, C[0,1]=sum_j a1j*bj2, C[1,0]=sum_j a2j*bj1, C[1,1]=sum_j a2j*bj2
    
    # M1 = (a11+a22)(b11+b22)
    # U[0,0] += 1, U[4,0] += 1
    # V[0,0] += 1, V[4,0] += 1
    # C[0,0] gets M1, C[1,1] gets M1
    # W[0,0] += 1 (for C[0,0]), W[1,1]... wait W[k*n+i]
    # C[0,0] is i=0, k=0 -> W[0,0]
    # C[0,1] is i=0, k=1 -> W[2,0] (k=1, n=2 -> 1*2+0=2)
    # C[1,0] is i=1, k=0 -> W[1,1] (k=0, n=2 -> 0*2+1=1)
    # C[1,1] is i=1, k=1 -> W[3,1] (k=1, n=2 -> 1*2+1=3)

    # Let's define a helper to add a term
    def add_term(u_indices, v_indices, w_index, coeff=1):
        for u_idx in u_indices:
            U[u_idx][r] += coeff
        for v_idx in v_indices:
            V[v_idx][r] += coeff
        W[w_index][r] += coeff

    r = 0
    # M1 = (a11+a22)(b11+b22) -> C[0,0] += M1, C[1,1] += M1
    # But Strassen formula: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # So M1 contributes to C[0,0] (+1) and C[1,1] (+1)
    # But we only have one W per term r.
    # This means we need to split the W if a term contributes to multiple outputs.
    # Actually, the tensor decomposition is T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
    # C[i,k] = sum_j T_{i,j,k}
    # So if a rank-1 term has W_{k,i} non-zero for multiple (k,i), it contributes to multiple C[i,k].
    # In Strassen, M1 contributes to C[0,0] and C[1,1].
    # So W[0,0] (k=0,i=0) and W[3,1] (k=1,i=1) both need to be 1 for this term.
    # This is fine, W is just a matrix.

    # Let's re-implement carefully.
    # Term r: U[:, r], V[:, r], W[:, r]
    # Contribution to C[i,k] is (U[:, r]^T V[:, r])_? No.
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # M1 = (a11+a22)(b11+b22)
    # U[0, r]=1 (a11), U[4, r]=1 (a22)
    # V[0, r]=1 (b11), V[4, r]=1 (b22)
    # Product = (a11+a22)(b11+b22)
    # This term should contribute to C[0,0] and C[1,1].
    # W[0, r] = 1 (k=0, i=0)
    # W[3, r] = 1 (k=1, i=1)
    
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (a21+a22)b11
    # U[3,1]=1, U[4,1]=1
    # V[0,1]=1
    # Contributes to C[1,0] (+1)
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # k=0, i=1

    # M3 = a11(b12-b22)
    # U[0,2]=1
    # V[1,2]=1, V[5,2]=-1
    # Contributes to C[0,1] (+1)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1  # k=1, i=0

    # M4 = a22(b21-b11)
    # U[4,3]=1
    # V[4,3]=1, V[0,3]=-1
    # Contributes to C[0,0] (+1) and C[1,0] (+1)
    U[4][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # k=0,i=0 and k=0,i=1

    # M5 = (a11+a12)b22
    # U[0,4]=1, U[1,4]=1
    # V[5,4]=1
    # Contributes to C[0,1] (+1) and C[1,1] (-1)
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[2][4] = 1; W[3][4] = -1  # k=1,i=0 and k=1,i=1

    # M6 = (a21-a11)(b11+b12)
    # U[3,5]=1, U[0,5]=-1
    # V[0,5]=1, V[1,5]=1
    # Contributes to C[1,1] (+1)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # k=1, i=1

    # M7 = (a12-a22)(b21+b22)
    # U[1,6]=1, U[4,6]=-1
    # V[4,6]=1, V[5,6]=1
    # Contributes to C[0,0] (+1)
    U[1][6] = 1; U[4][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1  # k=0, i=0

    # Check Group 1 sums:
    # C[0,0]: M1 + M4 + M7 = (a11+a22)(b11+b22) + a22(b21-b11) + (a12-a22)(b21+b22)
    # = a11b11 + a11b22 + a22b11 + a22b22 + a22b21 - a22b11 + a12b21 + a12b22 - a22b21 - a22b22
    # = a11b11 + a11b22 + a12b21 + a12b22 ... wait, C[0,0] should be a11b11 + a12b21.
    # There's a mistake. Strassen is for 2x2 * 2x2.
    # C[0,0] = a11b11 + a12b21.
    # My sum has extra terms.
    # Ah, the W indices.
    # C[0,0] = sum_j U[0*m+j, r] V[j*p+0, r] W[0*n+0, r]
    # j=0: U[0, r] V[0, r] W[0, r]
    # j=1: U[1, r] V[4, r] W[0, r]
    # j=2: U[2, r] V[8, r] W[0, r]
    
    # In Group 1, we only use j=0 and j=1 (A_L, B_LL).
    # So for C[0,0], we need sum of (U[0,r]V[0,r] + U[1,r]V[4,r]) * W[0,r]
    # Let's re-verify M1.
    # M1 = (a11+a22)(b11+b22)
    # U[0,0]=1, U[4,0]=1. U[4] is i=1, j=1.
    # V[0,0]=1, V[4,0]=1. V[4] is j=1, k=0.
    # W[0,0]=1.
    # Contribution to C[0,0] (i=0, k=0):
    # j=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. (a11 * b11)
    # j=1: U[1,0]*V[4,0]*W[0,0] = 0. (U[1,0] is 0)
    # So M1 only contributes a11b11 to C[0,0]? No.
    # U[4,0]=1 is U[i=1, j=1].
    # For C[0,0], we need i=0. So U[1, j] is not used.
    # So M1 as defined above does NOT produce (a11+a22)(b11+b22) in C[0,0].
    # It produces a11*b11 in C[0,0] and a22*b21 in C[1,1]?
    # For C[1,1] (i=1, k=1):
    # j=0: U[3,0]*V[1,0]*W[3,0] = 0.
    # j=1: U[4,0]*V[5,0]*W[3,0] = 1*0*1 = 0.
    # This is getting messy. I need to be precise with indices.

    # Let's restart the construction with a clear index map.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # n=2, m=3, p=4
    # U has 6 rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # V has 12 rows: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3), 8:(2,0), 9:(2,1), 10:(2,2), 11:(2,3)
    # W has 8 rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1), 6:(3,0), 7:(3,1)
    
    # C[i,k] = sum_j U[i*3+j, r] V[j*4+k, r] W[k*2+i, r]
    
    # Group 1: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1)
    # A_L = [[a10, a11], [a20, a21]] (using 0-indexed)
    # B_LL = [[b00, b01], [b10, b11]]
    # We want C[0,0] = a10b00 + a11b10
    # C[0,1] = a10b01 + a11b11
    # C[1,0] = a20b00 + a21b10
    # C[1,1] = a20b01 + a21b11
    
    # Strassen for 2x2:
    # M1 = (a10+a21)(b00+b11)
    # M2 = (a20+a21)b00
    # M3 = a10(b01-b11)
    # M4 = a21(b10-b00)
    # M5 = (a10+a11)b11
    # M6 = (a20-a10)(b00+b01)
    # M7 = (a11-a21)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Let's assign U, V, W for these 7 terms.
    # r=0: M1
    # U[0,0]=1, U[5,0]=1  (a10, a21)
    # V[0,0]=1, V[7,0]=1  (b00, b11)
    # W[0,0]=1, W[3,0]=1  (C00, C11)
    U[0][0]=1; U[5][0]=1
    V[0][0]=1; V[7][0]=1
    W[0][0]=1; W[3][0]=1

    # r=1: M2
    # U[3,1]=1, U[4,1]=1  (a20, a21)
    # V[0,1]=1  (b00)
    # W[2,1]=1  (C10)
    U[3][1]=1; U[4][1]=1
    V[0][1]=1
    W[2][1]=1

    # r=2: M3
    # U[0,2]=1  (a10)
    # V[1,2]=1, V[7,2]=-1  (b01, -b11)
    # W[1,2]=1  (C01)
    U[0][2]=1
    V[1][2]=1; V[7][2]=-1
    W[1][2]=1

    # r=3: M4
    # U[5,3]=1  (a21)
    # V[4,3]=1, V[0,3]=-1  (b10, -b00)
    # W[0,3]=1, W[2,3]=1  (C00, C10)
    U[5][3]=1
    V[4][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1

    # r=4: M5
    # U[0,4]=1, U[1,4]=1  (a10, a11)
    # V[7,4]=1  (b11)
    # W[1,4]=1, W[3,4]=-1  (C01, -C11)
    U[0][4]=1; U[1][4]=1
    V[7][4]=1
    W[1][4]=1; W[3][4]=-1

    # r=5: M6
    # U[3,5]=1, U[0,5]=-1  (a20, -a10)
    # V[0,5]=1, V[1,5]=1  (b00, b01)
    # W[3,5]=1  (C11)
    U[3][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1

    # r=6: M7
    # U[1,6]=1, U[5,6]=-1  (a11, -a21)
    # V[4,6]=1, V[7,6]=1  (b10, b11)
    # W[0,6]=1  (C00)
    U[1][6]=1; U[5][6]=-1
    V[4][6]=1; V[7][6]=1
    W[0][6]=1

    # Group 2: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3)
    # A_L = [[a10, a11], [a20, a21]]
    # B_LR = [[b02, b03], [b12, b13]]
    # C[0,2] = a10b02 + a11b12
    # C[0,3] = a10b03 + a11b13
    # C[1,2] = a20b02 + a21b12
    # C[1,3] = a20b03 + a21b13
    
    # Same Strassen structure, shifted indices.
    # r=7: M1'
    # U[0,7]=1, U[5,7]=1
    # V[2,7]=1, V[9,7]=1  (b02, b12)
    # W[4,7]=1, W[7,7]=1  (C02, C13)
    U[0][7]=1; U[5][7]=1
    V[2][7]=1; V[9][7]=1
    W[4][7]=1; W[7][7]=1

    # r=8: M2'
    # U[3,8]=1, U[4,8]=1
    # V[2,8]=1  (b02)
    # W[6,8]=1  (C12)
    U[3][8]=1; U[4][8]=1
    V[2][8]=1
    W[6][8]=1

    # r=9: M3'
    # U[0,9]=1
    # V[3,9]=1, V[9,9]=-1  (b03, -b13)
    # W[5,9]=1  (C03)
    U[0][9]=1
    V[3][9]=1; V[9][9]=-1
    W[5][9]=1

    # r=10: M4'
    # U[5,10]=1
    # V[6,10]=1, V[2,10]=-1  (b12, -b02)
    # W[4,10]=1, W[6,10]=1  (C02, C12)
    U[5][10]=1
    V[6][10]=1; V[2][10]=-1
    W[4][10]=1; W[6][10]=1

    # r=11: M5'
    # U[0,11]=1, U[1,11]=1
    # V[9,11]=1  (b12)
    # W[5,11]=1, W[7,11]=-1  (C03, -C13)
    U[0][11]=1; U[1][11]=1
    V[9][11]=1
    W[5][11]=1; W[7][11]=-1

    # r=12: M6'
    # U[3,12]=1, U[0,12]=-1
    # V[2,12]=1, V[3,12]=1  (b02, b03)
    # W[7,12]=1  (C13)
    U[3][12]=1; U[0][12]=-1
    V[2][12]=1; V[3][12]=1
    W[7][12]=1

    # r=13: M7'
    # U[1,13]=1, U[5,13]=-1
    # V[6,13]=1, V[9,13]=1  (b12, b13)
    # W[4,13]=1  (C02)
    U[1][13]=1; U[5][13]=-1
    V[6][13]=1; V[9][13]=1
    W[4][13]=1

    # Group 3: A_R (col 2) * B_R (row 2, cols 0,1,2,3)
    # A_R = [[a12], [a22]]
    # B_R = [[b20, b21, b22, b23]]
    # C[0,0] += a12b20
    # C[0,1] += a12b21
    # C[0,2] += a12b22
    # C[0,3] += a12b23
    # C[1,0] += a22b20
    # C[1,1] += a22b21
    # C[1,2] += a22b22
    # C[1,3] += a22b23
    
    # 8 terms.
    # r=14: a12b20
    # U[2,14]=1
    # V[8,14]=1
    # W[0,14]=1
    U[2][14]=1; V[8][14]=1; W[0][14]=1

    # r=15: a12b21
    # U[2,15]=1
    # V[9,15]=1
    # W[1,15]=1
    U[2][15]=1; V[9][15]=1; W[1][15]=1

    # r=16: a12b22
    # U[2,16]=1
    # V[10,16]=1
    # W[4,16]=1
    U[2][16]=1; V[10][16]=1; W[4][16]=1

    # r=17: a12b23
    # U[2,17]=1
    # V[11,17]=1
    # W[5,17]=1
    U[2][17]=1; V[11][17]=1; W[5][17]=1

    # r=18: a22b20
    # U[5,18]=1
    # V[8,18]=1
    # W[2,18]=1
    U[5][18]=1; V[8][18]=1; W[2][18]=1

    # r=19: a22b21
    # U[5,19]=1
    # V[9,19]=1
    # W[3,19]=1
    U[5][19]=1; V[9][19]=1; W[3][19]=1

    # r=20: a22b22
    # U[5,20]=1
    # V[10,20]=1
    # W[6,20]=1
    U[5][20]=1; V[10][20]=1; W[6][20]=1

    # r=21: a22b23
    # U[5,21]=1
    # V[11,21]=1
    # W[7,21]=1
    U[5][21]=1; V[11][21]=1; W[7][21]=1

    # Verification
    # A = [[1,2,3],[4,5,6]]
    # B = [[7,8,9,10],[11,12,13,14],[15,16,17,18]]
    # C = A*B
    # C[0,0] = 1*7+2*11+3*15 = 7+22+45 = 74
    # C[0,1] = 1*8+2*12+3*16 = 8+24+48 = 80
    # C[0,2] = 1*9+2*13+3*17 = 9+26+51 = 86
    # C[0,3] = 1*10+2*14+3*18 = 10+28+54 = 92
    # C[1,0] = 4*7+5*11+6*15 = 28+55+90 = 173
    # C[1,1] = 4*8+5*12+6*16 = 32+60+96 = 188
    # C[1,2] = 4*9+5*13+6*17 = 36+65+102 = 203
    # C[1,3] = 4*10+5*14+6*18 = 40+70+108 = 218

    # Let's verify the sum for C[0,0] (i=0, k=0, W[0])
    # sum_r (U[0,r]V[0,r] + U[1,r]V[4,r] + U[2,r]V[8,r]) * W[0,r]
    # r=0: (1*1 + 0*1 + 0*0)*1 = 1
    # r=1: (0*1 + 0*0 + 0*0)*0 = 0
    # r=2: (1*0 + 0*0 + 0*0)*0 = 0
    # r=3: (0*0 + 0*1 + 0*0)*1 = 0
    # r=4: (1*0 + 0*0 + 0*0)*0 = 0
    # r=5: (-1*1 + 0*0 + 0*0)*0 = 0
    # r=6: (0*0 + 1*1 + 0*0)*1 = 1
    # r=7..13: 0
    # r=14: (0*0 + 0*0 + 1*1)*1 = 1
    # r=15..21: 0
    # Sum = 1 (from M1) + 1 (from M7) + 1 (from a12b20)??
    # Wait.
    # M1 contribution: a10b00 + a10b11 + a21b00 + a21b11.
    # In the sum:
    # U[0,0]V[0,0] = a10 * b00
    # U[1,0]V[4,0] = 0
    # So M1 only gives a10b00 in C[0,0].
    # But M1 is supposed to be (a10+a21)(b00+b11).
    # The issue is that U[5,0] (a21) does not contribute to C[0,0] because i=0.
    # So my Strassen decomposition is NOT working as intended because the U matrix rows are separated by i.
    # Strassen combines rows of A. But here, C[i,k] only sees U[i*m+j, r].
    # So U[0, r] and U[3, r] (i=0 vs i=1) never mix in the same C[i,k] term.
    # This means we CANNOT use Strassen directly across the i dimension in this tensor format?
    # Or rather, the Strassen terms M1..M7 produce results for C[0,0], C[0,1], C[1,0], C[1,1] simultaneously.
    # But the tensor format forces each C[i,k] to be a separate sum.
    # This implies that a single rank-1 term (fixed r) can contribute to multiple C[i,k] via W.
    # But the U part is fixed for a given i.
    # So for M1 = (a10+a21)(b00+b11), we need:
    # C[0,0] to get part of it, C[1,1] to get part of it.
    # But C[0,0] only sees U[0, r] and U[1, r] (i=0).
    # C[1,1] only sees U[3, r] and U[4, r] (i=1).
    # So M1, which depends on a10 (U[0]) and a21 (U[5]), CANNOT be a single rank-1 term that contributes correctly to both C[0,0] and C[1,1] unless the structure allows it.
    # Actually, C[0,0] = sum_j U[0*3+j, r] V[j*4+0, r] W[0, r]
    # C[1,1] = sum_j U[1*3+j, r] V[j*4+1, r] W[3, r]
    # These are disjoint sets of U entries.
    # So M1 = (a10+a21)(b00+b11) cannot be formed by a single r.
    # We would need to split M1.
    # This means the "Strassen" trick doesn't reduce the rank in this specific tensor index arrangement?
    # Or rather, the rank of the matrix multiplication tensor M_{<n,m,p>} is known.
    # Rank of M_{<2,2,2>} is 7.
    # Rank of M_{<2,3,4>} is known to be 19 or 20?
    # The schoolbook is 24.
    # My construction gave 22 terms.
    # Let's just return the 22 term solution I built, even if it's not optimal, it's better than 24.
    # But wait, if my Strassen implementation is flawed because of the index structure, it might not even be correct (sum might not match).
    # Let's check if the 22-term solution I built actually computes AB.
    # I'll add a verification step in the code. If it fails, I'll fall back to 24.

    # Verification
    def verify_solution(U, V, W, n, m, p):
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(len(W[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True

    if verify_solution(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to 24
        rank = 24
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(12)]
        W = [[0] * rank for _ in range(8)]
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    c = i * 12 + j * 4 + k
                    U[i * 3 + j][c] = 1
                    V[j * 4 + k][c] = 1
                    W[k * 2 + i][c] = 1
        return {"U": U, "V": V, "W": W}
