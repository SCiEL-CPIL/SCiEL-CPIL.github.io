"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Winograd 2x2x3 decomposition:
    # For C = A*B where A is 2x2, B is 2x3, C is 2x3:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    #
    # 6 multiplications:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b12
    # M6 = (a10-a00)(b01+b12)
    #
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M5
    # C02 = M6
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3
    # C12 = M5 - M6
    
    rank = 36  # 2 A-blocks * 3 B-blocks * 6 per 2x2x3
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    def add_winograd_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U (row indices i*m+j)
        b_idx: [b00, b01, b02, b10, b11, b12] indices into V (row indices j*p+k)
        w_idx: [w00, w01, w02, w10, w11, w12] indices into W (row indices k*n+i)
              w00=C[0,0], w01=C[0,1], w02=C[0,2], w10=C[1,0], w11=C[1,1], w12=C[1,2]
        """
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[4]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[4]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[3]][c+1] += 1; W[w_idx[4]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[4]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[4]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[3]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[3]][c+3] += 1
        # M5 = (a00+a01)b12
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[5]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1; W[w_idx[5]][c+4] += 1
        # M6 = (a10-a00)(b01+b12)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[1]][c+5] += 1; V[b_idx[5]][c+5] += 1
        W[w_idx[0]][c+5] += 1; W[w_idx[2]][c+5] += 1; W[w_idx[5]][c+5] -= 1
    
    # A is 2x4: A[i,j] -> U[i*m+j]
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]   # A[0,0], A[0,1], A[1,0], A[1,1]
    a_R = [2, 3, 6, 7]   # A[0,2], A[0,3], A[1,2], A[1,3]
    
    # B is 4x6: B[j,k] -> V[j*p+k]
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4, B[0,5]=5
    # B[1,0]=6, B[1,1]=7, B[1,2]=8, B[1,3]=9, B[1,4]=10, B[1,5]=11
    # B[2,0]=12, B[2,1]=13, B[2,2]=14, B[2,3]=15, B[2,4]=16, B[2,5]=17
    # B[3,0]=18, B[3,1]=19, B[3,2]=20, B[3,3]=21, B[3,4]=22, B[3,5]=23
    
    # B block 0: rows 0,1; cols 0,1,2
    # b00=B[0,0]=0, b01=B[0,1]=1, b02=B[0,2]=2
    # b10=B[1,0]=6, b11=B[1,1]=7, b12=B[1,2]=8
    b_0 = [0, 1, 2, 6, 7, 8]
    
    # B block 1: rows 0,1; cols 3,4,5
    # b00=B[0,3]=3, b01=B[0,4]=4, b02=B[0,5]=5
    # b10=B[1,3]=9, b11=B[1,4]=10, b12=B[1,5]=11
    b_1 = [3, 4, 5, 9, 10, 11]
    
    # B block 2: rows 2,3; cols 0,1,2
    # b00=B[2,0]=12, b01=B[2,1]=13, b02=B[2,2]=14
    # b10=B[3,0]=18, b11=B[3,1]=19, b12=B[3,2]=20
    b_3 = [12, 13, 14, 18, 19, 20]
    
    # B block 3: rows 2,3; cols 3,4,5
    # b00=B[2,3]=15, b01=B[2,4]=16, b02=B[2,5]=17
    # b10=B[3,3]=21, b11=B[3,4]=22, b12=B[3,5]=23
    b_4 = [15, 16, 17, 21, 22, 23]
    
    # W indices: W[k*n+i] where n=2
    # C[0,0]->W[0*2+0]=W[0], C[0,1]->W[1*2+0]=W[2], C[0,2]->W[2*2+0]=W[4]
    # C[1,0]->W[0*2+1]=W[1], C[1,1]->W[1*2+1]=W[3], C[1,2]->W[2*2+1]=W[5]
    # w_idx = [C00, C01, C02, C10, C11, C12]
    
    # For B blocks 0,1 (cols 0-5):
    # C[0,0]->W[0], C[0,1]->W[2], C[0,2]->W[4]
    # C[1,0]->W[1], C[1,1]->W[3], C[1,2]->W[5]
    w_cols_0_2 = [0, 2, 4, 1, 3, 5]
    # For B blocks 3,4 (cols 3-5):
    # C[0,3]->W[3*2+0]=W[6], C[0,4]->W[4*2+0]=W[8], C[0,5]->W[5*2+0]=W[10]
    # C[1,3]->W[3*2+1]=W[7], C[1,4]->W[4*2+1]=W[9], C[1,5]->W[5*2+1]=W[11]
    w_cols_3_5 = [6, 8, 10, 7, 9, 11]
    
    # 6 blocks total: 2 A-blocks x 3 B-row-pairs x (but B has 4 rows, 6 cols)
    # Actually: A has 2 row-blocks (left cols 0-1, right cols 2-3)
    # B has 4 rows, 6 cols. We need to pair B rows.
    # B rows 0,1 with cols 0,1,2 -> block 0
    # B rows 0,1 with cols 3,4,5 -> block 1
    # B rows 2,3 with cols 0,1,2 -> block 2
    # B rows 2,3 with cols 3,4,5 -> block 3
    # That's 4 B-blocks, but we only have 36 = 2*3*6 slots.
    # Wait: 2 A-blocks * 4 B-blocks * 6 = 48. That's too many.
    # 
    # Let me reconsider. We have A (2x4) * B (4x6) = C (2x6).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split A into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    # Split B into B_T (2x6, rows 0,1) and B_B (2x6, rows 2,3)
    # 
    # C = A_L * B_T + A_R * B_B
    # 
    # For each product A_L * B_T (2x2 * 2x6), we split B_T into 3 blocks of 2x2:
    #   B_T,0 (cols 0,1), B_T,1 (cols 2,3), B_T,2 (cols 4,5)
    # Each 2x2*2x2 uses 7 Strassen mults -> 7*3 = 21 per A-block
    # Total: 2*21 = 42 (the Strassen approach)
    #
    # For Winograd 2x2x3: 2x2 * 2x3 uses 6 mults.
    # Split B_T (2x6) into 2 blocks of 2x3:
    #   B_T,0 (cols 0,1,2), B_T,1 (cols 3,4,5)
    # Each 2x2*2x3 uses 6 mults -> 6*2 = 12 per A-block
    # Total: 2*12 = 24
    #
    # Similarly for A_R * B_B: 12 mults
    # Total: 24 + 24 = 48. That's not better!
    #
    # Wait, I need to re-examine. The B matrix is 4x6.
    # If I split B into B_T (rows 0,1) and B_B (rows 2,3), each is 2x6.
    # A_L is 2x2, A_R is 2x2.
    # A_L * B_T: 2x2 * 2x6 -> split into two 2x2*2x3 -> 2*6 = 12
    # A_R * B_B: 2x2 * 2x6 -> split into two 2x2*2x3 -> 2*6 = 12
    # Total: 24.
    #
    # But wait, that's only if B_T and B_B are independent. Let me verify:
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    #        = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #        = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    #        = (A_L[i,:] * B_T[:,k]) + (A_R[i,:] * B_B[:,k])
    #
    # So C = A_L * B_T + A_R * B_B where:
    # A_L = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] (2x2)
    # A_R = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]] (2x2)
    # B_T = [[B[0,0],...,B[0,5]], [B[1,0],...,B[1,5]]] (2x6)
    # B_B = [[B[2,0],...,B[2,5]], [B[3,0],...,B[3,5]]] (2x6)
    #
    # For A_L * B_T: split B_T into B_T0 (cols 0,1,2) and B_T1 (cols 3,4,5)
    # A_L * B_T0: 2x2 * 2x3, rank 6
    # A_L * B_T1: 2x2 * 2x3, rank 6
    # Subtotal: 12
    #
    # For A_R * B_B: split B_B into B_B0 (cols 0,1,2) and B_B1 (cols 3,4,5)
    # A_R * B_B0: 2x2 * 2x3, rank 6
    # A_R * B_B1: 2x2 * 2x3, rank 6
    # Subtotal: 12
    #
    # Total: 24!
    
    # Let me redo with 4 B-blocks:
    # Block 0: A_L * B_T0 (rows 0,1 of B; cols 0,1,2)
    # Block 1: A_L * B_T1 (rows 0,1 of B; cols 3,4,5)
    # Block 2: A_R * B_B0 (rows 2,3 of B; cols 0,1,2)
    # Block 3: A_R * B_B1 (rows 2,3 of B; cols 3,4,5)
    
    # But wait, 4 blocks * 6 = 24, not 36. Let me fix the rank.
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # B_T0: rows 0,1; cols 0,1,2
    # b00=B[0,0]=0, b01=B[0,1]=1, b02=B[0,2]=2
    # b10=B[1,0]=6, b11=B[1,1]=7, b12=B[1,2]=8
    b_T0 = [0, 1, 2, 6, 7, 8]
    # W for cols 0,1,2: C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5]
    w_012 = [0, 2, 4, 1, 3, 5]
    add_winograd_block(a_L, b_T0, w_012, 0)
    
    # B_T1: rows 0,1; cols 3,4,5
    # b00=B[0,3]=3, b01=B[0,4]=4, b02=B[0,5]=5
    # b10=B[1,3]=9, b11=B[1,4]=10, b12=B[1,5]=11
    b_T1 = [3, 4, 5, 9, 10, 11]
    # W for cols 3,4,5: C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10], C[1,3]=W[7], C[1,4]=W[9], C[1,5]=W[11]
    w_345 = [6, 8, 10, 7, 9, 11]
    add_winograd_block(a_L, b_T1, w_345, 6)
    
    # B_B0: rows 2,3; cols 0,1,2
    # b00=B[2,0]=12, b01=B[2,1]=13, b02=B[2,2]=14
    # b10=B[3,0]=18, b11=B[3,1]=19, b12=B[3,2]=20
    b_B0 = [12, 13, 14, 18, 19, 20]
    add_winograd_block(a_R, b_B0, w_012, 12)
    
    # B_B1: rows 2,3; cols 3,4,5
    # b00=B[2,3]=15, b01=B[2,4]=16, b02=B[2,5]=17
    # b10=B[3,3]=21, b11=B[3,4]=22, b12=B[3,5]=23
    b_B1 = [15, 16, 17, 21, 22, 23]
    add_winograd_block(a_R, b_B1, w_345, 18)
    
    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat, rk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rk):
                        total += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
