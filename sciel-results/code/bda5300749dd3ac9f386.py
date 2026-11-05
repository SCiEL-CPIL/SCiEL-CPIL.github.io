"""Matrix multiplication 6x6 using Strassen-based block decomposition."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # Use 3x3 block structure, each block is 2x2
    block_size = 2
    num_blocks = 3  # 6 / 2 = 3
    
    # Strassen's 2x2 multiplication: 7 scalar multiplications
    # For A (2x2), B (2x2), C (2x2):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U_2[r, c]: row is (i_local * 2 + j_local) in A, col is multiplication index c (0..6)
    # V_2[r, c]: row is (i_local * 2 + j_local) in B, col is multiplication index c (0..6)
    # W_2[r, c]: row is (j_local * 2 + i_local) in C (trace convention: col*2+row), col is c
    
    U_2 = [[0]*7 for _ in range(4)]
    V_2 = [[0]*7 for _ in range(4)]
    W_2 = [[0]*7 for _ in range(4)]
    
    # Row indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A00=row0, A01=row1, A10=row2, A11=row3
    # B00=row0, B01=row1, B10=row2, B11=row3
    
    # M1 = (A00+A11)(B00+B11)
    U_2[0][0] = 1; U_2[3][0] = 1
    V_2[0][0] = 1; V_2[3][0] = 1
    # C00 += M1, C11 += M1
    W_2[0][0] = 1; W_2[3][0] = 1  # C00: row0, C11: row3 (j=1,i=1 -> 1*2+1=3)
    
    # M2 = (A10+A11)B00
    U_2[2][1] = 1; U_2[3][1] = 1
    V_2[0][1] = 1
    # C10 += M2, C11 -= M2
    W_2[2][1] = 1; W_2[3][1] = -1  # C10: j=0,i=1 -> 0*2+1=1... wait
    
    # Let me recheck: C[i,j] where i is row of C, j is col of C
    # W row = j*2 + i (trace convention for 2x2: W[k*n+i] where n=2, so row = j*2+i)
    # C00: i=0,j=0 -> row=0
    # C01: i=0,j=1 -> row=1
    # C10: i=1,j=0 -> row=2
    # C11: i=1,j=1 -> row=3
    
    # M2: C10 += M2, C11 -= M2
    W_2[2][1] = 1; W_2[3][1] = -1
    
    # M3 = A00(B01-B11)
    U_2[0][2] = 1
    V_2[1][2] = 1; V_2[3][2] = -1
    # C01 += M3, C11 += M3
    W_2[1][2] = 1; W_2[3][2] = 1
    
    # M4 = A11(B10-B00)
    U_2[3][3] = 1
    V_2[2][3] = 1; V_2[0][3] = -1
    # C00 += M4, C10 += M4
    W_2[0][3] = 1; W_2[2][3] = 1
    
    # M5 = (A00+A01)B11
    U_2[0][4] = 1; U_2[1][4] = 1
    V_2[3][4] = 1
    # C00 -= M5, C01 += M5
    W_2[0][4] = -1; W_2[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_2[2][5] = 1; U_2[0][5] = -1
    V_2[0][5] = 1; V_2[1][5] = 1
    # C11 += M6
    W_2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_2[1][6] = 1; U_2[3][6] = -1
    V_2[2][6] = 1; V_2[3][6] = 1
    # C00 += M7
    W_2[0][6] = 1
    
    # Now build the full 6x6 decomposition
    # Block (I,J,K) where I=block row of A, J=block col of B, K=block row of C
    # Actually, for C = A*B:
    # C[I,J] = sum over K of A[I,K] * B[K,J]
    # So blocks: I=0,1,2 (row of A), K=0,1,2 (col of A / row of B), J=0,1,2 (col of B)
    # C block at (I,J) receives contribution from A block (I,K) * B block (K,J)
    
    rank = num_blocks * num_blocks * 7  # 27 * 7 = 189
    U = [[0]*rank for _ in range(n*m)]  # 36 x 189
    V = [[0]*rank for _ in range(m*p)]  # 36 x 189
    W = [[0]*rank for _ in range(p*n)]  # 36 x 189
    
    for I in range(num_blocks):
        for K in range(num_blocks):
            for J in range(num_blocks):
                base_col = (I * num_blocks + K) * num_blocks + J  # 0..26
                for c_inner in range(7):
                    col = base_col * 7 + c_inner
                    
                    # U: A block (I, K), local row r (0..3)
                    for r in range(4):
                        val = U_2[r][c_inner]
                        if val != 0:
                            # Global row in A: row = (I*block_size + r//2) * m + (K*block_size + r%2)
                            ai = I * block_size + r // 2
                            aj = K * block_size + r % 2
                            global_row = ai * m + aj
                            U[global_row][col] += val
                    
                    # V: B block (K, J), local row r (0..3)
                    for r in range(4):
                        val = V_2[r][c_inner]
                        if val != 0:
                            # Global row in B: row = (K*block_size + r//2) * p + (J*block_size + r%2)
                            bi = K * block_size + r // 2
                            bj = J * block_size + r % 2
                            global_row = bi * p + bj
                            V[global_row][col] += val
                    
                    # W: C block (I, J), local row r (0..3)
                    for r in range(4):
                        val = W_2[r][c_inner]
                        if val != 0:
                            # Global row in C: C[I*bs + r//2, J*bs + r%2]
                            # W row = j*6 + i where i = I*bs + r//2, j = J*bs + r%2
                            ci = I * block_size + r // 2
                            cj = J * block_size + r % 2
                            global_row = cj * n + ci
                            W[global_row][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
