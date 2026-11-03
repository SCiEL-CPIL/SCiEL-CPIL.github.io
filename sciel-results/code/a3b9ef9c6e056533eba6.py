"""Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7 instead of 8.
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11-A22)(B22-B11)
    # M2 = (A11+A22)B21
    # M3 = A12(B11-B22)
    # M4 = (A21-A11)B11
    # M5 = (A11+A12)B22
    # M6 = (A21+A22)(B11-B12)
    # M7 = (A11-A21)(B12+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A is 2x2: A[i,j] for i in {0,1}, j in {0,1}
    # B is 2x2: B[j,k] for j in {0,1}, k in {0,1}
    # C is 2x2: C[i,k] for i in {0,1}, k in {0,1}
    #
    # U maps (i,j) -> products, V maps (j,k) -> products, W maps (k,i) -> products
    # Indexing: A index = i*2+j, B index = j*2+k, C index = k*2+i (trace convention W[k*n+i])
    #
    # For 2x2: A index = i*2+j, B index = j*2+k, W index = k*2+i
    
    # Products:
    # M1: (A00-A11)(B11-B00)
    # M2: (A00+A11)B10
    # M3: A01(B00-B11)
    # M4: (A10-A00)B00
    # M5: (A00+A01)B11
    # M6: (A10+A11)(B00-B01)
    # M7: (A00-A10)(B01+B11)
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00-A11)(B11-B00)
    # A00: U[0,0]=1, A11: U[3,0]=-1
    # B11: V[3,0]=1, B00: V[0,0]=-1
    # C11 = ... + M1, C22 = ... + M1
    # C11: i=0,k=0 -> W[0*2+0]=W[0], C22: i=1,k=1 -> W[1*2+1]=W[3]
    U[0][0] = 1; U[3][0] = -1
    V[3][0] = 1; V[0][0] = -1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A00+A11)B10
    # A00: U[0,1]=1, A11: U[3,1]=1
    # B10: V[2,1]=1
    # C21: i=1,k=0 -> W[0*2+1]=W[1]
    U[0][1] = 1; U[3][1] = 1
    V[2][1] = 1
    W[1][1] = 1
    
    # M3: A01(B00-B11)
    # A01: U[1,2]=1
    # B00: V[0,2]=1, B11: V[3,2]=-1
    # C12: i=0,k=1 -> W[1*2+0]=W[2], C22: i=1,k=1 -> W[3]
    U[1][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: (A10-A00)B00
    # A10: U[2,3]=1, A00: U[0,3]=-1
    # B00: V[0,3]=1
    # C11: i=0,k=0 -> W[0], C21: i=1,k=0 -> W[1]
    U[2][3] = 1; U[0][3] = -1
    V[0][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C11: i=0,k=0 -> W[0] (with -1), C12: i=0,k=1 -> W[2]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10+A11)(B00-B01)
    # A10: U[2,5]=1, A11: U[3,5]=1
    # B00: V[0,5]=1, B01: V[1,5]=-1
    # C22: i=1,k=1 -> W[3]
    U[2][5] = 1; U[3][5] = 1
    V[0][5] = 1; V[1][5] = -1
    W[3][5] = 1
    
    # M7: (A00-A10)(B01+B11)
    # A00: U[0,6]=1, A10: U[2,6]=-1
    # B01: V[1,6]=1, B11: V[3,6]=1
    # C11: i=0,k=0 -> W[0]
    U[0][6] = 1; U[2][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    
    Strategy: Split the 3x4 matrix into a 2x4 block and a 1x4 block.
    - The 2x4 x 4x8 part: split 4 columns into 2x2 blocks, use Strassen for each 2x2x2 block.
      2x4 x 4x8 = 2x(2x2x2) x 2x(2x2x2) -> 4 subproblems of 2x2x2, each rank 7 = 28
    - The 1x4 x 4x8 part: schoolbook, rank 4*8 = 32
    - Total: 28 + 32 = 60
    
    Wait, let me reconsider the decomposition more carefully.
    
    A is 3x4, B is 4x8, C is 3x8.
    
    Split A into A_top (2x4) and A_bottom (1x4).
    Split B into B_left (4x4) and B_right (4x4).
    
    C_top_left (2x4) = A_top * B_left
    C_top_right (2x4) = A_top * B_right
    C_bottom_left (1x4) = A_bottom * B_left
    C_bottom_right (1x4) = A_bottom * B_right
    
    For C_top_left (2x4 x 4x4):
      Split B_left into 2x2 blocks: B_left[:,0:2] and B_left[:,2:4]
      C_top_left[:,0:2] = A_top * B_left[:,0:2] (2x4 x 4x2)
      C_top_left[:,2:4] = A_top * B_left[:,2:4] (2x4 x 4x2)
      
    For 2x4 x 4x2: Split A_top into A_top[:,0:2] and A_top[:,2:4]
      C_top_left[:,0:2] = A_top[:,0:2] * B_left[:,0:2] + A_top[:,2:4] * B_left[:,2:4]
      Each term is 2x2 x 2x2, use Strassen rank 7.
      So 2x4 x 4x2 has rank 2*7 = 14.
      
    Similarly for all four quadrants of C_top (2x8):
      C_top (2x8) = A_top(2x4) * B(4x8)
      Split into 4 subproblems of 2x2 x 2x2:
        A_top[:,0:2] * B[0:2,0:2] -> C_top[0:2,0:2]
        A_top[:,0:2] * B[0:2,2:4] -> C_top[0:2,2:4]
        A_top[:,0:2] * B[0:2,4:6] -> C_top[0:2,4:6]
        A_top[:,0:2] * B[0:2,6:8] -> C_top[0:2,6:8]
        A_top[:,2:4] * B[2:4,0:2] -> C_top[0:2,0:2] (add)
        A_top[:,2:4] * B[2:4,2:4] -> C_top[0:2,2:4] (add)
        A_top[:,2:4] * B[2:4,4:6] -> C_top[0:2,4:6] (add)
        A_top[:,2:4] * B[2:4,6:8] -> C_top[0:2,6:8] (add)
      
      That's 8 subproblems of 2x2x2, each rank 7 = 56.
      
    For C_bottom (1x8) = A_bottom(1x4) * B(4x8):
      Schoolbook: 1*4*8 = 32.
      
    Total rank = 56 + 32 = 88.
    
    But wait, we can do better. Let me think about using Strassen more cleverly.
    
    Actually, for the 1x4 x 4x8 part, we can't use Strassen directly since it's 1x4x8.
    But we could potentially use a different approach.
    
    Let me just implement the 88-rank decomposition first, which is already better than 96.
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by combining Strassen for 2x2x2 blocks
    # and schoolbook for the remaining parts.
    
    # Strategy: 
    # Rows 0-1 of A (2 rows) with all 4 cols and 8 cols of B: 8 Strassen 2x2x2 blocks, rank 56
    # Row 2 of A (1 row) with all 4 cols and 8 cols of B: schoolbook, rank 32
    # Total: 88
    
    # Let's build this explicitly.
    
    # For the Strassen part (rows 0-1):
    # A[0:2, 0:4] * B[0:4, 0:8] -> C[0:2, 0:8]
    # Split into 8 blocks: A[0:2, c0:c0+2] * B[c0:c0+2, c1:c1+2] -> C[0:2, c1:c1+2]
    # where c0 in {0, 2} and c1 in {0, 2, 4, 6}
    
    # For the schoolbook part (row 2):
    # A[2:3, 0:4] * B[0:4, 0:8] -> C[2:3, 0:8]
    # Each (i,j,k) triple contributes 1 rank
    
    # Let's construct the full U, V, W matrices.
    
    # First, let's get the Strassen base decomposition
    sU, sV, sW = _strassen_2x2x2()
    
    # Total rank: 8*7 + 1*4*8 = 56 + 32 = 88
    total_rank = 88
    
    U = [[0] * total_rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 8*3 = 24 rows
    
    col = 0
    
    # Strassen blocks for rows 0-1 of A
    for a_row_block in range(2):  # A rows 0-1
        for a_col_block in range(2):  # A cols 0-1, 2-3
            for b_col_block in range(4):  # B cols 0-1, 2-3, 4-5, 6-7
                # A subblock: rows a_row_block*2 to a_row_block*2+1, cols a_col_block*2 to a_col_block*2+1
                # B subblock: rows a_col_block*2 to a_col_block*2+1, cols b_col_block*2 to b_col_block*2+1
                # C subblock: rows a_row_block*2 to a_row_block*2+1, cols b_col_block*2 to b_col_block*2+1
                
                a_r0 = a_row_block * 2
                a_c0 = a_col_block * 2
                b_c0 = b_col_block * 2
                
                # For each of the 7 Strassen products:
                for s in range(7):
                    # U entries: A indices (i, j) where i in {a_r0, a_r0+1}, j in {a_c0, a_c0+1}
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = a_r0 + i_local
                            j_global = a_c0 + j_local
                            U[i_global * m + j_global][col] += sU[i_local * 2 + j_local][s]
                    
                    # V entries: B indices (j, k) where j in {a_c0, a_c0+1}, k in {b_c0, b_c0+1}
                    for j_local in range(2):
                        for k_local in range(2):
                            j_global = a_c0 + j_local
                            k_global = b_c0 + k_local
                            V[j_global * p + k_global][col] += sV[j_local * 2 + k_local][s]
                    
                    # W entries: C indices (k, i) where i in {a_r0, a_r0+1}, k in {b_c0, b_c0+1}
                    # W index = k*n + i
                    for k_local in range(2):
                        for i_local in range(2):
                            i_global = a_r0 + i_local
                            k_global = b_c0 + k_local
                            W[k_global * n + i_global][col] += sW[k_local * 2 + i_local][s]
                    
                    col += 1
    
    # Schoolbook for row 2 of A
    for j in range(m):  # 4 columns
        for k in range(p):  # 8 rows of B
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
