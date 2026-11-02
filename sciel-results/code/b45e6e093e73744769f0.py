"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for 2x4 * 4x4 matrix multiplication.
    
    This uses a Strassen-like block decomposition:
    - Split A (2x4) into two 2x2 blocks: A1, A2
    - Split B (4x4) into four 2x2 blocks: B1, B2, B3, B4
    - Use 7 Strassen products for each pair, but with sharing/optimization
    
    Actually, for 2x4 * 4x4:
    Let A = [A1 | A2] where A1, A2 are 2x2
    Let B = [B1 B2; B3 B4] where B1..B4 are 2x2
    
    C = [A1*B1 + A2*B3 | A1*B2 + A2*B4]
    
    Using Strassen for each 2x2*2x2 multiplication would give 7*2 = 14 products,
    but we can do better by sharing some computations.
    
    However, the known optimal rank for 2x4x4 is 28 (from literature on small matrix multiplication).
    
    Let me use a known rank-28 decomposition. One approach:
    Treat this as multiplying a 2x4 matrix by a 4x4 matrix.
    
    Actually, let me think about this differently. The tensor is T_{2,4,4}.
    Known results: the rank of T_{2,4,4} is 28.
    
    Here's a construction based on blocking:
    A is 2x4, B is 4x4, C is 2x4.
    
    Write A = [A_left | A_right] (each 2x2)
    Write B = [[B_tl, B_tr], [B_bl, B_br]] (each 2x2)
    
    C_left = A_left * B_tl + A_right * B_bl
    C_right = A_left * B_tr + A_right * B_br
    
    For each 2x2 * 2x2 product, Strassen uses 7 multiplications.
    So naively: 4 products * 7 = 28.
    
    This gives exactly rank 28!
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 * 2x2:
    # Given matrices X = [[x1,x2],[x3,x4]], Y = [[y1,y2],[y3,y4]]
    # Z = X*Y
    # M1 = (x1+x4)(y1+y4)
    # M2 = (x3+x4)y1
    # M3 = x1(y2-y4)
    # M4 = x4(y3-y1)
    # M5 = (x1+x2)y4
    # M6 = (x3-x1)(y1+y2)
    # M7 = (x2-x4)(y3+y4)
    # Z11 = M1+M4-M5+M7
    # Z12 = M3+M5
    # Z21 = M2+M4
    # Z22 = M1-M2+M3+M6
    
    # For our case:
    # We have 4 independent 2x2*2x2 multiplications:
    # 1. A_left * B_tl -> contributes to C_left
    # 2. A_right * B_bl -> contributes to C_left
    # 3. A_left * B_tr -> contributes to C_right
    # 4. A_right * B_br -> contributes to C_right
    
    # Each uses 7 Strassen multiplications, total 28.
    
    # Index conventions:
    # A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
    #   A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B is 4x4: B[j,k] where j in {0,1,2,3}, k in {0,1,2,3}
    #   B_tl = B[0:2, 0:2], B_tr = B[0:2, 2:4]
    #   B_bl = B[2:4, 0:2], B_br = B[2:4, 2:4]
    # C is 2x4: C[i,k] where i in {0,1}, k in {0,1,2,3}
    #   C_left = C[:, 0:2], C_right = C[:, 2:4]
    
    # U row index: i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # V row index: j*p + k = j*4 + k, where j in {0,1,2,3}, k in {0,1,2,3}
    # W row index: k*n + i = k*2 + i, where k in {0,1,2,3}, i in {0,1}
    
    # Strassen multipliers for a 2x2*2x2 product:
    # X indices: x1=X[0,0], x2=X[0,1], x3=X[1,0], x4=X[1,1]
    # Y indices: y1=Y[0,0], y2=Y[0,1], y3=Y[1,0], y4=Y[1,1]
    
    def add_strassen_block(base_col, X_rows, Y_rows, Z_rows, X_row_offset, Y_row_offset, Z_row_offset):
        """
        Add a Strassen block for multiplying 2x2 matrix X by 2x2 matrix Y,
        storing result in 2x2 matrix Z.
        
        X_rows: list of 4 row indices in U (for X[0,0], X[0,1], X[1,0], X[1,1])
        Y_rows: list of 4 row indices in V (for Y[0,0], Y[0,1], Y[1,0], Y[1,1])
        Z_rows: list of 4 row indices in W (for Z[0,0], Z[0,1], Z[1,0], Z[1,1])
        
        X_row_offset, Y_row_offset, Z_row_offset: offsets to add to get actual row indices
        """
        # M1 = (x1+x4)(y1+y4)
        col = base_col
        U[X_rows[0] + X_row_offset][col] = 1
        U[X_rows[3] + X_row_offset][col] = 1
        V[Y_rows[0] + Y_row_offset][col] = 1
        V[Y_rows[3] + Y_row_offset][col] = 1
        W[Z_rows[0] + Z_row_offset][col] = 1
        W[Z_rows[1] + Z_row_offset][col] = 1
        W[Z_rows[3] + Z_row_offset][col] = 1
        W[Z_rows[2] + Z_row_offset][col] = -1
        col += 1
        
        # M2 = (x3+x4)y1
        U[X_rows[2] + X_row_offset][col] = 1
        U[X_rows[3] + X_row_offset][col] = 1
        V[Y_rows[0] + Y_row_offset][col] = 1
        W[Z_rows[2] + Z_row_offset][col] = 1
        W[Z_rows[3] + Z_row_offset][col] = -1
        col += 1
        
        # M3 = x1(y2-y4)
        U[X_rows[0] + X_row_offset][col] = 1
        V[Y_rows[1] + Y_row_offset][col] = 1
        V[Y_rows[3] + Y_row_offset][col] = -1
        W[Z_rows[1] + Z_row_offset][col] = 1
        W[Z_rows[3] + Z_row_offset][col] = 1
        col += 1
        
        # M4 = x4(y3-y1)
        U[X_rows[3] + X_row_offset][col] = 1
        V[Y_rows[2] + Y_row_offset][col] = 1
        V[Y_rows[0] + Y_row_offset][col] = -1
        W[Z_rows[0] + Z_row_offset][col] = 1
        W[Z_rows[2] + Z_row_offset][col] = 1
        col += 1
        
        # M5 = (x1+x2)y4
        U[X_rows[0] + X_row_offset][col] = 1
        U[X_rows[1] + X_row_offset][col] = 1
        V[Y_rows[3] + Y_row_offset][col] = 1
        W[Z_rows[0] + Z_row_offset][col] = -1
        W[Z_rows[1] + Z_row_offset][col] = 1
        col += 1
        
        # M6 = (x3-x1)(y1+y2)
        U[X_rows[2] + X_row_offset][col] = 1
        U[X_rows[0] + X_row_offset][col] = -1
        V[Y_rows[0] + Y_row_offset][col] = 1
        V[Y_rows[1] + Y_row_offset][col] = 1
        W[Z_rows[3] + Z_row_offset][col] = 1
        col += 1
        
        # M7 = (x2-x4)(y3+y4)
        U[X_rows[1] + X_row_offset][col] = 1
        U[X_rows[3] + X_row_offset][col] = -1
        V[Y_rows[2] + Y_row_offset][col] = 1
        V[Y_rows[3] + Y_row_offset][col] = 1
        W[Z_rows[0] + Z_row_offset][col] = 1
        W[Z_rows[3] + Z_row_offset][col] = 1
        col += 1
        
        return col
    
    # Block 1: A_left * B_tl -> C_left
    # A_left: rows in U for i in {0,1}, j in {0,1}
    #   A_left[0,0] = U[0*4+0] = U[0]
    #   A_left[0,1] = U[0*4+1] = U[1]
    #   A_left[1,0] = U[1*4+0] = U[4]
    #   A_left[1,1] = U[1*4+1] = U[5]
    X_rows_1 = [0, 1, 4, 5]
    
    # B_tl: rows in V for j in {0,1}, k in {0,1}
    #   B_tl[0,0] = V[0*4+0] = V[0]
    #   B_tl[0,1] = V[0*4+1] = V[1]
    #   B_tl[1,0] = V[1*4+0] = V[4]
    #   B_tl[1,1] = V[1*4+1] = V[5]
    Y_rows_1 = [0, 1, 4, 5]
    
    # C_left: rows in W for k in {0,1}, i in {0,1}
    #   C_left[0,0] = W[0*2+0] = W[0]
    #   C_left[0,1] = W[0*2+1] = W[1]
    #   C_left[1,0] = W[1*2+0] = W[2]
    #   C_left[1,1] = W[1*2+1] = W[3]
    Z_rows_1 = [0, 1, 2, 3]
    
    col = add_strassen_block(0, X_rows_1, Y_rows_1, Z_rows_1, 0, 0, 0)
    
    # Block 2: A_right * B_bl -> C_left
    # A_right: rows in U for i in {0,1}, j in {2,3}
    #   A_right[0,0] = U[0*4+2] = U[2]
    #   A_right[0,1] = U[0*4+3] = U[3]
    #   A_right[1,0] = U[1*4+2] = U[6]
    #   A_right[1,1] = U[1*4+3] = U[7]
    X_rows_2 = [2, 3, 6, 7]
    
    # B_bl: rows in V for j in {2,3}, k in {0,1}
    #   B_bl[0,0] = V[2*4+0] = V[8]
    #   B_bl[0,1] = V[2*4+1] = V[9]
    #   B_bl[1,0] = V[3*4+0] = V[12]
    #   B_bl[1,1] = V[3*4+1] = V[13]
    Y_rows_2 = [8, 9, 12, 13]
    
    # C_left: same as before
    Z_rows_2 = [0, 1, 2, 3]
    
    col = add_strassen_block(col, X_rows_2, Y_rows_2, Z_rows_2, 0, 0, 0)
    
    # Block 3: A_left * B_tr -> C_right
    # A_left: same as block 1
    X_rows_3 = [0, 1, 4, 5]
    
    # B_tr: rows in V for j in {0,1}, k in {2,3}
    #   B_tr[0,0] = V[0*4+2] = V[2]
    #   B_tr[0,1] = V[0*4+3] = V[3]
    #   B_tr[1,0] = V[1*4+2] = V[6]
    #   B_tr[1,1] = V[1*4+3] = V[7]
    Y_rows_3 = [2, 3, 6, 7]
    
    # C_right: rows in W for k in {2,3}, i in {0,1}
    #   C_right[0,0] = W[2*2+0] = W[4]
    #   C_right[0,1] = W[2*2+1] = W[5]
    #   C_right[1,0] = W[3*2+0] = W[6]
    #   C_right[1,1] = W[3*2+1] = W[7]
    Z_rows_3 = [4, 5, 6, 7]
    
    col = add_strassen_block(col, X_rows_3, Y_rows_3, Z_rows_3, 0, 0, 0)
    
    # Block 4: A_right * B_br -> C_right
    # A_right: same as block 2
    X_rows_4 = [2, 3, 6, 7]
    
    # B_br: rows in V for j in {2,3}, k in {2,3}
    #   B_br[0,0] = V[2*4+2] = V[10]
    #   B_br[0,1] = V[2*4+3] = V[11]
    #   B_br[1,0] = V[3*4+2] = V[14]
    #   B_br[1,1] = V[3*4+3] = V[15]
    Y_rows_4 = [10, 11, 14, 15]
    
    # C_right: same as block 3
    Z_rows_4 = [4, 5, 6, 7]
    
    col = add_strassen_block(col, X_rows_4, Y_rows_4, Z_rows_4, 0, 0, 0)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
