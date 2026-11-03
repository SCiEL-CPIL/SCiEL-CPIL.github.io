import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    
    We use a block decomposition approach with Strassen for 2x2x2 blocks.
    
    Split A (3x3) into quadrants:
    A = [[A11 (2x2), A12 (2x1)],
         [A21 (1x2), A22 (1x1)]]
    
    Split B (3x5) into:
    B = [[B11 (2x2), B12 (2x1), B13 (2x2)],
         [B21 (1x2), B22 (1x1), B23 (1x2)]]
    
    Where:
    - B11: rows 0-1, cols 0-1
    - B12: rows 0-1, col 2
    - B13: rows 0-1, cols 3-4
    - B21: row 2, cols 0-1
    - B22: row 2, col 2
    - B23: row 2, cols 3-4
    
    Sub-multiplications and their ranks:
    - A11*B11: M_{2,2,2} -> rank 7 (Strassen)
    - A11*B12: M_{2,2,1} -> rank 4 (direct)
    - A11*B13: M_{2,2,2} -> rank 7 (Strassen)
    - A12*B21: M_{2,1,2} -> rank 4 (direct)
    - A12*B22: M_{2,1,1} -> rank 2 (direct)
    - A12*B23: M_{2,1,2} -> rank 4 (direct)
    - A21*B11: M_{1,2,2} -> rank 4 (direct)
    - A21*B12: M_{1,2,1} -> rank 2 (direct)
    - A21*B13: M_{1,2,2} -> rank 4 (direct)
    - A22*B21: M_{1,1,2} -> rank 2 (direct)
    - A22*B22: M_{1,1,1} -> rank 1 (direct)
    - A22*B23: M_{1,1,2} -> rank 2 (direct)
    
    Total rank: 7+4+7+4+2+4+4+2+4+2+1+2 = 43
    """
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds to leave margin
    
    # Index mappings:
    # U index: i*m + j = i*3 + j, where i is row of A (0,1,2), j is col of A (0,1,2)
    # V index: j*p + k = j*5 + k, where j is row of B (0,1,2), k is col of B (0,1,2,3,4)
    # W index: k*n + i = k*3 + i, where k is col of C (0,1,2,3,4), i is row of C (0,1,2)
    
    U = [[] for _ in range(9)]
    V = [[] for _ in range(15)]
    W = [[] for _ in range(15)]
    
    def term(u_idx, v_idx, w_idx, val=1):
        """Add a rank-1 term"""
        U[u_idx].append(val)
        V[v_idx].append(val)
        W[w_idx].append(val)
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for a 2x2 block multiplication.
        a_rows, a_cols: indices into A for the 2x2 block
        b_rows, b_cols: indices into B for the 2x2 block  
        c_rows, c_cols: indices into C for the 2x2 output block
        """
        a00 = a_rows[0]*3 + a_cols[0]
        a01 = a_rows[0]*3 + a_cols[1]
        a10 = a_rows[1]*3 + a_cols[0]
        a11 = a_rows[1]*3 + a_cols[1]
        
        b00 = b_rows[0]*5 + b_cols[0]
        b01 = b_rows[0]*5 + b_cols[1]
        b10 = b_rows[1]*5 + b_cols[0]
        b11 = b_rows[1]*5 + b_cols[1]
        
        c00 = c_rows[0]*3 + c_cols[0]
        c01 = c_rows[0]*3 + c_cols[1]
        c10 = c_rows[1]*3 + c_cols[0]
        c11 = c_rows[1]*3 + c_cols[1]
        
        # M1 = (a00+a11)*(b00+b11)
        term(a00, b00, c00); term(a00, b11, c00); term(a11, b00, c00); term(a11, b11, c00)
        term(a00, b00, c01); term(a00, b11, c01); term(a11, b00, c01); term(a11, b11, c01)
        term(a00, b00, c10); term(a00, b11, c10); term(a11, b00, c10); term(a11, b11, c10)
        term(a00, b00, c11); term(a00, b11, c11); term(a11, b00, c11); term(a11, b11, c11)
        
        # M2 = (a10+a11)*b00
        term(a10, b00, c10); term(a11, b00, c10)
        term(a10, b00, c11, -1); term(a11, b00, c11, -1)
        
        # M3 = a00*(b01-b11)
        term(a00, b01, c01); term(a00, b11, c01, -1)
        term(a00, b01, c11); term(a00, b11, c11)
        
        # M4 = a11*(b10-b00)
        term(a11, b10, c00); term(a11, b00, c00, -1)
        term(a11, b10, c10); term(a11, b00, c10)
        
        # M5 = (a00+a01)*b11
        term(a00, b11, c00, -1); term(a01, b11, c00, -1)
        term(a00, b11, c01); term(a01, b11, c01)
        
        # M6 = (a10-a00)*(b00+b01)
        term(a10, b00, c11); term(a10, b01, c11)
        term(a00, b00, c11, -1); term(a00, b01, c11, -1)
        
        # M7 = (a01-a11)*(b10+b11)
        term(a01, b10, c00); term(a01, b11, c00)
        term(a11, b10, c00, -1); term(a11, b11, c00, -1)
    
    # Add Strassen for A11*B11 -> C[0:2, 0:2]
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Add direct 2x2x1 for A11*B12 -> C[0:2, 2]
    term(0*3+0, 0*5+2, 0*3+2)
    term(0*3+1, 1*5+2, 0*3+2)
    term(1*3+0, 0*5+2, 1*3+2)
    term(1*3+1, 1*5+2, 1*3+2)
    
    # Add Strassen for A11*B13 -> C[0:2, 3:5]
    add_strassen_block([0,1], [0,1], [0,1], [3,4], [0,1], [3,4])
    
    # Add direct 2x1x2 for A12*B21 -> C[0:2, 0:2]
    # A12 is A[0:2, 2] (rows 0-1, col 2), B21 is B[2, 0:2] (row 2, cols 0-1)
    # C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
    term(0*3+2, 2*5+0, 0*3+0)
    term(0*3+2, 2*5+1, 0*3+1)
    term(1*3+2, 2*5+0, 1*3+0)
    term(1*3+2, 2*5+1, 1*3+1)
    
    # Add direct 2x1x1 for A12*B22 -> C[0:2, 2]
    # A12 is A[0:2, 2], B22 is B[2, 2]
    # C[0,2] += A[0,2]*B[2,2], C[1,2] += A[1,2]*B[2,2]
    term(0*3+2, 2*5+2, 0*3+2)
    term(1*3+2, 2*5+2, 1*3+2)
    
    # Add direct 2x1x2 for A12*B23 -> C[0:2, 3:5]
    # A12 is A[0:2, 2], B23 is B[2, 3:5]
    # C[0,3] += A[0,2]*B[2,3], C[0,4] += A[0,2]*B[2,4]
    # C[1,3] += A[1,2]*B[2,3], C[1,4] += A[1,2]*B[2,4]
    term(0*3+2, 2*5+3, 0*3+3)
    term(0*3+2, 2*5+4, 0*3+4)
    term(1*3+2, 2*5+3, 1*3+3)
    term(1*3+2, 2*5+4, 1*3+4)
    
    # Add direct 1x2x2 for A21*B11 -> C[2, 0:2]
    # A21 is A[2, 0:2] (row 2, cols 0-1), B11 is B[0:2, 0:2]
    # C[2,0] += A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] += A[2,0]*B[0,1] + A[2,1]*B[1,1]
    term(2*3+0, 0*5+0, 2*3+0)
    term(2*3+1, 1*5+0, 2*3+0)
    term(2*3+0, 0*5+1, 2*3+1)
    term(2*3+1, 1*5+1, 2*3+1)
    
    # Add direct 1x2x1 for A21*B12 -> C[2, 2]
    # A21 is A[2, 0:2], B12 is B[0:2, 2]
    # C[2,2] += A[2,0]*B[0,2] + A[2,1]*B[1,2]
    term(2*3+0, 0*5+2, 2*3+2)
    term(2*3+1, 1*5+2, 2*3+2)
    
    # Add direct 1x2x2 for A21*B13 -> C[2, 3:5]
    # A21 is A[2, 0:2], B13 is B[0:2, 3:5]
    # C[2,3] += A[2,0]*B[0,3] + A[2,1]*B[1,3]
    # C[2,4] += A[2,0]*B[0,4] + A[2,1]*B[1,4]
    term(2*3+0, 0*5+3, 2*3+3)
    term(2*3+1, 1*5+3, 2*3+3)
    term(2*3+0, 0*5+4, 2*3+4)
    term(2*3+1, 1*5+4, 2*3+4)
    
    # Add direct 1x1x2 for A22*B21 -> C[2, 0:2]
    # A22 is A[2, 2], B21 is B[2, 0:2]
    # C[2,0] += A[2,2]*B[2,0], C[2,1] += A[2,2]*B[2,1]
    term(2*3+2, 2*5+0, 2*3+0)
    term(2*3+2, 2*5+1, 2*3+1)
    
    # Add direct 1x1x1 for A22*B22 -> C[2, 2]
    # A22 is A[2, 2], B22 is B[2, 2]
    # C[2,2] += A[2,2]*B[2,2]
    term(2*3+2, 2*5+2, 2*3+2)
    
    # Add direct 1x1x2 for A22*B23 -> C[2, 3:5]
    # A22 is A[2, 2], B23 is B[2, 3:5]
    # C[2,3] += A[2,2]*B[2,3], C[2,4] += A[2,2]*B[2,4]
    term(2*3+2, 2*5+3, 2*3+3)
    term(2*3+2, 2*5+4, 2*3+4)
    
    # Convert to proper format
    rank = len(U[0])
    U_mat = [[0] * rank for _ in range(9)]
    V_mat = [[0] * rank for _ in range(15)]
    W_mat = [[0] * rank for _ in range(15)]
    
    for i in range(9):
        for c in range(rank):
            U_mat[i][c] = U[i][c]
    for i in range(15):
        for c in range(rank):
            V_mat[i][c] = V[i][c]
    for i in range(15):
        for c in range(rank):
            W_mat[i][c] = W[i][c]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
