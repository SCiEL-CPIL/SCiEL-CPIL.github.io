import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for 2x2x1 blocks.
    # Layout:
    # A is 2x4, B is 4x7, C is 2x7
    # Split A into [A0 | A1] where A0, A1 are 2x2
    # Split B into [B0; B1; B2] where B0 is 2x3, B1 is 2x3, B2 is 2x1
    # Actually, let's split B into blocks of 2 rows: B_top (2x7), B_bottom (2x7)
    # Then C = A0 * B_top + A1 * B_bottom
    # Each A_i * B_j is 2x2 times 2x7 = 2x7
    # For 2x2 * 2x7: split into three 2x2*2x2 blocks and one 2x2*2x1 block
    # 2x2*2x2: Strassen rank 7
    # 2x2*2x1: schoolbook rank 4
    # So each 2x2*2x7: 3*7 + 4 = 25
    # Total: 2 * 25 = 50
    
    # Let's use a more careful indexing.
    # A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # C[i,k] for i in {0,1}, k in {0,1,2,3,4,5,6}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group j: {0,1} and {2,3}
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    # For fixed i, the first sum is (A[i,0], A[i,1]) dot (B[0,k], B[1,k])
    # This is a 1x2 times 2x7 product.
    # But we can do it in blocks of k.
    
    # Let's just use the block decomposition:
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x7), B_bottom = B[2:4, :] (2x7)
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 * 2x7
    # Split B_top into B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:7]
    # = 2x2*2x2 + 2x2*2x2 + 2x2*2x2 + 2x2*2x1
    # = 7 + 7 + 7 + 4 = 25 terms
    
    # Similarly A_right * B_bottom = 25 terms
    # Total = 50 terms
    
    # Let's implement this.
    
    rank = 50
    U = np.zeros((n*m, rank))  # 8 x 50
    V = np.zeros((m*p, rank))  # 28 x 50
    W = np.zeros((p*n, rank))  # 14 x 50
    
    col = 0
    
    def strassen_222(a_rows, b_cols, c_rows, c_cols, offset):
        """
        Add Strassen terms for A[a_rows, :2] * B[:2, b_cols] -> C[c_rows, c_cols]
        a_rows: list of 2 row indices in A (0,1)
        b_cols: list of 2 col indices in B (0,1) - these are the row indices in B
        c_rows: list of 2 row indices in C
        c_cols: list of 2 col indices in C
        offset: starting column
        Returns next column
        """
        # A entries: A[a_rows[0], 0], A[a_rows[0], 1], A[a_rows[1], 0], A[a_rows[1], 1]
        # B entries: B[0, b_cols[0]], B[0, b_cols[1]], B[1, b_cols[0]], B[1, b_cols[1]]
        # C entries: C[c_rows[0], c_cols[0]], etc.
        
        # Index in U: a_rows[0]*m + 0, a_rows[0]*m + 1, a_rows[1]*m + 0, a_rows[1]*m + 1
        a00 = a_rows[0]*m + 0
        a01 = a_rows[0]*m + 1
        a10 = a_rows[1]*m + 0
        a11 = a_rows[1]*m + 1
        
        # Index in V: 0*p + b_cols[0], 0*p + b_cols[1], 1*p + b_cols[0], 1*p + b_cols[1]
        b00 = 0*p + b_cols[0]
        b01 = 0*p + b_cols[1]
        b10 = 1*p + b_cols[0]
        b11 = 1*p + b_cols[1]
        
        # Index in W: c_cols[0]*n + c_rows[0], c_cols[0]*n + c_rows[1], c_cols[1]*n + c_rows[0], c_cols[1]*n + c_rows[1]
        c00 = c_cols[0]*n + c_rows[0]
        c01 = c_cols[0]*n + c_rows[1]
        c10 = c_cols[1]*n + c_rows[0]
        c11 = c_cols[1]*n + c_rows[1]
        
        # Strassen 7 terms
        # M1 = (A11 - A01) * (B00 + B10)
        U[a11][offset+0] += 1
        U[a01][offset+0] -= 1
        V[b00][offset+0] += 1
        V[b10][offset+0] += 1
        W[c00][offset+0] += 1
        W[c11][offset+0] += 1
        
        # M2 = A11 * B10
        U[a11][offset+1] += 1
        V[b10][offset+1] += 1
        W[c10][offset+1] += 1
        W[c11][offset+1] -= 1
        
        # M3 = (A00 + A11) * B11
        U[a00][offset+2] += 1
        U[a11][offset+2] += 1
        V[b11][offset+2] += 1
        W[c00][offset+2] += 1
        W[c01][offset+2] += 1
        
        # M4 = A00 * (B01 - B00)
        U[a00][offset+3] += 1
        V[b01][offset+3] += 1
        V[b00][offset+3] -= 1
        W[c00][offset+3] += 1
        W[c10][offset+3] += 1
        
        # M5 = (A10 + A00) * B00
        U[a10][offset+4] += 1
        U[a00][offset+4] += 1
        V[b00][offset+4] += 1
        W[c00][offset+4] -= 1
        W[c01][offset+4] += 1
        
        # M6 = (A11 - A10) * (B00 + B10)
        U[a11][offset+5] += 1
        U[a10][offset+5] -= 1
        V[b00][offset+5] += 1
        V[b10][offset+5] += 1
        W[c11][offset+5] += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[a01][offset+6] += 1
        U[a11][offset+6] -= 1
        V[b10][offset+6] += 1
        V[b11][offset+6] += 1
        W[c00][offset+6] += 1
        
        return offset + 7
    
    def schoolbook_221(a_rows, b_col, c_rows, c_col, offset):
        """
        Add schoolbook terms for A[a_rows, :2] * B[:2, b_col] -> C[c_rows, c_col]
        """
        a00 = a_rows[0]*m + 0
        a01 = a_rows[0]*m + 1
        a10 = a_rows[1]*m + 0
        a11 = a_rows[1]*m + 1
        
        b0 = 0*p + b_col
        b1 = 1*p + b_col
        
        c0 = c_col*n + c_rows[0]
        c1 = c_col*n + c_rows[1]
        
        # A00*B0 -> C0
        U[a00][offset+0] += 1
        V[b0][offset+0] += 1
        W[c0][offset+0] += 1
        
        # A01*B1 -> C0
        U[a01][offset+1] += 1
        V[b1][offset+1] += 1
        W[c0][offset+1] += 1
        
        # A10*B0 -> C1
        U[a10][offset+2] += 1
        V[b0][offset+2] += 1
        W[c1][offset+2] += 1
        
        # A11*B1 -> C1
        U[a11][offset+3] += 1
        V[b1][offset+3] += 1
        W[c1][offset+3] += 1
        
        return offset + 4
    
    # A_left * B_top
    # A_left rows: 0,1; cols 0,1
    # B_top rows: 0,1; cols 0-6
    # C rows: 0,1; cols 0-6
    
    # Block 1: k in {0,1}
    col = strassen_222([0,1], [0,1], [0,1], [0,1], col)
    # Block 2: k in {2,3}
    col = strassen_222([0,1], [0,1], [0,1], [2,3], col)
    # Block 3: k in {4,5}
    col = strassen_222([0,1], [0,1], [0,1], [4,5], col)
    # Block 4: k in {6}
    col = schoolbook_221([0,1], 6, [0,1], 6, col)
    
    # A_right * B_bottom
    # A_right rows: 0,1; cols 2,3
    # B_bottom rows: 2,3; cols 0-6
    # C rows: 0,1; cols 0-6
    
    # Block 1: k in {0,1}
    col = strassen_222([0,1], [2,3], [0,1], [0,1], col)
    # Block 2: k in {2,3}
    col = strassen_222([0,1], [2,3], [0,1], [2,3], col)
    # Block 3: k in {4,5}
    col = strassen_222([0,1], [2,3], [0,1], [4,5], col)
    # Block 4: k in {6}
    col = schoolbook_221([0,1], 6, [0,1], 6, col)
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to list of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
