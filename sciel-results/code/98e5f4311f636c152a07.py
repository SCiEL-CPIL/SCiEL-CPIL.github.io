import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds, leave 30s margin for safety

    # The tensor T for matrix multiplication (n x m) * (m x p) -> (n x p)
    # Convention: W[k*n + i, r] as per task description.
    # Let's define the tensor T[i, j, k] = 1 for the multiplication A[i,j] * B[j,k] = C[i,k].
    # The flattening indices:
    # A index: a = i*m + j
    # B index: b = j*p + k
    # C index: c = i*p + k  (standard row-major for C)
    # BUT the task says: "The output factor follows the official trace convention W[k*n+i,r]"
    # This implies the C index is c = k*n + i.
    # Let's verify: C has dimensions (n, p). If c = k*n + i, then i = c % n and k = c // n.
    # This is column-major order for C if we consider i as row and k as column?
    # Or it's just a specific indexing.
    # Let's stick to the explicit instruction: W index corresponds to k*n + i.
    
    R_max = 120
    U = np.zeros((n*m, R_max), dtype=int)
    V = np.zeros((m*p, R_max), dtype=int)
    W = np.zeros((p*n, R_max), dtype=int)
    
    col = 0
    
    # 1. Try Strassen for 2x2x2 sub-blocks
    # We can partition A (2x5) into A0 (1x5) and A1 (1x5)? No, Strassen needs square blocks.
    # We can't apply standard Strassen directly to 2x5 * 5x5 without padding to 2x6 * 6x6 or similar.
    # However, we can apply it to sub-blocks if dimensions allow.
    # Since n=2, we can split A into two 1x5 rows.
    # Since p=5, we can split C into 5 columns.
    # Let's try a block decomposition.
    
    # Actually, for 2x5 * 5x5, the schoolbook rank is 50.
    # We can try to use Strassen on the 2x2 sub-blocks of the multiplication.
    # Let's partition the 5 columns of A and rows of B into blocks of size 2 and 1.
    # 5 = 2 + 2 + 1
    # Let's create 3 blocks of A: A0 (2x2), A1 (2x2), A2 (2x1)
    # Let's create 3 blocks of B: B0 (2x5), B1 (2x5), B2 (1x5)
    # This doesn't align for multiplication.
    
    # Better partition:
    # A is 2x5. B is 5x5.
    # Split A into A_top (1x5) and A_bot (1x5).
    # Split B into B_left (5x2), B_mid (5x2), B_right (5x1).
    # C = A_top * B + A_bot * B.
    # C is 2x5.
    
    # Let's try a different approach: Use the known rank for small matrices.
    # Rank of 2x2x2 is 7 (Strassen).
    # Rank of 2x3x2 is 11?
    # Rank of 2x5x5 is unknown, but likely less than 50.
    
    # Let's construct a decomposition using block Strassen where possible.
    # We can view the 2x5 matrix A as two 1x5 vectors a0, a1.
    # And B as a 5x5 matrix.
    # C = a0^T B + a1^T B? No, A is 2x5, so A[i, j] * B[j, k].
    # C[i, k] = sum_j A[i, j] B[j, k].
    
    # Let's partition the summation index j (0..4) into blocks.
    # j = 0, 1, 2, 3, 4
    # We can group j=0,1 and j=2,3 and j=4.
    # For j=0,1: A[:, 0:2] (2x2) * B[0:2, :] (2x5) -> (2x5).
    # For j=2,3: A[:, 2:4] (2x2) * B[2:4, :] (2x5) -> (2x5).
    # For j=4: A[:, 4] (2x1) * B[4, :] (1x5) -> (2x5).
    
    # The first two blocks are 2x2 * 2x5.
    # We can use Strassen for the 2x2 part?
    # Strassen computes 2x2 * 2x2. Here we have 2x2 * 2x5.
    # We can split B[0:2, :] into B[0:2, 0:2] (2x2), B[0:2, 2:4] (2x2), B[0:2, 4] (2x1).
    # So A0 (2x2) * B00 (2x2) -> 2x2 output. Use Strassen (rank 7).
    # A0 (2x2) * B01 (2x2) -> 2x2 output. Use Strassen (rank 7).
    # A0 (2x2) * B02 (2x1) -> 2x1 output. Use schoolbook (rank 2*1*1=2? Or 2*2*1=4? No, 2x2*2x1 is 4 mults).
    # Actually, 2x2 * 2x1 can be done with 2*2=4 mults.
    
    # Let's refine the blocks to maximize Strassen usage.
    # Partition A (2x5) into A0 (2x2), A1 (2x2), A2 (2x1).
    # Partition B (5x5) into B0 (2x5), B1 (2x5), B2 (1x5).
    
    # Block 0: A0 (2x2) * B0 (2x5)
    # Split B0 into B00 (2x2), B01 (2x2), B02 (2x1).
    # A0 * B00 -> 2x2 (Strassen, rank 7)
    # A0 * B01 -> 2x2 (Strassen, rank 7)
    # A0 * B02 -> 2x1 (Schoolbook, rank 4)
    
    # Block 1: A1 (2x2) * B1 (2x5)
    # Split B1 into B10 (2x2), B11 (2x2), B12 (2x1).
    # A1 * B10 -> 2x2 (Strassen, rank 7)
    # A1 * B11 -> 2x2 (Strassen, rank 7)
    # A1 * B12 -> 2x1 (Schoolbook, rank 4)
    
    # Block 2: A2 (2x1) * B2 (1x5)
    # This is 2x1 * 1x5 -> 2x5.
    # Schoolbook rank 2*1*5 = 10.
    
    # Total rank = 7+7+4 + 7+7+4 + 10 = 46.
    # This is better than 50.
    
    # Let's implement this.
    
    def add_strassen_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset, col_idx):
        """
        Adds Strassen terms for A[a_row_offset:a_row_offset+2, b_col_offset:b_col_offset+2] * B[b_col_offset:b_col_offset+2, c_col_offset:c_col_offset+2]
        Note: Indices are global.
        A index: i*m + j
        B index: j*p + k
        C index: k*n + i
        """
        n_local = 2
        m_local = 2
        p_local = 2
        
        # Strassen formulas for 2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Helper to get indices
        def a_idx(i, j):
            return (a_row_offset + i) * m + (b_col_offset + j)
        
        def b_idx(i, j):
            return (b_col_offset + i) * p + (c_col_offset + j)
        
        def c_idx(i, k):
            # Task convention: W[k*n + i]
            return (c_col_offset + k) * n + (c_row_offset + i)
        
        # M1
        U[a_idx(0,0)][col_idx] += 1
        U[a_idx(1,1)][col_idx] += 1
        V[b_idx(0,0)][col_idx] += 1
        V[b_idx(1,1)][col_idx] += 1
        W[c_idx(0,0)][col_idx] += 1
        W[c_idx(1,1)][col_idx] += 1
        col_idx += 1
        
        # M2
        U[a_idx(1,0)][col_idx] += 1
        U[a_idx(1,1)][col_idx] += 1
        V[b_idx(0,0)][col_idx] += 1
        W[c_idx(1,0)][col_idx] += 1
        col_idx += 1
        
        # M3
        U[a_idx(0,0)][col_idx] += 1
        V[b_idx(1,0)][col_idx] += 1
        V[b_idx(1,1)][col_idx] -= 1
        W[c_idx(0,1)][col_idx] += 1
        W[c_idx(1,1)][col_idx] += 1
        col_idx += 1
        
        # M4
        U[a_idx(1,1)][col_idx] += 1
        V[b_idx(0,1)][col_idx] += 1
        V[b_idx(0,0)][col_idx] -= 1
        W[c_idx(0,0)][col_idx] += 1
        W[c_idx(1,0)][col_idx] += 1
        col_idx += 1
        
        # M5
        U[a_idx(0,0)][col_idx] += 1
        U[a_idx(0,1)][col_idx] += 1
        V[b_idx(1,1)][col_idx] += 1
        W[c_idx(0,0)][col_idx] -= 1
        W[c_idx(0,1)][col_idx] += 1
        col_idx += 1
        
        # M6
        U[a_idx(1,0)][col_idx] += 1
        U[a_idx(0,0)][col_idx] -= 1
        V[b_idx(0,0)][col_idx] += 1
        V[b_idx(0,1)][col_idx] += 1
        W[c_idx(1,1)][col_idx] += 1
        col_idx += 1
        
        # M7
        U[a_idx(0,1)][col_idx] += 1
        U[a_idx(1,1)][col_idx] -= 1
        V[b_idx(1,0)][col_idx] += 1
        V[b_idx(1,1)][col_idx] += 1
        W[c_idx(0,0)][col_idx] += 1
        col_idx += 1
        
        return col_idx

    def add_schoolbook(a_row_offset, a_col_offset, a_rows, a_cols, b_row_offset, b_col_offset, b_rows, b_cols, c_row_offset, c_col_offset, col_idx):
        """
        Adds schoolbook terms for A[a_row_offset:a_row_offset+a_rows, a_col_offset:a_col_offset+a_cols] * B[b_row_offset:b_row_offset+b_rows, b_col_offset:b_col_offset+b_cols]
        Assumes a_cols == b_rows.
        """
        for i in range(a_rows):
            for j in range(a_cols):
                for k in range(b_cols):
                    a_i = (a_row_offset + i) * m + (a_col_offset + j)
                    b_i = (b_row_offset + j) * p + (b_col_offset + k)
                    c_i = (c_col_offset + k) * n + (c_row_offset + i)
                    U[a_i][col_idx] = 1
                    V[b_i][col_idx] = 1
                    W[c_i][col_idx] = 1
                    col_idx += 1
        return col_idx

    # Block 0: A[:, 0:2] * B[0:2, :]
    # A0 is at rows 0,1 cols 0,1.
    # B0 is at rows 0,1 cols 0,4.
    
    # A0 * B00 (cols 0,1 of B)
    col = add_strassen_2x2(0, 0, 0, 0, col)
    # A0 * B01 (cols 2,3 of B)
    col = add_strassen_2x2(0, 0, 0, 2, col)
    # A0 * B02 (col 4 of B)
    col = add_schoolbook(0, 0, 2, 2, 0, 4, 2, 1, 0, 4, col)
    
    # Block 1: A[:, 2:4] * B[2:4, :]
    # A1 is at rows 0,1 cols 2,3.
    # B1 is at rows 2,3 cols 0,4.
    
    # A1 * B10 (cols 0,1 of B)
    col = add_strassen_2x2(0, 2, 0, 0, col)
    # A1 * B11 (cols 2,3 of B)
    col = add_strassen_2x2(0, 2, 0, 2, col)
    # A1 * B12 (col 4 of B)
    col = add_schoolbook(0, 2, 2, 2, 2, 4, 2, 1, 0, 4, col)
    
    # Block 2: A[:, 4] * B[4, :]
    # A2 is at rows 0,1 col 4.
    # B2 is at row 4 cols 0,4.
    
    # A2 * B2
    col = add_schoolbook(0, 4, 2, 1, 4, 0, 1, 5, 0, 0, col)
    
    rank = col
    
    # Verify
    if time.time() > deadline:
        # Return what we have, but might be incomplete if we exceeded time.
        # However, construction is fast.
        pass
        
    # Convert to lists
    U_list = U[:, :rank].tolist()
    V_list = V[:, :rank].tolist()
    W_list = W[:, :rank].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
