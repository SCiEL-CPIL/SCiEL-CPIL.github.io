"""Improved program: block decomposition using Strassen for 2x2x2 blocks.
M_{2,4,5}: split into two M_{2,2,5} blocks.
Each M_{2,2,5}: split 5 cols into (2,2,1) => 2*7 + 1*4 = 18 rank.
Total rank = 2 * 18 = 36.
"""
import time
import sys

DIMENSIONS = [2, 4, 5]

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We need U of shape (n*m, rank) = (8, 36)
    # V of shape (m*p, rank) = (20, 36)
    # W of shape (p*n, rank) = (10, 36)
    
    total_rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to build Strassen 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # Indices in A: a(A,0,0), a(A,0,1), a(A,1,0), a(A,1,1)
    # Indices in B: b(B,0,0), b(B,0,1), b(B,1,0), b(B,1,1)
    # Indices in C: c(C,0,0), c(C,0,1), c(C,1,0), c(C,1,1)
    #
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen(a_offset, b_offset, c_offset, U_rows, V_rows, W_rows, rank_offset):
        """Add 7 rank-1 tensors for 2x2x2 Strassen multiplication.
        a_offset: offset in A rows (in 0..7 for the 2x4 matrix A)
        b_offset: offset in B rows (in 0..19 for the 4x5 matrix B)
        c_offset: offset in C rows (in 0..9 for the 5x2 matrix C, flattened)
        """
        # A is 2x4, stored row-major in U: U[i*4+j, k]
        # So A element (i,j) is at U row (i*4+j)
        # B is 4x5, stored row-major in V: V[j*5+k, k]
        # So B element (j,k) is at V row (j*5+k)
        # C is 5x2, stored as W: W[k*2+i, k]
        # So C element (k,i) is at W row (k*2+i)
        
        # For the subblock at a_offset (row in 2x4), the 2x2 A block uses rows:
        # a_offset, a_offset+1 (within the 4-column width)
        # For the subblock at b_offset (row in 4x5), the 2x2 B block uses columns:
        # b_offset, b_offset+1 (within the 5-column width)
        # But wait, B is 4x5, so row j, col k is at V row j*5+k.
        # The B subblock is 2x2 using B rows j, j+1 and cols k, k+1.
        # In V: rows j*5+k, j*5+k+1, (j+1)*5+k, (j+1)*5+k+1
        
        # C subblock: 2x2 output at C rows (k, k+1) and cols (i, i+1) where i=a_offset//2... 
        # Actually let's be more careful.
        
        # Let's define the subblock more precisely.
        # A subblock: rows 2a, 2a+1 (of the 2 rows? No, A is 2x4)
        # Wait, A is 2x4. We split the 4 columns into two 2-column blocks.
        # Block 0: A[:, 0:2], Block 1: A[:, 2:4]
        # For A subblock at block b (0 or 1), columns 2b, 2b+1.
        # A element (i, 2b+j) is at U row (i*4 + 2b + j)
        
        # B is 4x5. We split rows into two 2-row blocks AND columns into (2,2,1).
        # Actually the task is M_{2,4,5}.
        # A is 2x4, B is 4x5, C is 2x5.
        # We split A's 4 columns into 2 blocks of 2: A_L (cols 0-1), A_R (cols 2-3).
        # We split B's 4 rows into 2 blocks of 2: B_T (rows 0-1), B_B (rows 2-3).
        # C = A_L * B_T (2x2 block at C rows 0-1? No, C is 2x5)
        
        # Let me reconsider the decomposition.
        # A is 2x4: A = [A_L | A_R] where A_L, A_R are 2x2.
        # B is 4x5: B = [B_T; B_B] where B_T, B_B are 2x5.
        # C = A_L * B_T + A_R * B_B, both 2x5.
        # So we have two M_{2,2,5} multiplications.
        
        # For each M_{2,2,5}: A is 2x2, B is 2x5, C is 2x5.
        # Split B's 5 columns into (2,2,1).
        # Part 1: A * B[:, 0:2] => M_{2,2,2} Strassen (rank 7)
        # Part 2: A * B[:, 2:4] => M_{2,2,2} Strassen (rank 7)
        # Part 3: A * B[:, 4:5] => M_{2,2,1} (rank 4)
        # Total: 18 per M_{2,2,5}, 36 total.
        
        # Now mapping to U, V, W:
        # U has shape (8, 36): rows indexed by A element (i,j) where i in 0,1, j in 0..3
        # V has shape (20, 36): rows indexed by B element (j,k) where j in 0..3, k in 0..4
        # W has shape (10, 36): rows indexed by C element (k,i) where k in 0..4, i in 0,1
        # C[k,i] = sum_j A[i,j] * B[j,k]
        
        # For the first M_{2,2,5} (A_L, B_T):
        # A_L uses A[i,0], A[i,1] => U rows (i*4+0), (i*4+1)
        # B_T uses B[0,k], B[1,k] => V rows (0*5+k), (1*5+k)
        # Output C_T uses C[k,0], C[k,1] for k in 0..4 => W rows (k*2+0), (k*2+1)
        
        # For the second M_{2,2,5} (A_R, B_B):
        # A_R uses A[i,2], A[i,3] => U rows (i*4+2), (i*4+3)
        # B_B uses B[2,k], B[3,k] => V rows (2*5+k), (3*5+k)
        # Output C_B uses C[k,0], C[k,1] for k in 0..4 => W rows (k*2+0), (k*2+1)
        
        # Wait, C = A_L B_T + A_R B_B. Both contribute to all of C (2x5).
        # So C[k,i] = (A_L B_T)[k,i] + (A_R B_B)[k,i]
        # The W tensor sums contributions from all rank-1 terms.
        
        # Let me define the Strassen contribution for a specific subblock.
        
        def emit_strassen(a_row0, a_row1, b_col0, b_col1, b_row0, b_row1, c_col_base, rank_start):
            """
            a_row0, a_row1: U row indices for A[0,?] and A[1,?] (the 2x2 A subblock)
            Actually A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
            In our mapping for a specific subblock:
            A[0,0] -> U row a_row0
            A[0,1] -> U row a_row0 + 1 (same i, next col in subblock)
            A[1,0] -> U row a_row1
            A[1,1] -> U row a_row1 + 1
            
            B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
            B[0,0] -> V row (b_row0 * 5 + b_col0)
            B[0,1] -> V row (b_row0 * 5 + b_col0 + 1)
            B[1,0] -> V row (b_row1 * 5 + b_col0)
            B[1,1] -> V row (b_row1 * 5 + b_col0 + 1)
            
            C output: 2x2 at C rows b_col0, b_col0+1 and cols c_col_base, c_col_base+1
            C[k, i] where k in {b_col0, b_col0+1}, i in {c_col_base, c_col_base+1}
            W row for C[k,i] is (k * 2 + i)
            """
            # U row indices for A elements
            A00 = a_row0
            A01 = a_row0 + 1
            A10 = a_row1
            A11 = a_row1 + 1
            
            # V row indices for B elements
            B00 = b_row0 * 5 + b_col0
            B01 = b_row0 * 5 + b_col0 + 1
            B10 = b_row1 * 5 + b_col0
            B11 = b_row1 * 5 + b_col0 + 1
            
            # W row indices for C elements
            # C rows are b_col0, b_col0+1; C cols are c_col_base, c_col_base+1
            C00 = (b_col0) * 2 + c_col_base
            C01 = (b_col0) * 2 + c_col_base + 1
            C10 = (b_col0 + 1) * 2 + c_col_base
            C11 = (b_col0 + 1) * 2 + c_col_base + 1
            
            # Strassen 7 mults:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            mults = []
            
            # M1
            mults.append(([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], [(C00, 1), (C11, 1)]))
            # M2
            mults.append(([(A10, 1), (A11, 1)], [(B00, 1)], [(C10, 1), (C11, -1)]))
            # M3
            mults.append(([(A00, 1)], [(B01, 1), (B11, -1)], [(C01, 1), (C11, 1)]))
            # M4
            mults.append(([(A11, 1)], [(B10, 1), (B00, -1)], [(C10, 1), (C11, -1)]))
            # M5
            mults.append(([(A00, 1), (A01, 1)], [(B11, 1)], [(C00, -1), (C01, 1)]))
            # M6
            mults.append(([(A10, 1), (A00, -1)], [(B00, 1), (B01, 1)], [(C11, 1)]))
            # M7
            mults.append(([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)], [(C00, 1)]))
            
            for (u_entries, v_entries, w_entries) in mults:
                col = rank_start + len(mults) - 1  # Wait, need to track properly
                # Actually let me restructure
        
        # Let me redo this more carefully by building U, V, W columns one by one.
        
        U = [[0] * 36 for _ in range(8)]
        V = [[0] * 36 for _ in range(20)]
        W = [[0] * 36 for _ in range(10)]
        
        col = 0
        
        # Define a helper to add a rank-1 term
        def add_rank1(u_entries, v_entries, w_entries, col_idx):
            for (row, val) in u_entries:
                U[row][col_idx] = val
            for (row, val) in v_entries:
                V[row][col_idx] = val
            for (row, val) in w_entries:
                W[row][col_idx] = val
        
        # For each of the two M_{2,2,5} blocks:
        # Block 0: A_L (U rows i*4+0, i*4+1), B_T (V rows 0*5+k, 1*5+k), C cols 0-1 of C (W col offset 0)
        # Block 1: A_R (U rows i*4+2, i*4+3), B_B (V rows 2*5+k, 3*5+k), C cols 0-1 of C (W col offset 0)
        
        # Wait, both blocks contribute to the same C (2x5). So C col offset is 0 for both.
        # C[k, i] where i in 0,1. W row is k*2+i.
        
        for block in range(2):
            # A subblock: i in 0,1; j in 2*block, 2*block+1
            # U row for A[i, 2*block + dj] is i*4 + 2*block + dj
            A00 = 0*4 + 2*block + 0  # A[0, 2*block]
            A01 = 0*4 + 2*block + 1  # A[0, 2*block+1]
            A10 = 1*4 + 2*block + 0  # A[1, 2*block]
            A11 = 1*4 + 2*block + 1  # A[1, 2*block+1]
            
            # B subblock: rows 2*block, 2*block+1; cols 0..4
            b_row0 = 2*block
            b_row1 = 2*block + 1
            
            # Part 1: cols 0,1
            b_col0 = 0
            b_col1 = 1
            B00 = b_row0 * 5 + b_col0
            B01 = b_row0 * 5 + b_col1
            B10 = b_row1 * 5 + b_col0
            B11 = b_row1 * 5 + b_col1
            C00 = b_col0 * 2 + 0  # C[0,0]
            C01 = b_col0 * 2 + 1  # C[0,1]
            C10 = b_col1 * 2 + 0  # C[1,0]
            C11 = b_col1 * 2 + 1  # C[1,1]
            
            # Strassen for part 1
            # M1
            add_rank1([(A00,1),(A11,1)], [(B00,1),(B11,1)], [(C00,1),(C11,1)], col)
            col += 1
            # M2
            add_rank1([(A10,1),(A11,1)], [(B00,1)], [(C10,1),(C11,-1)], col)
            col += 1
            # M3
            add_rank1([(A00,1)], [(B01,1),(B11,-1)], [(C01,1),(C11,1)], col)
            col += 1
            # M4
            add_rank1([(A11,1)], [(B10,1),(B00,-1)], [(C10,1),(C11,-1)], col)
            col += 1
            # M5
            add_rank1([(A00,1),(A01,1)], [(B11,1)], [(C00,-1),(C01,1)], col)
            col += 1
            # M6
            add_rank1([(A10,1),(A00,-1)], [(B00,1),(B01,1)], [(C11,1)], col)
            col += 1
            # M7
            add_rank1([(A01,1),(A11,-1)], [(B10,1),(B11,1)], [(C00,1)], col)
            col += 1
            
            # Part 2: cols 2,3
            b_col0 = 2
            b_col1 = 3
            B00 = b_row0 * 5 + b_col0
            B01 = b_row0 * 5 + b_col1
            B10 = b_row1 * 5 + b_col0
            B11 = b_row1 * 5 + b_col1
            C00 = b_col0 * 2 + 0  # C[2,0]
            C01 = b_col0 * 2 + 1  # C[2,1]
            C10 = b_col1 * 2 + 0  # C[3,0]
            C11 = b_col1 * 2 + 1  # C[3,1]
            
            # Strassen for part 2
            # M1
            add_rank1([(A00,1),(A11,1)], [(B00,1),(B11,1)], [(C00,1),(C11,1)], col)
            col += 1
            # M2
            add_rank1([(A10,1),(A11,1)], [(B00,1)], [(C10,1),(C11,-1)], col)
            col += 1
            # M3
            add_rank1([(A00,1)], [(B01,1),(B11,-1)], [(C01,1),(C11,1)], col)
            col += 1
            # M4
            add_rank1([(A11,1)], [(B10,1),(B00,-1)], [(C10,1),(C11,-1)], col)
            col += 1
            # M5
            add_rank1([(A00,1),(A01,1)], [(B11,1)], [(C00,-1),(C01,1)], col)
            col += 1
            # M6
            add_rank1([(A10,1),(A00,-1)], [(B00,1),(B01,1)], [(C11,1)], col)
            col += 1
            # M7
            add_rank1([(A01,1),(A11,-1)], [(B10,1),(B11,1)], [(C00,1)], col)
            col += 1
            
            # Part 3: col 4 (2x2x1, rank 4)
            # A is 2x2, B is 2x1, C is 2x1
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # 4 rank-1 terms:
            b_col0 = 4
            B00 = b_row0 * 5 + b_col0
            B10 = b_row1 * 5 + b_col0
            C0 = b_col0 * 2 + 0  # C[4,0]
            C1 = b_col0 * 2 + 1  # C[4,1]
            
            # A[0,0]*B[0,0]
            add_rank1([(A00,1)], [(B00,1)], [(C0,1)], col)
            col += 1
            # A[0,1]*B[1,0]
            add_rank1([(A01,1)], [(B10,1)], [(C0,1)], col)
            col += 1
            # A[1,0]*B[0,0]
            add_rank1([(A10,1)], [(B00,1)], [(C1,1)], col)
            col += 1
            # A[1,1]*B[1,0]
            add_rank1([(A11,1)], [(B10,1)], [(C1,1)], col)
            col += 1
        
        # col should be 36
        if col != 36:
            # Fallback to schoolbook if something went wrong
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        
        return {"U": U, "V": V, "W": W}
