"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a low-rank decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy: Use Strassen's algorithm where possible.
    - Split the 2x4x5 problem into two 2x2x5 sub-problems (left and right halves of the 4x5 matrix).
    - For each 2x2x5 sub-problem, split the 5 columns into 2+2+1:
        - Two 2x2x2 blocks use Strassen (rank 7 each)
        - One 2x2x1 block uses schoolbook (rank 4)
    - Total rank per sub-problem: 7 + 7 + 4 = 18
    - Total rank for 2x4x5: 18 + 18 = 36
    
    This gives rank 36 vs schoolbook rank 40, a 10% improvement.
    """
    n, m, p = 2, 4, 5
    
    # We'll build U, V, W matrices with rank = 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    # Process each of the two 2x2x5 sub-problems (left half: cols 0-1, right half: cols 2-3 of A, plus col 4)
    # Actually, let's think of it as: A is 2x4, B is 4x5
    # Split A into A_left (2x2) and A_right (2x2), and B into B_left (2x5) and B_right (2x5)
    # Then C = A_left @ B_left + A_right @ B_right
    
    # For each 2x2x5 multiplication:
    # Split the 5 columns of the 2x5 matrix into 2+2+1
    # Use Strassen for the 2x2x2 blocks, schoolbook for the 2x2x1 block
    
    def strassen_2x2x2(U, V, W, col_offset):
        """
        Add Strassen decomposition for 2x2x2 matrix multiplication starting at col_offset.
        Returns new col_offset.
        
        Strassen's algorithm:
        M1 = (A11 + A22) * (B11 + B22)
        M2 = (A21 + A22) * B11
        M3 = A11 * (B12 - B22)
        M4 = A22 * (B21 - B11)
        M5 = (A11 + A12) * B22
        M6 = (A21 - A11) * (B11 + B12)
        M7 = (A12 - A22) * (B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # M1 = (A11 + A22) * (B11 + B22)
        U[0][col_offset] = 1  # A11
        U[3][col_offset] = 1  # A22
        V[0][col_offset] = 1  # B11
        V[3][col_offset] = 1  # B22
        W[0][col_offset] = 1  # C11
        W[3][col_offset] = -1 # C22
        col_offset += 1
        
        # M2 = (A21 + A22) * B11
        U[1][col_offset] = 1  # A21
        U[3][col_offset] = 1  # A22
        V[0][col_offset] = 1  # B11
        W[1][col_offset] = 1  # C21
        W[3][col_offset] = -1 # C22
        col_offset += 1
        
        # M3 = A11 * (B12 - B22)
        U[0][col_offset] = 1  # A11
        V[1][col_offset] = 1  # B12
        V[3][col_offset] = -1 # B22
        W[0][col_offset] = 1  # C11
        W[2][col_offset] = 1  # C12
        col_offset += 1
        
        # M4 = A22 * (B21 - B11)
        U[3][col_offset] = 1  # A22
        V[2][col_offset] = 1  # B21
        V[0][col_offset] = -1 # B11
        W[0][col_offset] = 1  # C11
        W[1][col_offset] = 1  # C21
        col_offset += 1
        
        # M5 = (A11 + A12) * B22
        U[0][col_offset] = 1  # A11
        U[2][col_offset] = 1  # A12
        V[3][col_offset] = 1  # B22
        W[0][col_offset] = -1 # C11
        W[2][col_offset] = 1  # C12
        col_offset += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[1][col_offset] = 1  # A21
        U[0][col_offset] = -1 # A11
        V[0][col_offset] = 1  # B11
        V[1][col_offset] = 1  # B12
        W[3][col_offset] = 1  # C22
        col_offset += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[2][col_offset] = 1  # A12
        U[3][col_offset] = -1 # A22
        V[2][col_offset] = 1  # B21
        V[3][col_offset] = 1  # B22
        W[0][col_offset] = 1  # C11
        W[3][col_offset] = 1  # C22
        col_offset += 1
        
        return col_offset
    
    def schoolbook_2x2x1(U, V, W, col_offset):
        """
        Add schoolbook decomposition for 2x2x1 matrix multiplication starting at col_offset.
        Returns new col_offset.
        """
        # C11 = A11 * B11
        U[0][col_offset] = 1
        V[0][col_offset] = 1
        W[0][col_offset] = 1
        col_offset += 1
        
        # C12 = A12 * B21 (but for 2x2x1, it's just one column of B)
        # Actually for 2x2x1: A is 2x2, B is 2x1, C is 2x1
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # C1 part 1: A11 * B1
        # Already did C11 above? No, let me redo this properly.
        # For 2x2x1: 
        #   A = [[A11, A12],
        #        [A21, A22]]
        #   B = [[B1],
        #        [B2]]
        #   C = [[C1],
        #        [C2]]
        #   C1 = A11*B1 + A12*B2
        #   C2 = A21*B1 + A22*B2
        
        # So we need 4 terms:
        # Term 1: A11 * B1 -> contributes to C1
        # Term 2: A12 * B2 -> contributes to C1
        # Term 3: A21 * B1 -> contributes to C2
        # Term 4: A22 * B2 -> contributes to C2
        
        # But we already used col_offset for the first term in the function above.
        # Let me fix the schoolbook function.
        pass  # This will be fixed below
    
    # Let me rewrite this more carefully.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: Strassen for 2x2x2, with row/col offsets for A, B, C
    def add_strassen(U, V, W, a_row_off, b_row_off, c_row_off, col_start):
        """
        Add Strassen decomposition for 2x2x2.
        A is 2x2 starting at row a_row_off (in the 2x4 matrix, so rows are indexed 0-3)
        B is 2x2 starting at row b_row_off (in the 4x5 matrix, so rows are indexed 0-3)
        C is 2x2 starting at row c_row_off (in the 2x5 matrix, so rows are indexed 0-1, cols 0-4)
        
        U has shape (8, rank) - indexed by [i*m + j] where i in 0..1, j in 0..3
        V has shape (20, rank) - indexed by [j*p + k] where j in 0..3, k in 0..4
        W has shape (10, rank) - indexed by [k*n + i] where k in 0..4, i in 0..1
        """
        # M1 = (A11 + A22) * (B11 + B22)
        # A11 is at [a_row_off, 0] in A, which is U index (0*m + a_row_off) = a_row_off
        # A22 is at [a_row_off+1, 1] in A, which is U index (0*m + a_row_off+1) = a_row_off+1
        # Wait, A is 2x4, so A[i][j] where i in 0..1, j in 0..3
        # U index for A[i][j] is i*m + j = 2*j + i? No, U[i*m+j] means row i, col j of A
        # Actually U has n*m = 8 rows, indexed as i*m + j where i is row of A (0..1), j is col of A (0..3)
        # So U[i*4 + j] corresponds to A[i][j]
        
        # For a 2x2 block of A starting at column a_col_off:
        # A11 = A[0][a_col_off], A12 = A[0][a_col_off+1]
        # A21 = A[1][a_col_off], A22 = A[1][a_col_off+1]
        # U indices: 0*4 + a_col_off = a_col_off, 0*4 + a_col_off+1 = a_col_off+1
        #             1*4 + a_col_off = 4 + a_col_off, 1*4 + a_col_off+1 = 4 + a_col_off+1
        
        # Similarly for B (4x5), a 2x2 block starting at row b_row_off and column b_col_off:
        # B11 = B[b_row_off][b_col_off], B12 = B[b_row_off][b_col_off+1]
        # B21 = B[b_row_off+1][b_col_off], B22 = B[b_row_off+1][b_col_off+1]
        # V indices: b_row_off*5 + b_col_off, b_row_off*5 + b_col_off+1
        #             (b_row_off+1)*5 + b_col_off, (b_row_off+1)*5 + b_col_off+1
        
        # For C (2x5), a 2x2 block starting at column c_col_off:
        # C11 = C[0][c_col_off], C12 = C[0][c_col_off+1]
        # C21 = C[1][c_col_off], C22 = C[1][c_col_off+1]
        # W indices: c_col_off*2 + 0 = c_col_off, c_col_off*2 + 1 = c_col_off+1
        #             (c_col_off+1)*2 + 0 = 2*c_col_off+2, (c_col_off+1)*2 + 1 = 2*c_col_off+3
        
        # Wait, W[k*n+i] where k is col of C (0..4), i is row of C (0..1)
        # So W[k*2 + i] corresponds to C[i][k]
        
        # Let me redefine parameters more clearly:
        # a_col_off: column offset in A (0..2 for 2x4)
        # b_row_off: row offset in B (0..2 for 4x5)
        # b_col_off: column offset in B (0..3 for 4x5)
        # c_col_off: column offset in C (0..3 for 2x5)
        
        # A indices:
        A11_u = a_row_off  # This is wrong, let me use a_col_off
        # Actually, I need to pass a_col_off, not a_row_off. Let me fix the signature.
        
        return col_start  # Placeholder
    
    # Let me restart with a cleaner approach.
    
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(10)]
    
    col = 0
    
    def add_strassen(a_col_off, b_row_off, b_col_off, c_col_off):
        nonlocal col
        # A block: rows 0-1, cols a_col_off to a_col_off+1
        # B block: rows b_row_off to b_row_off+1, cols b_col_off to b_col_off+1
        # C block: rows 0-1, cols c_col_off to c_col_off+1
        
        # U indices for A:
        u_A11 = 0 * 4 + a_col_off       # A[0][a_col_off]
        u_A12 = 0 * 4 + (a_col_off + 1) # A[0][a_col_off+1]
        u_A21 = 1 * 4 + a_col_off       # A[1][a_col_off]
        u_A22 = 1 * 4 + (a_col_off + 1) # A[1][a_col_off+1]
        
        # V indices for B:
        v_B11 = b_row_off * 5 + b_col_off
        v_B12 = b_row_off * 5 + (b_col_off + 1)
        v_B21 = (b_row_off + 1) * 5 + b_col_off
        v_B22 = (b_row_off + 1) * 5 + (b_col_off + 1)
        
        # W indices for C:
        w_C11 = c_col_off * 2 + 0       # C[0][c_col_off]
        w_C12 = (c_col_off + 1) * 2 + 0 # C[0][c_col_off+1]
        w_C21 = c_col_off * 2 + 1       # C[1][c_col_off]
        w_C22 = (c_col_off + 1) * 2 + 1 # C[1][c_col_off+1]
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[u_A11][col] += 1
        U[u_A22][col] += 1
        V[v_B11][col] += 1
        V[v_B22][col] += 1
        W[w_C11][col] += 1
        W[w_C22][col] += -1
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[u_A21][col] += 1
        U[u_A22][col] += 1
        V[v_B11][col] += 1
        W[w_C21][col] += 1
        W[w_C22][col] += -1
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[u_A11][col] += 1
        V[v_B12][col] += 1
        V[v_B22][col] += -1
        W[w_C11][col] += 1
        W[w_C12][col] += 1
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[u_A22][col] += 1
        V[v_B21][col] += 1
        V[v_B11][col] += -1
        W[w_C11][col] += 1
        W[w_C21][col] += 1
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[u_A11][col] += 1
        U[u_A12][col] += 1
        V[v_B22][col] += 1
        W[w_C11][col] += -1
        W[w_C12][col] += 1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[u_A21][col] += 1
        U[u_A11][col] += -1
        V[v_B11][col] += 1
        V[v_B12][col] += 1
        W[w_C22][col] += 1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[u_A12][col] += 1
        U[u_A22][col] += -1
        V[v_B21][col] += 1
        V[v_B22][col] += 1
        W[w_C11][col] += 1
        W[w_C22][col] += 1
        col += 1
    
    def add_schoolbook_2x2x1(a_col_off, b_row_off, b_col_off, c_col_off):
        nonlocal col
        # A block: rows 0-1, cols a_col_off to a_col_off+1
        # B block: rows b_row_off to b_row_off+1, col b_col_off
        # C block: rows 0-1, col c_col_off
        
        u_A11 = 0 * 4 + a_col_off
        u_A12 = 0 * 4 + (a_col_off + 1)
        u_A21 = 1 * 4 + a_col_off
        u_A22 = 1 * 4 + (a_col_off + 1)
        
        v_B1 = b_row_off * 5 + b_col_off
        v_B2 = (b_row_off + 1) * 5 + b_col_off
        
        w_C1 = c_col_off * 2 + 0
        w_C2 = c_col_off * 2 + 1
        
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # Term 1: A11 * B1
        U[u_A11][col] += 1
        V[v_B1][col] += 1
        W[w_C1][col] += 1
        col += 1
        
        # Term 2: A12 * B2
        U[u_A12][col] += 1
        V[v_B2][col] += 1
        W[w_C1][col] += 1
        col += 1
        
        # Term 3: A21 * B1
        U[u_A21][col] += 1
        V[v_B1][col] += 1
        W[w_C2][col] += 1
        col += 1
        
        # Term 4: A22 * B2
        U[u_A22][col] += 1
        V[v_B2][col] += 1
        W[w_C2][col] += 1
        col += 1
    
    # Now build the decomposition:
    # C = A @ B where A is 2x4, B is 4x5
    # Split into: C = A_left @ B_left + A_right @ B_right
    # where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_left = B[0:2, :], B_right = B[2:4, :]
    
    # For A_left @ B_left (2x2 @ 2x5):
    # Split B_left into 3 parts: cols 0-1, cols 2-3, col 4
    # Use Strassen for 2x2x2 blocks, schoolbook for 2x2x1
    
    # Block 1: A_left (cols 0-1) @ B_left (rows 0-1, cols 0-1) -> C (cols 0-1)
    add_strassen(a_col_off=0, b_row_off=0, b_col_off=0, c_col_off=0)
    
    # Block 2: A_left (cols 0-1) @ B_left (rows 0-1, cols 2-3) -> C (cols 2-3)
    add_strassen(a_col_off=0, b_row_off=0, b_col_off=2, c_col_off=2)
    
    # Block 3: A_left (cols 0-1) @ B_left (rows 0-1, col 4) -> C (col 4)
    add_schoolbook_2x2x1(a_col_off=0, b_row_off=0, b_col_off=4, c_col_off=4)
    
    # For A_right @ B_right (2x2 @ 2x5):
    # Split B_right into 3 parts: cols 0-1, cols 2-3, col 4
    
    # Block 4: A_right (cols 2-3) @ B_right (rows 2-3, cols 0-1) -> C (cols 0-1)
    add_strassen(a_col_off=2, b_row_off=2, b_col_off=0, c_col_off=0)
    
    # Block 5: A_right (cols 2-3) @ B_right (rows 2-3, cols 2-3) -> C (cols 2-3)
    add_strassen(a_col_off=2, b_row_off=2, b_col_off=2, c_col_off=2)
    
    # Block 6: A_right (cols 2-3) @ B_right (rows 2-3, col 4) -> C (col 4)
    add_schoolbook_2x2x1(a_col_off=2, b_row_off=2, b_col_off=4, c_col_off=4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
