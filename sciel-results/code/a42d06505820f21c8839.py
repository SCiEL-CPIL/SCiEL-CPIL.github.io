"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses a block decomposition with Strassen for 2x2 blocks and schoolbook for others.
    Total rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m + j, c] = coefficient for A[i,j] in rank-1 term c
    # V[j*p + k, c] = coefficient for B[j,k] in rank-1 term c
    # W[k*n + i, c] = coefficient for C[i,k] in rank-1 term c
    
    # Block structure:
    # A is 4x5, B is 5x5, C is 4x5
    # Let's partition A into rows and columns, B into rows and columns
    # A: 4x5, B: 5x5, C: 4x5
    # 
    # Strategy: Partition the 5 dimension of A's columns and B's rows into blocks.
    # Let's try partitioning the 5 into 2+3.
    # A columns: [0,1] and [2,3,4]
    # B rows: [0,1] and [2,3,4]
    # B columns: [0,1] and [2,3,4]
    # C columns: [0,1] and [2,3,4]
    # A rows: partition 4 into 2+2
    
    # So we have blocks:
    # A rows: [0,1] (block 0), [2,3] (block 1)
    # A cols / B rows: [0,1] (block 0), [2,3,4] (block 1)
    # B cols / C cols: [0,1] (block 0), [2,3,4] (block 1)
    # C rows: [0,1] (block 0), [2,3] (block 1)
    
    # Block products:
    # (0,0): A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2] : 2x2 @ 2x2 -> 2x2, use Strassen, rank 7
    # (0,1): A[0:2,0:2] @ B[0:2,2:5] -> C[0:2,2:5] : 2x2 @ 2x3 -> 2x3, schoolbook, rank 12
    # (1,0): A[0:2,2:5] @ B[2:5,0:2] -> C[0:2,0:2] : 2x3 @ 3x2 -> 2x2, schoolbook, rank 12
    # (1,1): A[0:2,2:5] @ B[2:5,2:5] -> C[0:2,2:5] : 2x3 @ 3x3 -> 2x3, schoolbook, rank 18
    # (2,0): A[2:4,0:2] @ B[0:2,0:2] -> C[2:4,0:2] : 2x2 @ 2x2 -> 2x2, use Strassen, rank 7
    # (2,1): A[2:4,0:2] @ B[0:2,2:5] -> C[2:4,2:5] : 2x2 @ 2x3 -> 2x3, schoolbook, rank 12
    # (3,0): A[2:4,2:5] @ B[2:5,0:2] -> C[2:4,0:2] : 2x3 @ 3x2 -> 2x2, schoolbook, rank 12
    # (3,1): A[2:4,2:5] @ B[2:5,2:5] -> C[2:4,2:5] : 2x3 @ 3x3 -> 2x3, schoolbook, rank 18
    
    # Total rank = 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: schoolbook for R x C @ C x S -> R x S
    # a_row_start, a_col_start: start indices in A
    # b_row_start, b_col_start: start indices in B
    # c_row_start, c_col_start: start indices in C
    # R, C, S: block dimensions
    def schoolbook(a_row_start, a_col_start, b_row_start, b_col_start, 
                   c_row_start, c_col_start, R, C, S):
        nonlocal col
        for i in range(R):
            for j in range(C):
                for k in range(S):
                    a_idx = (a_row_start + i) * m + (a_col_start + j)
                    b_idx = (b_row_start + j) * p + (b_col_start + k)
                    c_idx = (c_col_start + k) * n + (c_row_start + i)
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1
    
    # Helper: Strassen for 2x2 @ 2x2 -> 2x2
    def strassen(a_row_start, a_col_start, b_row_start, b_col_start, 
                 c_row_start, c_col_start):
        nonlocal col
        # A indices: A[ar, ac] where ar in {a_row_start, a_row_start+1}, ac in {a_col_start, a_col_start+1}
        # B indices: B[br, bc] where br in {b_row_start, b_row_start+1}, bc in {b_col_start, b_col_start+1}
        # C indices: C[cr, cc] where cr in {c_row_start, c_row_start+1}, cc in {c_col_start, c_col_start+1}
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00 = A[ar0, ac0], A11 = A[ar1, ac1]
        # B00 = B[br0, bc0], B11 = B[br1, bc1]
        # C: C00 += M1, C01 -= M1, C11 += M1
        U[(a_row_start) * m + a_col_start][col] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = 1
        V[(b_row_start) * p + b_col_start][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        W[(c_col_start) * n + c_row_start][col] = 1          # C00
        W[(c_col_start + 1) * n + c_row_start][col] = -1      # C01
        W[(c_col_start) * n + (c_row_start + 1)][col] = 0     # C10 (not involved)
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1 # C11
        col += 1
        
        # M2 = (A10 + A11)B00
        # A10 = A[ar1, ac0], A11 = A[ar1, ac1]
        # B00 = B[br0, bc0]
        # C: C10 -= M2, C11 += M2
        U[(a_row_start + 1) * m + a_col_start][col] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = 1
        V[(b_row_start) * p + b_col_start][col] = 1
        W[(c_col_start) * n + (c_row_start + 1)][col] = -1   # C10
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = 1 # C11
        col += 1
        
        # M3 = A00(B01 - B11)
        # A00 = A[ar0, ac0]
        # B01 = B[br0, bc1], B11 = B[br1, bc1]
        # C: C10 += M3, C11 -= M3
        U[(a_row_start) * m + a_col_start][col] = 1
        V[(b_row_start) * p + (b_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = -1
        W[(c_col_start) * n + (c_row_start + 1)][col] = 1   # C10
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = -1 # C11
        col += 1
        
        # M4 = A11(B10 - B00)
        # A11 = A[ar1, ac1]
        # B10 = B[br1, bc0], B00 = B[br0, bc0]
        # C: C00 += M4, C11 -= M4
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + b_col_start][col] = 1
        V[(b_row_start) * p + b_col_start][col] = -1
        W[(c_col_start) * n + c_row_start][col] = 1          # C00
        W[(c_col_start + 1) * n + (c_row_start + 1)][col] = -1 # C11
        col += 1
        
        # M5 = (A00 + A01)B11
        # A00 = A[ar0, ac0], A01 = A[ar0, ac1]
        # B11 = B[br1, bc1]
        # C: C00 -= M5, C01 += M5
        U[(a_row_start) * m + a_col_start][col] = 1
        U[(a_row_start) * m + (a_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        W[(c_col_start) * n + c_row_start][col] = -1          # C00
        W[(c_col_start + 1) * n + c_row_start][col] = 1      # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10 = A[ar1, ac0], A00 = A[ar0, ac0]
        # B00 = B[br0, bc0], B01 = B[br0, bc1]
        # C: C01 += M6
        U[(a_row_start + 1) * m + a_col_start][col] = 1
        U[(a_row_start) * m + a_col_start][col] = -1
        V[(b_row_start) * p + b_col_start][col] = 1
        V[(b_row_start) * p + (b_col_start + 1)][col] = 1
        W[(c_col_start + 1) * n + c_row_start][col] = 1      # C01
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01 = A[ar0, ac1], A11 = A[ar1, ac1]
        # B10 = B[br1, bc0], B11 = B[br1, bc1]
        # C: C00 += M7, C10 += M7
        U[(a_row_start) * m + (a_col_start + 1)][col] = 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col] = -1
        V[(b_row_start + 1) * p + b_col_start][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        W[(c_col_start) * n + c_row_start][col] = 1          # C00
        W[(c_col_start) * n + (c_row_start + 1)][col] = 1   # C10
        col += 1
    
    # Block (0,0): A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    strassen(0, 0, 0, 0, 0, 0)
    
    # Block (0,1): A[0:2,0:2] @ B[0:2,2:5] -> C[0:2,2:5]
    schoolbook(0, 0, 0, 2, 0, 2, 2, 2, 3)
    
    # Block (1,0): A[0:2,2:5] @ B[2:5,0:2] -> C[0:2,0:2]
    schoolbook(0, 2, 2, 0, 0, 0, 2, 3, 2)
    
    # Block (1,1): A[0:2,2:5] @ B[2:5,2:5] -> C[0:2,2:5]
    schoolbook(0, 2, 2, 2, 0, 2, 2, 3, 3)
    
    # Block (2,0): A[2:4,0:2] @ B[0:2,0:2] -> C[2:4,0:2]
    strassen(2, 0, 0, 0, 2, 0)
    
    # Block (2,1): A[2:4,0:2] @ B[0:2,2:5] -> C[2:4,2:5]
    schoolbook(2, 0, 0, 2, 2, 2, 2, 2, 3)
    
    # Block (3,0): A[2:4,2:5] @ B[2:5,0:2] -> C[2:4,0:2]
    schoolbook(2, 2, 2, 0, 2, 0, 2, 3, 2)
    
    # Block (3,1): A[2:4,2:5] @ B[2:5,2:5] -> C[2:4,2:5]
    schoolbook(2, 2, 2, 2, 2, 2, 2, 3, 3)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
