"""Matrix multiplication 2x4x7: structured decomposition using Strassen for 2x2x2 blocks."""
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    # W[k*n+i, c] convention: output element at row k, col i gets contribution from rank c

    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 14 rows

    col = 0

    def add_strassen_block(a_row_offset, b_row_offset, c_col_offset):
        """Add a Strassen 2x2x2 block.
        a_row_offset: starting row of A's 2x2 submatrix (0 or 2)
        b_row_offset: starting row of B's 2x2 submatrix (0, 2, or 4)
        c_col_offset: starting col of C's 2x2 submatrix in output (which row of B maps to)
        
        A submatrix: A[a_row_offset:a_row_offset+2, 0:2] and A[a_row_offset:a_row_offset+2, 2:4]
        Wait, let me reconsider the indexing.
        
        A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
        U index: i*m + j = i*4 + j
        
        B is 4x7: B[j,k] where j in {0,1,2,3}, k in {0,...,6}
        V index: j*p + k = j*7 + k
        
        C is 2x7: C[i,k] where i in {0,1}, k in {0,...,6}
        W index: k*n + i = k*2 + i
        
        For a 2x2x2 block:
        A part: 2 rows of A (all 4 cols? No, we need to split A's columns too)
        
        Actually, let me reconsider. The 4 columns of A need to be paired with the 4 rows of B.
        A is 2x4, B is 4x7. The multiplication is C = A*B.
        
        Let me split A's 4 columns into two groups of 2:
        - A_left = A[:, 0:2] (2x2)
        - A_right = A[:, 2:4] (2x2)
        
        And B's 4 rows (out of 7) into groups of 2:
        - B_top = B[0:2, :] (2x7)
        - B_mid = B[2:4, :] (2x7)
        
        Then C = A_left * B_top + A_right * B_mid + contributions from B rows 4,5,6
        
        Wait, that's not quite right either. Let me think more carefully.
        
        C[i,k] = sum over j=0 to 3 of A[i,j] * B[j,k]
        
        Split j into {0,1} and {2,3}:
        C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
        
        For the first sum, A[:, 0:2] is 2x2, B[0:2, :] is 2x7
        For the second sum, A[:, 2:4] is 2x2, B[2:4, :] is 2x7
        
        Now split B's 7 columns into {0,1}, {2,3}, {4,5}, {6}:
        
        Block (A[:,0:2] * B[0:2, :]): 2x2 x 2x7 = sum of 2x2x2 blocks for cols {0,1},{2,3},{4,5} plus 2x2x1 for col 6
        Block (A[:,2:4] * B[2:4, :]): same structure
        
        Total: 2 blocks * (3 * Strassen(7) + 1 * Standard(4)) = 2*(21+4) = 50
        """
        nonlocal col
        
        # A part: 2x2 submatrix from A
        # a_row_offset tells us which 2 columns of A: 0 for cols 0,1; 1 for cols 2,3
        # b_row_offset tells us which 2 rows of B: 0 for rows 0,1; 1 for rows 2,3
        
        # A[i, a_row_offset*2 + j] where i in {0,1}, j in {0,1}
        # U index: i*m + (a_row_offset*2 + j) = i*4 + a_row_offset*2 + j
        
        # B[(b_row_offset*2 + j), k] where j in {0,1}, k in {0,1}
        # V index: (b_row_offset*2 + j)*p + k = (b_row_offset*2 + j)*7 + k
        
        # C[i, c_col_offset*2 + k] where i in {0,1}, k in {0,1}
        # W index: (c_col_offset*2 + k)*n + i = (c_col_offset*2 + k)*2 + i
        
        # Strassen 7 products:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Where A00=A[i=0,j=0], A01=A[i=0,j=1], A10=A[i=1,j=0], A11=A[i=1,j=1]
        # And B00=B[j=0,k=0], B01=B[j=0,k=1], B10=B[j=1,k=0], B11=B[j=1,k=1]
        
        a_base = a_row_offset * 2  # 0 or 2
        b_base = b_row_offset * 2  # 0, 2, or 4
        c_base = c_col_offset * 2  # 0, 2, or 4
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: i=0, col=a_base+0 -> U index 0*4 + a_base+0 = a_base
        # A11: i=1, col=a_base+1 -> U index 1*4 + a_base+1 = 4+a_base+1
        # B00: j=b_base+0, k=c_base+0 -> V index (b_base+0)*7 + (c_base+0)
        # B11: j=b_base+1, k=c_base+1 -> V index (b_base+1)*7 + (c_base+1)
        # C00: i=0, k=c_base+0 -> W index (c_base+0)*2 + 0
        # C11: i=1, k=c_base+1 -> W index (c_base+1)*2 + 1
        
        U[a_base][col] += 1
        U[4 + a_base + 1][col] += 1
        V[(b_base + 0) * 7 + (c_base + 0)][col] += 1
        V[(b_base + 1) * 7 + (c_base + 1)][col] += 1
        W[(c_base + 0) * 2 + 0][col] += 1
        W[(c_base + 1) * 2 + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        # A10: i=1, col=a_base+0 -> U index 4+a_base
        # A11: i=1, col=a_base+1 -> U index 4+a_base+1
        # B00: j=b_base+0, k=c_base+0 -> V index (b_base+0)*7 + (c_base+0)
        # C20: i=1, k=c_base+0 -> W index (c_base+0)*2 + 1
        # C22: i=1, k=c_base+1 -> W index (c_base+1)*2 + 1
        
        U[4 + a_base][col] += 1
        U[4 + a_base + 1][col] += 1
        V[(b_base + 0) * 7 + (c_base + 0)][col] += 1
        W[(c_base + 0) * 2 + 1][col] += 1
        W[(c_base + 1) * 2 + 1][col] += 1
        col += 1
        
        # M3 = A00(B01 - B11)
        # A00: U index a_base
        # B01: j=b_base+0, k=c_base+1 -> V index (b_base+0)*7 + (c_base+1)
        # B11: j=b_base+1, k=c_base+1 -> V index (b_base+1)*7 + (c_base+1)
        # C01: i=0, k=c_base+1 -> W index (c_base+1)*2 + 0
        # C22: i=1, k=c_base+1 -> W index (c_base+1)*2 + 1
        
        U[a_base][col] += 1
        V[(b_base + 0) * 7 + (c_base + 1)][col] += 1
        V[(b_base + 1) * 7 + (c_base + 1)][col] -= 1
        W[(c_base + 1) * 2 + 0][col] += 1
        W[(c_base + 1) * 2 + 1][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        # A11: U index 4+a_base+1
        # B10: j=b_base+1, k=c_base+0 -> V index (b_base+1)*7 + (c_base+0)
        # B00: j=b_base+0, k=c_base+0 -> V index (b_base+0)*7 + (c_base+0)
        # C10: i=0, k=c_base+0 -> W index (c_base+0)*2 + 0
        # C22: i=1, k=c_base+1 -> W index (c_base+1)*2 + 1
        
        U[4 + a_base + 1][col] += 1
        V[(b_base + 1) * 7 + (c_base + 0)][col] += 1
        V[(b_base + 0) * 7 + (c_base + 0)][col] -= 1
        W[(c_base + 0) * 2 + 0][col] += 1
        W[(c_base + 1) * 2 + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        # A00: U index a_base
        # A01: i=0, col=a_base+1 -> U index a_base+1
        # B11: j=b_base+1, k=c_base+1 -> V index (b_base+1)*7 + (c_base+1)
        # C11: i=1, k=c_base+1 -> W index (c_base+1)*2 + 1
        
        U[a_base][col] += 1
        U[a_base + 1][col] += 1
        V[(b_base + 1) * 7 + (c_base + 1)][col] += 1
        W[(c_base + 1) * 2 + 1][col] -= 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10: U index 4+a_base
        # A00: U index a_base
        # B00: V index (b_base+0)*7 + (c_base+0)
        # B01: V index (b_base+0)*7 + (c_base+1)
        # C20: i=1, k=c_base+0 -> W index (c_base+0)*2 + 1
        # C22: i=1, k=c_base+1 -> W index (c_base+1)*2 + 1
        
        U[4 + a_base][col] += 1
        U[a_base][col] -= 1
        V[(b_base + 0) * 7 + (c_base + 0)][col] += 1
        V[(b_base + 0) * 7 + (c_base + 1)][col] += 1
        W[(c_base + 0) * 2 + 1][col] += 1
        W[(c_base + 1) * 2 + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01: U index a_base+1
        # A11: U index 4+a_base+1
        # B10: V index (b_base+1)*7 + (c_base+0)
        # B11: V index (b_base+1)*7 + (c_base+1)
        # C00: i=0, k=c_base+0 -> W index (c_base+0)*2 + 0
        # C01: i=0, k=c_base+1 -> W index (c_base+1)*2 + 0
        
        U[a_base + 1][col] += 1
        U[4 + a_base + 1][col] -= 1
        V[(b_base + 1) * 7 + (c_base + 0)][col] += 1
        V[(b_base + 1) * 7 + (c_base + 1)][col] += 1
        W[(c_base + 0) * 2 + 0][col] += 1
        W[(c_base + 1) * 2 + 0][col] -= 1
        col += 1

    def add_standard_block(a_row_offset, b_row_offset, c_col_offset):
        """Add a standard 2x2x1 block (4 products).
        a_row_offset: 0 or 1 (which 2 cols of A)
        b_row_offset: which row of B (single row, so 0,1,2,3 for the 2x2 blocks, or 4,5,6 for single)
        Wait, for 2x2x1, B is 2x1.
        c_col_offset: which single column of C
        """
        nonlocal col
        
        a_base = a_row_offset * 2
        b_base = b_row_offset  # single row index
        c_k = c_col_offset  # single column index
        
        # C[i, c_k] = A[i, a_base]*B[b_base, c_k] + A[i, a_base+1]*B[b_base+1, c_k]
        # 4 products:
        # P1 = A[i=0, col=a_base] * B[row=b_base, col=c_k]
        # P2 = A[i=0, col=a_base+1] * B[row=b_base+1, col=c_k]
        # P3 = A[i=1, col=a_base] * B[row=b_base, col=c_k]
        # P4 = A[i=1, col=a_base+1] * B[row=b_base+1, col=c_k]
        
        # P1: contributes to C[0, c_k] -> W index c_k*2 + 0
        U[a_base][col] += 1
        V[b_base * 7 + c_k][col] += 1
        W[c_k * 2 + 0][col] += 1
        col += 1
        
        # P2: contributes to C[0, c_k] -> W index c_k*2 + 0
        U[a_base + 1][col] += 1
        V[(b_base + 1) * 7 + c_k][col] += 1
        W[c_k * 2 + 0][col] += 1
        col += 1
        
        # P3: contributes to C[1, c_k] -> W index c_k*2 + 1
        U[4 + a_base][col] += 1
        V[b_base * 7 + c_k][col] += 1
        W[c_k * 2 + 1][col] += 1
        col += 1
        
        # P4: contributes to C[1, c_k] -> W index c_k*2 + 1
        U[4 + a_base + 1][col] += 1
        V[(b_base + 1) * 7 + c_k][col] += 1
        W[c_k * 2 + 1][col] += 1
        col += 1

    # Process 6 Strassen blocks: A[:, 0:2] * B[0:2, :] and A[:, 2:4] * B[2:4, :]
    # Each has 3 sub-blocks for cols {0,1}, {2,3}, {4,5}
    
    for a_off in range(2):  # a_off = 0 for A cols 0,1; a_off = 1 for A cols 2,3
        b_off = a_off * 2  # b_off = 0 for B rows 0,1; b_off = 2 for B rows 2,3
        for c_off in range(3):  # c_off = 0,1,2 for col pairs {0,1},{2,3},{4,5}
            add_strassen_block(a_off, b_off // 2, c_off)
    
    # Process 2 standard blocks for col 6: A[:, 0:2] * B[0:2, 6] and A[:, 2:4] * B[2:4, 6]
    for a_off in range(2):
        b_off = a_off * 2  # 0 or 2
        add_standard_block(a_off, b_off, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
