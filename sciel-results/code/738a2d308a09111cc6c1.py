import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 7 rank-1 terms corresponding to a Strassen decomposition of a 2x2x2 block.
    
    a_rows, a_cols: indices for the 2x2 A submatrix
    b_rows, b_cols: indices for the 2x2 B submatrix
    c_rows, c_cols: indices for the 2x2 C submatrix
    
    Returns the next column index.
    """
    # Local indices
    # A block:
    # a = A[a_rows[0], a_cols[0]]  -> u_a
    # b = A[a_rows[0], a_cols[1]]  -> u_b
    # c = A[a_rows[1], a_cols[0]]  -> u_c
    # d = A[a_rows[1], a_cols[1]]  -> u_d
    
    u_a = a_rows[0] * m + a_cols[0]
    u_b = a_rows[0] * m + a_cols[1]
    u_c = a_rows[1] * m + a_cols[0]
    u_d = a_rows[1] * m + a_cols[1]
    
    # B block:
    # e = B[b_rows[0], b_cols[0]]  -> v_e
    # f = B[b_rows[0], b_cols[1]]  -> v_f
    # g = B[b_rows[1], b_cols[0]]  -> v_g
    # h = B[b_rows[1], b_cols[1]]  -> v_h
    
    v_e = b_rows[0] * p + b_cols[0]
    v_f = b_rows[0] * p + b_cols[1]
    v_g = b_rows[1] * p + b_cols[0]
    v_h = b_rows[1] * p + b_cols[1]
    
    # C block (output):
    # C[c_rows[0], c_cols[0]] -> w_00
    # C[c_rows[0], c_cols[1]] -> w_01
    # C[c_rows[1], c_cols[0]] -> w_10
    # C[c_rows[1], c_cols[1]] -> w_11
    
    # Note: The task specifies W[k*n+i, r] where k is the column index of C (which corresponds to b_cols in the multiplication A*B=C)
    # and i is the row index of C (which corresponds to a_rows).
    # So W index is (c_col * n + c_row).
    
    w_00 = c_cols[0] * n + c_rows[0]
    w_01 = c_cols[1] * n + c_rows[0]
    w_10 = c_cols[0] * n + c_rows[1]
    w_11 = c_cols[1] * n + c_rows[1]
    
    # Strassen Factors:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We add 7 columns.
    
    # M1
    U[u_a, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M2
    U[u_c, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    W[w_10, col] += 1.0
    W[w_11, col] += -1.0
    col += 1
    
    # M3
    U[u_a, col] += 1.0
    V[v_f, col] += 1.0
    V[v_h, col] += -1.0
    W[w_01, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M4
    U[u_d, col] += 1.0
    V[v_g, col] += 1.0
    V[v_e, col] += -1.0
    W[w_00, col] += 1.0
    W[w_10, col] += 1.0
    col += 1
    
    # M5
    U[u_a, col] += 1.0
    U[u_b, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += -1.0
    W[w_01, col] += 1.0
    col += 1
    
    # M6
    U[u_c, col] += 1.0
    U[u_a, col] += -1.0
    V[v_e, col] += 1.0
    V[v_f, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M7
    U[u_b, col] += 1.0
    U[u_d, col] += -1.0
    V[v_g, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    col += 1
    
    return col

def _schoolbook_decomposition(n, m, p):
    """Generates the standard n*m*p term decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Strategy:
    # n=3, m=4, p=8
    # We can decompose A (3x4) into a 2x2 block (rows 0-1, cols 0-1), a 2x2 block (rows 0-1, cols 2-3),
    # and a 1x4 block (row 2, cols 0-3).
    # 
    # Block 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # Block 2: A[0:2, 2:4] * B[2:4, :] -> C[0:2, :]
    # Block 3: A[2:3, 0:4] * B[0:4, :] -> C[2:3, :]
    #
    # For Block 1 and 2, we can use Strassen on the 2x2 A part.
    # B is 4x8. We split B into 2x2 blocks along the columns (k dimension).
    # Since p=8, we have 4 blocks of size 2x2 in the k-dimension for each of the 4 rows of B?
    # No, Strassen applies to the inner dimension m (which is 4 here, split into 2+2) and output dimension p (which is 8).
    # Wait, the standard Strassen applies to 2x2 matrix multiplication.
    # Here A is 3x4, B is 4x8.
    # We can view this as:
    # A = [A11 A12]  (2x2, 2x2)
    #     [A21 A22]  (1x2, 1x2)
    #
    # B is 4x8.
    # Let's split the middle dimension (4) into two halves of size 2.
    # B_top = B[0:2, :] (2x8)
    # B_bot = B[2:4, :] (2x8)
    #
    # C[0:2, :] = A11 * B_top + A12 * B_bot
    # C[2:3, :] = A21 * B_top + A22 * B_bot
    #
    # We can apply Strassen to the multiplication of the 2x2 A block [A11 A12; A21 A22] with the "stacked" B?
    # No, Strassen is for C = A*B where A, B, C are all 2x2.
    # Here the "C" for the top part is 2x8.
    # We can process the k-dimension (size 4) in chunks of 2.
    # There are 4 such chunks (0-1, 2-3, 4-5, 6-7) for the B matrix? No, B has 4 rows.
    # The inner product sum is over the 4 columns of A / 4 rows of B.
    # Let's split the inner dimension (size 4) into two blocks of size 2.
    # Inner block 1: cols 0-1 of A, rows 0-1 of B.
    # Inner block 2: cols 2-3 of A, rows 2-3 of B.
    #
    # So C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2:4] * B[2:4, :]
    # C[2:3, :] = A[2:3, 0:2] * B[0:2, :] + A[2:3, 2:4] * B[2:4, :]
    #
    # Now, look at the term A[0:2, 0:2] * B[0:2, :].
    # A_sub is 2x2. B_sub is 2x8.
    # We can apply Strassen-like logic?
    # Strassen reduces 8 multiplications to 7 for 2x2x2.
    # Here we have 2x2x8.
    # We can treat B_sub as four 2x2 blocks (along the column dimension of C, which is size 8, so 4 blocks of width 2).
    # Let B_sub = [B1, B2, B3, B4] where each Bi is 2x2.
    # Then C_sub = A_sub * B_sub = [A_sub*B1, A_sub*B2, A_sub*B3, A_sub*B4].
    # Each A_sub*Bi is a 2x2x2 multiplication.
    # Using Strassen, each takes 7 rank units.
    # So A_sub * B_sub takes 4 * 7 = 28 rank units.
    #
    # Similarly, A[0:2, 2:4] * B[2:4, :] takes 28 rank units.
    #
    # Total for C[0:2, :] is 28 + 28 = 56 rank units.
    #
    # Now for C[2:3, :].
    # A[2:3, 0:2] is 1x2. B[0:2, :] is 2x8.
    # A[2:3, 2:4] is 1x2. B[2:4, :] is 2x8.
    #
    # C[2, :] = A[2, 0:2] * B[0:2, :] + A[2, 2:4] * B[2:4, :]
    #
    # Let's look at A[2, 0:2] * B[0:2, :].
    # A_row is 1x2. B_sub is 2x8.
    # This is a 1x2x8 multiplication.
    # Schoolbook rank for 1x2x8 is 1*2*8 = 16.
    # Can we do better?
    # For 1x2x2, rank is 2 (schoolbook). Strassen doesn't help for 1x2x2 (rank 2 is minimal).
    # For 1x2x8, it's just 4 independent 1x2x2 multiplications.
    # So rank is 4 * 2 = 8.
    #
    # So A[2, 0:2] * B[0:2, :] takes 8 rank units.
    # A[2, 2:4] * B[2:4, :] takes 8 rank units.
    #
    # Total for C[2:3, :] is 8 + 8 = 16 rank units.
    #
    # Total Rank = 56 + 16 = 72.
    #
    # Wait, is this correct?
    # Let's verify the Strassen part.
    # A_sub (2x2) * B_block (2x2).
    # Strassen uses 7 multiplications.
    # Yes, rank 7 for 2x2x2.
    # There are 4 such blocks in B[0:2, :] (width 8, block width 2).
    # So 4 * 7 = 28.
    #
    # So the total rank should be 72.
    #
    # Let's implement this.

    rank = 72
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2:4] * B[2:4, :]
    
    # 1a: A[0:2, 0:2] * B[0:2, :]
    # A rows: 0,1. A cols: 0,1.
    # B rows: 0,1. B cols: 0..7.
    # C rows: 0,1. C cols: 0..7.
    
    for k_start in range(0, 8, 2):
        # B block cols: k_start, k_start+1
        # C block cols: k_start, k_start+1
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
        
    # 1b: A[0:2, 2:4] * B[2:4, :]
    # A rows: 0,1. A cols: 2,3.
    # B rows: 2,3. B cols: 0..7.
    # C rows: 0,1. C cols: 0..7.
    
    for k_start in range(0, 8, 2):
        # B block cols: k_start, k_start+1
        # C block cols: k_start, k_start+1
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
        
    # Part 2: C[2:3, :] = A[2:3, 0:2] * B[0:2, :] + A[2:3, 2:4] * B[2:4, :]
    
    # 2a: A[2:3, 0:2] * B[0:2, :]
    # A row: 2. A cols: 0,1.
    # B rows: 0,1. B cols: 0..7.
    # C row: 2. C cols: 0..7.
    # This is 1x2x8.
    # We can do this with schoolbook:
    # For each k in 0..7:
    #   Sum over j in 0..1 of A[2, j] * B[j, k]
    #   This is 2 terms per k.
    #   Total 8 * 2 = 16 terms?
    #   Wait.
    #   C[2, k] = A[2, 0]*B[0, k] + A[2, 1]*B[1, k]
    #   Term 1: U[2*m+0]=1, V[0*p+k]=1, W[k*n+2]=1
    #   Term 2: U[2*m+1]=1, V[1*p+k]=1, W[k*n+2]=1
    #   So 2 terms per k. 8 k's. Total 16 terms.
    
    for k in range(8):
        # Term 1: j=0
        U[2 * m + 0, col] = 1.0
        V[0 * p + k, col] = 1.0
        W[k * n + 2, col] = 1.0
        col += 1
        # Term 2: j=1
        U[2 * m + 1, col] = 1.0
        V[1 * p + k, col] = 1.0
        W[k * n + 2, col] = 1.0
        col += 1
        
    # 2b: A[2:3, 2:4] * B[2:4, :]
    # A row: 2. A cols: 2,3.
    # B rows: 2,3. B cols: 0..7.
    # C row: 2. C cols: 0..7.
    
    for k in range(8):
        # Term 1: j=2
        U[2 * m + 2, col] = 1.0
        V[2 * p + k, col] = 1.0
        W[k * n + 2, col] = 1.0
        col += 1
        # Term 2: j=3
        U[2 * m + 3, col] = 1.0
        V[3 * p + k, col] = 1.0
        W[k * n + 2, col] = 1.0
        col += 1
        
    # Check if we used exactly 72 columns
    if col != rank:
        # If col < rank, we have extra zeros. We can trim.
        # If col > rank, we have an error.
        if col < rank:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
            rank = col
        else:
            # This should not happen
            raise Exception(f"Rank overflow: expected {rank}, got {col}")

    # Verify correctness with random matrices
    # A (3x4), B (4x8)
    # C = A * B
    # Decomposition: C = U * (V * W) ? No.
    # Tensor decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Here:
    # U index: i*m + j (i=0..2, j=0..3) -> 3*4=12
    # V index: j*p + k (j=0..3, k=0..7) -> 4*8=32
    # W index: k*n + i (k=0..7, i=0..2) -> 8*3=24
    
    # Let's verify one entry.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Our decomposition computes:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This sums over r.
    # For a fixed i, j, k, the term is included if U[i*m+j, r] != 0 and V[j*p+k, r] != 0 and W[k*n+i, r] != 0.
    
    # Let's do a quick check with numpy.
    A_test = np.random.rand(3, 4)
    B_test = np.random.rand(4, 8)
    C_test = A_test @ B_test
    
    # Reconstruct C from U, V, W
    C_recon = np.zeros((3, 8))
    for i in range(3):
        for k in range(8):
            val = 0.0
            for j in range(4):
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_recon[i, k] = val
            
    if not np.allclose(C_test, C_recon, atol=1e-9):
        # If it fails, fall back to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        rank = n*m*p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
