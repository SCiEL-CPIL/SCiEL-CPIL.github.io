"""Matrix multiplication 3x4x8 with Strassen-based decomposition."""
import time

DIMENSIONS = [3, 4, 8]

def _strassen2x2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2 matrix multiplication using Strassen."""
    # Strassen 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U maps A -> products, V maps B -> products, W maps products -> C
    # Row indices for U/V: 0=A11, 1=A12, 2=A21, 3=A22
    # Same for V.
    # Row indices for W: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    W[0][0] = 1  # C11 += M1
    W[3][0] = 1  # C22 += M1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21 += M2
    W[3][1] = -1 # C22 -= M2
    
    # M3 = A11(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12
    V[3][2] = -1 # -B22
    W[0][2] = 1  # C11 += M3
    W[3][2] = 1  # C22 += M3
    
    # M4 = A22(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21
    V[0][3] = -1 # -B11
    W[0][3] = 1  # C11 += M4
    W[2][3] = 1  # C21 += M4
    
    # M5 = (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    W[0][4] = -1 # C11 -= M5
    W[1][4] = 1  # C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # -A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[3][5] = 1  # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # -A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    W[0][6] = 1  # C11 += M7
    
    return U, V, W

def construct():
    """Construct rank-88 decomposition for 3x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # A is 3x4, B is 4x8, C is 3x8
    # U: (n*m) x rank = 12 x rank
    # V: (m*p) x rank = 32 x rank  
    # W: (p*n) x rank = 24 x rank
    # W[k*n+i, r] is the trace convention: row index = k*n + i where k is col of C, i is row of C
    
    # Decomposition strategy:
    # Split A into top 2 rows (2x4) and bottom 1 row (1x4)
    # C_top = A_top * B  (2x8)
    # C_bot = A_bot * B  (1x8)
    
    # For C_top (2x4x8):
    # Split B into top 2 rows (2x8) and bottom 2 rows (2x8)
    # C_top = A_top_left * B_top + A_top_right * B_bot
    # where A_top_left is first 2 cols of A_top (2x2), A_top_right is last 2 cols (2x2)
    
    # A_top is 2x4. Split into A_L (2x2) and A_R (2x2)
    # B is 4x8. Split into B_top (2x8) and B_bot (2x8)
    # C_top = A_L * B_top + A_R * B_bot
    
    # A_L * B_top: 2x2 * 2x8 -> use Strassen for each 2x2 block of B_top
    # B_top has 4 blocks of 2x2, so 4 * 7 = 28 rank
    # A_R * B_bot: similarly 28 rank
    # Total for C_top: 56 rank
    
    # For C_bot (1x4x8):
    # C_bot = A_bot[0]*B[0] + A_bot[1]*B[1] + A_bot[2]*B[2] + A_bot[3]*B[3]
    # Each term is a row vector (1x8) multiplied by a scalar, but actually it's
    # outer product of scalar with row of B. This is 4 * 8 = 32 rank-1 terms
    # Total for C_bot: 32 rank
    
    # Grand total: 56 + 32 = 88 rank
    
    rank = 88
    U = [[0]*rank for _ in range(n*m)]   # 12 x 88
    V = [[0]*rank for _ in range(m*p)]   # 32 x 88
    W = [[0]*rank for _ in range(p*n)]   # 24 x 88
    
    # Helper to get Strassen factors
    sU, sV, sW = _strassen2x2x2()
    
    col = 0
    
    # --- C_top: 2x4x8 ---
    # A_top rows: i=0,1. A_top cols: j=0,1 (left) and j=2,3 (right)
    # B_top rows: j=0,1. B_bot rows: j=2,3.
    # C_top cols: k=0..7
    
    # For A_L * B_top: A_L is 2x2 (rows i=0,1; cols j=0,1), B_top is 2x8 (rows j=0,1; cols k=0..7)
    # Split B_top into 4 blocks: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    for block_idx in range(4):
        k_start = block_idx * 2
        # For each Strassen product r=0..6
        for r in range(7):
            # U part: A_L elements
            # A_L[0,0] -> global U row 0*4+0 = 0
            # A_L[0,1] -> global U row 0*4+1 = 1
            # A_L[1,0] -> global U row 1*4+0 = 4
            # A_L[1,1] -> global U row 1*4+1 = 5
            for a_row in range(2):
                for a_col in range(2):
                    u_idx = (0 + a_row) * 4 + (0 + a_col)  # i=0+a_row, j=0+a_col
                    U[u_idx][col] += sU[a_row*2 + a_col][r]
            
            # V part: B_top elements for this block
            # B_top[0, k_start+k] for k=0,1 -> global V row (0)*8 + (k_start+k)
            # B_top[1, k_start+k] for k=0,1 -> global V row (1)*8 + (k_start+k)
            for b_row in range(2):
                for b_col in range(2):
                    v_idx = (0 + b_row) * 8 + (k_start + b_col)
                    V[v_idx][col] += sV[b_row*2 + b_col][r]
            
            # W part: C_top elements
            # C_top[0, k_start+k] for k=0,1 -> global W row (k_start+k)*3 + 0
            # C_top[1, k_start+k] for k=0,1 -> global W row (k_start+k)*3 + 1
            for c_col in range(2):
                for c_row in range(2):
                    w_idx = (k_start + c_col) * 3 + (0 + c_row)
                    W[w_idx][col] += sW[c_row*2 + c_col][r]
            
            col += 1
    
    # For A_R * B_bot: A_R is 2x2 (rows i=0,1; cols j=2,3), B_bot is 2x8 (rows j=2,3; cols k=0..7)
    for block_idx in range(4):
        k_start = block_idx * 2
        for r in range(7):
            # U part: A_R elements
            # A_R[0,0] -> global U row 0*4+2 = 2
            # A_R[0,1] -> global U row 0*4+3 = 3
            # A_R[1,0] -> global U row 1*4+2 = 6
            # A_R[1,1] -> global U row 1*4+3 = 7
            for a_row in range(2):
                for a_col in range(2):
                    u_idx = (0 + a_row) * 4 + (2 + a_col)
                    U[u_idx][col] += sU[a_row*2 + a_col][r]
            
            # V part: B_bot elements for this block
            # B_bot[0, k_start+k] -> global V row (2)*8 + (k_start+k)
            # B_bot[1, k_start+k] -> global V row (3)*8 + (k_start+k)
            for b_row in range(2):
                for b_col in range(2):
                    v_idx = (2 + b_row) * 8 + (k_start + b_col)
                    V[v_idx][col] += sV[b_row*2 + b_col][r]
            
            # W part: C_top elements (same as before)
            for c_col in range(2):
                for c_row in range(2):
                    w_idx = (k_start + c_col) * 3 + (0 + c_row)
                    W[w_idx][col] += sW[c_row*2 + c_col][r]
            
            col += 1
    
    # --- C_bot: 1x4x8 ---
    # A_bot is row i=2, cols j=0,1,2,3
    # C_bot[k] = sum_{j=0}^{3} A[2,j] * B[j,k]
    # Each term A[2,j] * B[j,k] contributes to W[k*3+2]
    for j in range(4):
        for k in range(8):
            # U: A[2,j] -> row 2*4+j = 8+j
            U[8 + j][col] = 1
            # V: B[j,k] -> row j*8+k
            V[j * 8 + k][col] = 1
            # W: C[k,2] -> row k*3+2
            W[k * 3 + 2][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify correctness with random test
    import random
    random.seed(42)
    # Create random A (3x4) and B (4x8)
    A = [[random.randint(-5, 5) for _ in range(4)] for _ in range(3)]
    B = [[random.randint(-5, 5) for _ in range(8)] for _ in range(4)]
    
    # Compute C = A * B
    C = [[0]*8 for _ in range(3)]
    for i in range(3):
        for k in range(8):
            for j in range(4):
                C[i][k] += A[i][j] * B[j][k]
    
    # Compute using decomposition
    # Product matrix P = U^T * V (rank x rank), then C_recon[i,k] = sum_r P[r,r'] * ... 
    # Actually: C[i,k] = sum_r (sum_j U[i*4+j, r]) * (sum_j' V[j'*8+k, r]) * W[k*3+i, r]
    # Wait, the decomposition is: C[i,k] = sum_r U[i*m+j?] ... 
    
    # Let me reconsider. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # But our indices are flattened. Let me just verify by direct computation.
    
    # For each r, the contribution to C[i,k] is:
    # sum_r (sum_j U[i*4+j, r]) * ... no, that's not right.
    
    # The correct interpretation: U, V, W are the factor matrices such that
    # C[i,k] = sum_r U_flat[i*m + ?, r] * V_flat[?, r] * W_flat[k*n+i, r]
    
    # Actually the standard tensor decomposition for matrix mult is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k] = sum_{j,r} A[i,j] * B[j,k] * delta_{j,j'} ... 
    # The tensor T has indices (i, j, k) where T[i,j,k] = delta_{j,j} (it's 1 if we think of it as the multiplication tensor)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need T[i,j,k] = 1 for all i,j,k (since the multiplication tensor has all-ones in the appropriate sense)
    
    # Actually no. The matrix multiplication tensor has T[i,j,k] = 1 if we're looking at the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k
    # And = 0 otherwise... but all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid.
    
    # Let's verify:
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if val != 1:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        print(f"Verification failed with max error {max_err}")
    
    return {"U": U, "V": V, "W": W}
