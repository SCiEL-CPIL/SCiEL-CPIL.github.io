import time

DIMENSIONS = [2, 3, 7]

def _hybrid_decomposition():
    """
    Hybrid Strassen/Schoolbook decomposition for 2x3x7.
    
    Split A (2x3) = [A1 (2x2) | A2 (2x1)]
    Split B (3x7) = [B1 (2x7); B2 (1x7)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1: split B1 into 3 blocks of 2 cols + 1 col
      - 3 blocks: Strassen 2x2x2 (rank 7 each) = 21
      - 1 block: schoolbook 2x2x1 (rank 2) = 2
    For A2*B2: schoolbook 2x1x7 (rank 14)
    
    Total rank: 21 + 2 + 14 = 37
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank = 37
    U = [[0] * rank for _ in range(n * m)]  # (6, 37)
    V = [[0] * rank for _ in range(m * p)]  # (21, 37)
    W = [[0] * rank for _ in range(p * n)]  # (14, 37)
    
    col = 0
    
    # Part 1: A1*B1 where A1 is rows 0-1, cols 0-1 of A; B1 is rows 0-1, all 7 cols of B
    # Split B1 columns into blocks: [0,1], [2,3], [4,5], [6]
    # For each 2-col block, use Strassen (7 rank-1 terms)
    # For the single col, use schoolbook (2 rank-1 terms)
    
    def add_strassen_2x2x2(A_row_start, B_row_start, B_col_start, W_row_start):
        """Add Strassen decomposition for 2x2x2 block.
        A rows: A_row_start, A_row_start+1
        A cols: 0, 1 (the 2x2 part)
        B rows: B_row_start, B_row_start+1
        B cols: B_col_start, B_col_start+1
        W rows: W_row_start, W_row_start+1 (output rows i=0,1)
        """
        nonlocal col
        # Strassen intermediate values:
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        #
        # Output:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        #
        # U index: i*m + j where i is A row, j is A col
        # V index: j*p + k where j is B row, k is B col
        # W index: k*n + i where k is output col, i is output row
        
        a00 = A_row_start * m + 0
        a01 = A_row_start * m + 1
        a10 = (A_row_start + 1) * m + 0
        a11 = (A_row_start + 1) * m + 1
        
        b00 = B_row_start * p + B_col_start
        b01 = B_row_start * p + (B_col_start + 1)
        b10 = (B_row_start + 1) * p + B_col_start
        b11 = (B_row_start + 1) * p + (B_col_start + 1)
        
        w00 = (B_col_start) * n + 0
        w01 = (B_col_start + 1) * n + 0
        w10 = (B_col_start) * n + 1
        w11 = (B_col_start + 1) * n + 1
        
        # M1: (a00+a11)*(b00+b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        
        # M2: (a10+a11)*b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M3: a00*(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M4: a11*(b10-b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        
        # M5: (a00+a01)*b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1
        
        # M6: (a10-a00)*(b00+b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7: (a01-a11)*(b10+b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[w00][col] = 1
        col += 1
        
    # 3 Strassen blocks for B1 cols [0,1], [2,3], [4,5]
    for block in range(3):
        add_strassen_2x2x2(0, 0, block * 2, 0)
    
    # Schoolbook for B1 col [6]: 2x2x1
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    a00 = 0 * m + 0
    a01 = 0 * m + 1
    a10 = 1 * m + 0
    a11 = 1 * m + 1
    b06 = 0 * p + 6
    b16 = 1 * p + 6
    w06 = 6 * n + 0
    w16 = 6 * n + 1
    
    # Term 1: A[0,0]*B[0,6] -> C[0,6]
    U[a00][col] = 1
    V[b06][col] = 1
    W[w06][col] = 1
    col += 1
    
    # Term 2: A[0,1]*B[1,6] -> C[0,6]
    U[a01][col] = 1
    V[b16][col] = 1
    W[w06][col] = 1
    col += 1
    
    # Term 3: A[1,0]*B[0,6] -> C[1,6]
    U[a10][col] = 1
    V[b06][col] = 1
    W[w16][col] = 1
    col += 1
    
    # Term 4: A[1,1]*B[1,6] -> C[1,6]
    U[a11][col] = 1
    V[b16][col] = 1
    W[w16][col] = 1
    col += 1
    
    # Part 2: A2*B2 where A2 is 2x1 (col 2 of A), B2 is 1x7 (row 2 of B)
    # C[i,k] += A[i,2] * B[2,k]
    # Schoolbook: 2*7 = 14 terms
    for i in range(2):
        for k in range(7):
            a_idx = i * m + 2  # A[i,2]
            b_idx = 2 * p + k  # B[2,k]
            w_idx = k * n + i  # W[k*n+i]
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _check_decomposition(result):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # The tensor identity: for all i, j, k,
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is because the matrix multiplication tensor T has T[i,j,k] = 1
    # (each term A[i,j]*B[j,k] contributes to C[i,k])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Try the hybrid Strassen/Schoolbook decomposition (rank 37)
    result = _hybrid_decomposition()
    
    if _check_decomposition(result):
        return result
    
    # Fallback to schoolbook
    result = _schoolbook()
    if _check_decomposition(result):
        return result
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
