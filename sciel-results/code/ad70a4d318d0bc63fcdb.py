"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x10 matrix.
    
    Strategy: Use Strassen's algorithm on the 2x2 block structure.
    The 2x2x2 matrix multiplication tensor has rank 7 (Strassen) instead of 8 (schoolbook).
    
    For a 2x3 matrix A and 3x10 matrix B:
    - Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where A11, A12, A21, A22 are 1x1 or 1x2?
      Actually, let's think of it as:
      A is 2x3, B is 3x10.
      We can view this as multiplying [A_left | A_right] where A_left is 2x2 and A_right is 2x1,
      by [B_top; B_bottom] where B_top is 2x10 and B_bottom is 1x10.
      
    Better approach: Partition the 3 columns of A and 3 rows of B into:
    - First 2 columns/rows: use Strassen (rank 7 per 2x2x2 block)
    - Last 1 column/row: schoolbook (rank 4 for 2x1x1... wait, 2*1*1 = 2)
    
    Actually, let's partition differently. The tensor is for (2,3,10).
    We can split the middle dimension (3) into (2,1).
    - For the (2,2) part: we have a 2x2 matrix times 2x10 matrix.
      This can be seen as 5 independent 2x2x2 multiplications (since output is 2x10 = 2x(5*2)).
      Each 2x2x2 block uses Strassen with rank 7.
      Total for this part: 5 * 7 = 35.
    - For the (2,1) part: we have a 2x1 matrix times 1x10 matrix.
      This is 2*1*10 = 20 scalar multiplications in schoolbook.
      But we can also think of it as 10 independent 2x1x1 multiplications.
      Each 2x1x1 has rank 2 (since it's just outer product-like).
      Total for this part: 10 * 2 = 20.
    
    Total rank: 35 + 20 = 55.
    
    Let's verify the structure:
    A is 2x3: A[i, j] for i in {0,1}, j in {0,1,2}
    B is 3x10: B[j, k] for j in {0,1,2}, k in {0,...,9}
    C is 2x10: C[i, k] = sum_j A[i,j] * B[j,k]
    
    Partition j into j in {0,1} (Strassen part) and j=2 (schoolbook part).
    
    For j in {0,1}: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a 2x2x10 multiplication, which decomposes into 5 independent 2x2x2 multiplications
    (for k in {0,1}, {2,3}, {4,5}, {6,7}, {8,9}).
    
    For each 2x2x2 block (say for columns 2t, 2t+1 of B):
    We compute:
      C[i, 2t]   += A[i,0]*B[0,2t]   + A[i,1]*B[1,2t]
      C[i, 2t+1] += A[i,0]*B[0,2t+1] + A[i,1]*B[1,2t+1]
    
    Using Strassen for 2x2x2:
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    M2 = (A[1,0] + A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[1,0] - B[1,1])
    M4 = A[1,1] * (B[0,1] - B[0,0])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[1,1] - A[0,1]) * (B[0,1] + B[1,1])
    
    Then:
    C[0,0] = M1 + M4 - M5 + M7
    C[1,0] = M2 + M3 + M5 - M6
    C[0,1] = M1 + M3
    C[1,1] = M2 + M4
    
    For the schoolbook part (j=2):
    C[i, k] += A[i,2] * B[2,k] for i in {0,1}, k in {0,...,9}
    This is 2*10 = 20 terms.
    
    Total rank = 5*7 + 20 = 35 + 20 = 55.
    
    Now let's construct U, V, W matrices.
    n=2, m=3, p=10.
    U has shape (n*m, rank) = (6, 55)
    V has shape (m*p, rank) = (30, 55)
    W has shape (p*n, rank) = (20, 55)
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    Wait, the multiplication tensor T for matrix multiplication satisfies:
    T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    Actually, the structure tensor for matrix multiplication <n,m,p> has entries:
    T[i,j,k] = delta_{i,j,k} in the sense that it's 1 for all (i,j,k) in {0..n-1}x{0..m-1}x{0..p-1}.
    
    Let's build the decomposition column by column.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    # Strassen part: 5 blocks of 2x2x2
    for t in range(5):  # t = 0,1,2,3,4
        # B columns 2t and 2t+1
        # A columns 0 and 1
        # Output rows 0 and 1, output cols 2t and 2t+1
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,2t] + B[1,2t+1])
        # U: A[0,0] -> row 0, A[1,1] -> row 1*2+1=3? No.
        # U row index: i*m + j. i in {0,1}, j in {0,1,2}.
        # A[0,0]: i=0, j=0 -> row 0
        # A[1,1]: i=1, j=1 -> row 3
        # V row index: j*p + k. j in {0,1,2}, k in {0..9}.
        # B[0,2t]: j=0, k=2t -> row 2t
        # B[1,2t+1]: j=1, k=2t+1 -> row 10 + 2t + 1 = 11 + 2t
        # W row index: k*n + i. k in {0..9}, i in {0,1}.
        # C[0,2t]: k=2t, i=0 -> row 2t*2 + 0 = 4t
        # C[1,2t]: k=2t, i=1 -> row 2t*2 + 1 = 4t+1
        # C[0,2t+1]: k=2t+1, i=0 -> row (2t+1)*2 + 0 = 4t+2
        # C[1,2t+1]: k=2t+1, i=1 -> row (2t+1)*2 + 1 = 4t+3
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,2t] + B[1,2t+1])
        # U: +1 at row 0, +1 at row 3
        # V: +1 at row 2t, +1 at row 11+2t
        # W: contributes to C[0,2t], C[1,2t], C[0,2t+1], C[1,2t+1]
        # From Strassen:
        # C[0,2t] = M1 + M4 - M5 + M7
        # C[1,2t] = M2 + M3 + M5 - M6
        # C[0,2t+1] = M1 + M3
        # C[1,2t+1] = M2 + M4
        # So M1 contributes +1 to C[0,2t], C[0,2t+1]
        # W: +1 at row 4t (C[0,2t]), +1 at row 4t+2 (C[0,2t+1])
        
        U[0][col] = 1
        U[3][col] = 1
        V[2*t][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = 1
        W[4*t+2][col] = 1
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,2t]
        # A[1,0]: i=1, j=0 -> row 2
        # A[1,1]: i=1, j=1 -> row 3
        # B[0,2t]: j=0, k=2t -> row 2t
        # M2 contributes +1 to C[1,2t], C[1,2t+1]
        # W: +1 at row 4t+1 (C[1,2t]), +1 at row 4t+3 (C[1,2t+1])
        
        U[2][col] = 1
        U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1
        W[4*t+3][col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[1,2t] - B[1,2t+1])
        # A[0,0]: i=0, j=0 -> row 0
        # B[1,2t]: j=1, k=2t -> row 10+2t
        # B[1,2t+1]: j=1, k=2t+1 -> row 11+2t
        # M3 contributes +1 to C[1,2t], C[0,2t+1]
        # W: +1 at row 4t+1 (C[1,2t]), +1 at row 4t+2 (C[0,2t+1])
        
        U[0][col] = 1
        V[10+2*t][col] = 1
        V[11+2*t][col] = -1
        W[4*t+1][col] = 1
        W[4*t+2][col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[0,2t+1] - B[0,2t])
        # A[1,1]: i=1, j=1 -> row 3
        # B[0,2t+1]: j=0, k=2t+1 -> row 2t+1
        # B[0,2t]: j=0, k=2t -> row 2t
        # M4 contributes +1 to C[0,2t], C[1,2t+1]
        # W: +1 at row 4t (C[0,2t]), +1 at row 4t+3 (C[1,2t+1])
        
        U[3][col] = 1
        V[2*t+1][col] = 1
        V[2*t][col] = -1
        W[4*t][col] = 1
        W[4*t+3][col] = 1
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,2t+1]
        # A[0,0]: i=0, j=0 -> row 0
        # A[0,1]: i=0, j=1 -> row 1
        # B[1,2t+1]: j=1, k=2t+1 -> row 11+2t
        # M5 contributes -1 to C[0,2t], +1 to C[1,2t]
        # W: -1 at row 4t (C[0,2t]), +1 at row 4t+1 (C[1,2t])
        
        U[0][col] = 1
        U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1
        W[4*t+1][col] = 1
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,2t] + B[0,2t+1])
        # A[1,0]: i=1, j=0 -> row 2
        # A[0,0]: i=0, j=0 -> row 0
        # B[0,2t]: j=0, k=2t -> row 2t
        # B[0,2t+1]: j=0, k=2t+1 -> row 2t+1
        # M6 contributes -1 to C[1,2t]
        # W: -1 at row 4t+1 (C[1,2t])
        
        U[2][col] = 1
        U[0][col] = -1
        V[2*t][col] = 1
        V[2*t+1][col] = 1
        W[4*t+1][col] = -1
        col += 1
        
        # M7 = (A[1,1] - A[0,1]) * (B[0,2t+1] + B[1,2t+1])
        # A[1,1]: i=1, j=1 -> row 3
        # A[0,1]: i=0, j=1 -> row 1
        # B[0,2t+1]: j=0, k=2t+1 -> row 2t+1
        # B[1,2t+1]: j=1, k=2t+1 -> row 11+2t
        # M7 contributes +1 to C[0,2t]
        # W: +1 at row 4t (C[0,2t])
        
        U[3][col] = 1
        U[1][col] = -1
        V[2*t+1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = 1
        col += 1
    
    # Schoolbook part: j=2
    # C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0..9}
    for k in range(10):
        for i in range(2):
            # A[i,2]: i in {0,1}, j=2 -> row i*3+2 = 2+i
            # B[2,k]: j=2, k -> row 2*10+k = 20+k
            # C[i,k]: k, i -> row k*2+i = 2k+i
            
            U[2+i][col] = 1
            V[20+k][col] = 1
            W[2*k+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
