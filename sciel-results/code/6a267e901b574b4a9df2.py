"""Optimized matrix multiplication decomposition for 2x5 * 5x6 using Strassen-like blocks."""

def construct():
    n, m, p = 2, 5, 6
    
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (12, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        # u_vec: length 10
        # v_vec: length 30
        # w_vec: length 12
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose 2x5 * 5x6.
    # Split B's 6 columns into 3 pairs: (0,1), (2,3), (4,5).
    # For each pair, we have a 2x5 * 5x2 subproblem.
    # Split the 5 rows of the subproblem into 4 rows and 1 row.
    # The 4 rows can be split into two 2-row blocks, each giving a 2x2 * 2x2 Strassen block.
    # The 1 row gives a 2x1 * 1x2 subproblem, which has rank 2 (since C[i,k] = A[i,j] * B[j,k] for fixed j).
    
    # Let's define the indices:
    # U row index: i * m + j, where i in {0,1}, j in {0..4}
    # V row index: j * p + k, where j in {0..4}, k in {0..5}
    # W row index: k * n + i, where k in {0..5}, i in {0,1}
    
    # For each column pair (k1, k2):
    #   For each row block (j1, j2) in [(0,1), (2,3)]:
    #     Apply Strassen to A[i, j1..j2] and B[j1..j2, k1..k2]
    #   For the remaining row j=4:
    #     Apply rank-2 decomposition for 2x1 * 1x2
    
    # Strassen block for rows j1, j2 and cols k1, k2:
    # A block is 2x2: A[i, j1], A[i, j2] for i in {0,1}
    # B block is 2x2: B[j1, k1], B[j1, k2], B[j2, k1], B[j2, k2]
    
    def add_strassen_block(i_rows, j1, j2, k1, k2):
        # i_rows: list of row indices in A (should be [0, 1])
        # j1, j2: row indices in B (and column indices in A)
        # k1, k2: column indices in B (and row indices in C)
        
        # A indices: (i, j) -> U row i*m + j
        # B indices: (j, k) -> V row j*p + k
        # C indices: (k, i) -> W row k*n + i
        
        # M1 = (A[0,j1] + A[1,j2]) * (B[j1,k1] + B[j2,k2])
        # Wait, Strassen is:
        # M1 = (A00 + A11) * (B00 + B11)
        # where A00 = A[0,j1], A11 = A[1,j2], B00 = B[j1,k1], B11 = B[j2,k2]
        
        # Let's define the 2x2 blocks:
        # A_block = [[A[0,j1], A[0,j2]], [A[1,j1], A[1,j2]]]
        # B_block = [[B[j1,k1], B[j1,k2]], [B[j2,k1], B[j2,k2]]]
        
        # M1 = (A[0,j1] + A[1,j2]) * (B[j1,k1] + B[j2,k2])
        u = [0]*10; v = [0]*30; w = [0]*12
        u[0*m + j1] = 1; u[1*m + j2] = 1
        v[j1*p + k1] = 1; v[j2*p + k2] = 1
        w[k1*n + 0] = 1; w[k2*n + 1] = 1  # This is wrong, M1 contributes to C00 and C11
        # Actually, M1 contributes to C[0,k1] (via C00) and C[1,k2] (via C11)
        # C00 = M1 + M4 - M5 + M7 -> C[0,k1]
        # C11 = M1 - M2 + M3 + M6 -> C[1,k2]
        # So M1 contributes to W[k1*n + 0] and W[k2*n + 1]
        add_tensor(u, v, w)
        
        # M2 = (A[1,j1] + A[1,j2]) * B[j1,k1]
        u = [0]*10; v = [0]*30; w = [0]*12
        u[1*m + j1] = 1; u[1*m + j2] = 1
        v[j1*p + k1] = 1
        w[k1*n + 1] = 1  # C10 = M2 + M4 -> C[1,k1]
        add_tensor(u, v, w)
        
        # M3 = A[0,j1] * (B[j2,k1] - B[j2,k2])
        u = [0]*10; v = [0]*30; w = [0]*12
        u[0*m + j1] = 1
        v[j2*p + k1] = 1; v[j2*p + k2] = -1
        w[k1*n + 0] = 1  # C01 = M3 + M5 -> C[0,k2]... wait
        # C01 = M3 + M5 -> C[0,k2]
        # So M3 contributes to W[k2*n + 0]
        # Let me re-check Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # M3 = A00 * (B10 - B11) = A[0,j1] * (B[j2,k1] - B[j2,k2])
        # C01 = M3 + M5 -> C[0,k2]
        # So M3 contributes to W[k2*n + 0]
        w = [0]*12
        w[k2*n + 0] = 1
        add_tensor(u, v, w)
        
        # M4 = A[1,j2] * (B[j1,k2] - B[j1,k1])
        u = [0]*10; v = [0]*30; w = [0]*12
        u[1*m + j2] = 1
        v[j1*p + k2] = 1; v[j1*p + k1] = -1
        w[k1*n + 0] = 1; w[k1*n + 1] = 1  # C00 and C10
        # C00 = ... + M4 ... -> C[0,k1]
        # C10 = M2 + M4 -> C[1,k1]
        add_tensor(u, v, w)
        
        # M5 = (A[0,j1] + A[0,j2]) * B[j2,k2]
        u = [0]*10; v = [0]*30; w = [0]*12
        u[0*m + j1] = 1; u[0*m + j2] = 1
        v[j2*p + k2] = 1
        w[k1*n + 0] = -1; w[k2*n + 0] = 1  # C00: -M5, C01: +M5
        add_tensor(u, v, w)
        
        # M6 = (A[1,j1] - A[0,j1]) * (B[j1,k1] + B[j1,k2])
        u = [0]*10; v = [0]*30; w = [0]*12
        u[1*m + j1] = 1; u[0*m + j1] = -1
        v[j1*p + k1] = 1; v[j1*p + k2] = 1
        w[k2*n + 1] = 1  # C11 = ... + M6 -> C[1,k2]
        add_tensor(u, v, w)
        
        # M7 = (A[0,j2] - A[1,j2]) * (B[j2,k1] + B[j2,k2])
        u = [0]*10; v = [0]*30; w = [0]*12
        u[0*m + j2] = 1; u[1*m + j2] = -1
        v[j2*p + k1] = 1; v[j2*p + k2] = 1
        w[k1*n + 0] = 1  # C00 = ... + M7 -> C[0,k1]
        add_tensor(u, v, w)
    
    # Rank-2 decomposition for 2x1 * 1x2 (single row j, two cols k1, k2)
    # C[0,k1] = A[0,j] * B[j,k1]
    # C[0,k2] = A[0,j] * B[j,k2]
    # C[1,k1] = A[1,j] * B[j,k1]
    # C[1,k2] = A[1,j] * B[j,k2]
    # This can be done with 2 rank-1 terms:
    # Term 1: A[0,j] * B[j,k1] -> C[0,k1]
    # Term 2: A[0,j] * B[j,k2] -> C[0,k2]
    # Term 3: A[1,j] * B[j,k1] -> C[1,k1]
    # Term 4: A[1,j] * B[j,k2] -> C[1,k2]
    # Wait, that's 4 terms. But we can share:
    # Actually, for 2x1 * 1x2, the rank is 2 (since it's two independent dot products of length 1).
    # No, it's 4 scalar multiplications. But we can write it as:
    # C = [[A0, A0], [A1, A1]] * [[B0], [B1]]... no.
    # C[0,k1] = A[0,j] * B[j,k1]
    # C[0,k2] = A[0,j] * B[j,k2]
    # C[1,k1] = A[1,j] * B[j,k1]
    # C[1,k2] = A[1,j] * B[j,k2]
    # This is rank 4 (schoolbook). But wait, we can factor:
    # Let X = A[0,j], Y = A[1,j], P = B[j,k1], Q = B[j,k2]
    # C = [[X*P, X*Q], [Y*P, Y*Q]]
    # This is rank 2: [[X, X], [Y, Y]] * [[P, Q]]... no, that's not a valid decomposition.
    # Actually, the tensor for 2x1x2 has rank 2.
    # C[i,k] = A[i,j] * B[j,k]
    # We can write this as sum over r of U[i,j] * V[j,k] * W[k,i]
    # With r=1: U[0,j]=1, U[1,j]=0, V[j,k1]=1, V[j,k2]=0, W[k1,0]=1, W[k2,0]=0, W[k1,1]=0, W[k2,1]=0
    # That gives C[0,k1] = A[0,j]*B[j,k1], but C[0,k2]=0, C[1,k1]=0, C[1,k2]=0.
    # So we need 4 terms for the schoolbook.
    # But the rank of 2x1x2 is actually 2.
    # Let's verify: C[i,k] = A[i] * B[k] (vectors of length 2 and 2)
    # This is the outer product of A and B^T, which has rank 1 as a matrix, but as a tensor...
    # The tensor is T[i,j,k] = A[i] * B[k] (with j fixed).
    # This is a rank-1 tensor in the (i,k) space for fixed j.
    # But we need to decompose it as sum_r U[i,j] * V[j,k] * W[k,i].
    # With j fixed, V[j,k] is a vector of length 2 (for k=0,1).
    # U[i,j] is a vector of length 2 (for i=0,1).
    # W[k,i] is a matrix of size 2x2.
    # So we need to write T[i,k] = sum_r U[i] * V[k] * W[k,i]
    # This is equivalent to writing the matrix T as sum_r U * (V * W_r)
    # where U is 2x1, V is 1x2, W_r is 2x2.
    # T = [[A0*B0, A0*B1], [A1*B0, A1*B1]]
    # This can be written as:
    # r=1: U = [A0, 0], V = [B0, B1], W = [[1, 0], [0, 0]] -> gives [[A0*B0, A0*B1], [0, 0]]
    # r=2: U = [0, A1], V = [B0, B1], W = [[0, 0], [0, 1]] -> gives [[0, 0], [A1*B0, A1*B1]]
    # So rank 2.
    
    def add_rank2_block(j, k1, k2):
        # Term 1
        u = [0]*10; v = [0]*30; w = [0]*12
        u[0*m + j] = 1
        v[j*p + k1] = 1; v[j*p + k2] = 1
        w[k1*n + 0] = 1; w[k2*n + 0] = 1
        add_tensor(u, v, w)
        
        # Term 2
        u = [0]*10; v = [0]*30; w = [0]*12
        u[1*m + j] = 1
        v[j*p + k1] = 1; v[j*p + k2] = 1
        w[k1*n + 1] = 1; w[k2*n + 1] = 1
        add_tensor(u, v, w)
    
    # Now, for each column pair:
    for col_pair in [(0, 1), (2, 3), (4, 5)]:
        k1, k2 = col_pair
        # Row blocks: (0,1), (2,3), and single row 4
        add_strassen_block([0, 1], 0, 1, k1, k2)
        add_strassen_block([0, 1], 2, 3, k1, k2)
        add_rank2_block(4, k1, k2)
    
    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U[i][r] = U_cols[r][i]
        for j in range(m*p):
            V[j][r] = V_cols[r][j]
        for k in range(p*n):
            W[k][r] = W_cols[r][k]
    
    return {"U": U, "V": V, "W": W}
