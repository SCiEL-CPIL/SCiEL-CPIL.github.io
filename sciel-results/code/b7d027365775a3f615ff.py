"""Matrix multiplication 5x5x5 with block decomposition 4+1 and Strassen for 4x4x4."""
import time

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Returns dict with keys 'U', 'V', 'W' representing a tensor decomposition
    of the 5x5x5 matrix multiplication tensor.
    
    Uses block decomposition: 5 = 4 + 1.
    - M_{4,4,4}: Rank 49 via recursive Strassen (7^2)
    - Other blocks: Schoolbook
    
    Total rank = 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    """
    deadline = time.time() + 120.0
    
    def schoolbook(n, m, p):
        """Return (U, V, W) for schoolbook decomposition of M_{n,m,p}.
        U has shape (n*m, n*m*p), V has shape (m*p, n*m*p), W has shape (p*n, n*m*p)"""
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
        return U, V, W
    
    def strassen_decomp(n):
        """
        Return (U, V, W) for Strassen decomposition of M_{n,n,n} where n is power of 2.
        U has shape (n*n, rank), V has shape (n*n, rank), W has shape (n*n, rank)
        
        For n=2, rank=7. For n=4, rank=49. Etc.
        """
        if n == 2:
            # Standard 7-term Strassen for 2x2
            # C = AB where A, B, C are 2x2
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-e), M5 = (a+b)h, M6 = (c-d)(f+g), M7 = (b-d)(e+h)
            # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
            #
            # U rows indexed by (i,j) of A, i,j in 0..1
            # V rows indexed by (j,k) of B, j,k in 0..1
            # W rows indexed by (k,i) of C, k,i in 0..1
            #
            # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
            # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
            # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
            #
            # W index: k*n + i where k is row of C (output row), i is col of C (output col)
            # Wait, let me re-read the convention: W[k*n+i, r]
            # So W is indexed by (k, i) where k is the output column index and i is the output row index?
            # Let me think: C = A * B, C[i][k] = sum_j A[i][j] B[j][k]
            # So C[i][k] depends on A[i][j] and B[j][k]
            # The tensor is T[i, j, k] such that C[i][k] = sum_j T[i,j,k] A[i][j] B[j][k]... no wait
            # Actually the multiplication tensor M_{n,m,p} has components M[i,j,k]
            # C[i][k] = sum_j A[i][j] B[j][k]
            # So the tensor T has T[i,j,k] = 1 if we're looking at the (i,k) entry
            # W[k*n + i, r] - so W is indexed by (k, i) where k is the column index of C, i is the row index of C
            # This means W[k*n+i] corresponds to C[i][k]
            
            # U[i*m + j, r] - U is indexed by (i, j) where i is row of A, j is col of A
            # V[j*p + k, r] - V is indexed by (j, k) where j is row of B, k is col of B
            # W[k*n + i, r] - W is indexed by (k, i) where k is col of C, i is row of C
            
            # For 2x2:
            # U rows: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
            # V rows: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
            # W rows: (0,0)=C[0][0], (0,1)=C[0][1], (1,0)=C[1][0], (1,1)=C[1][1]
            #   i.e., W[k*2+i] where k is col of C, i is row of C
            #   W[0] = C[0][0], W[1] = C[0][1], W[2] = C[1][0], W[3] = C[1][1]
            
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            # M1 = (a+d)(e+h)
            # a+d: U[0]=1, U[3]=1 (rows 0 and 3 of U)
            # e+h: V[0]=1, V[3]=1
            # Contributes to C[0][0] and C[1][1]
            # C[0][0] = W[0], C[1][1] = W[3]
            # M1 = +1 to C[0][0], +1 to C[1][1]
            for row_u in [0, 3]:  # a, d
                U[row_u][0] += 1
            for row_v in [0, 3]:  # e, h
                V[row_v][0] += 1
            W[0][0] += 1   # C[0][0]
            W[3][0] += 1   # C[1][1]
            
            # M2 = (c+d)(e)
            # c+d: U[1]=1, U[3]=1
            # e: V[0]=1
            # C[1][0] = M2 + ... so W[2] += M2
            for row_u in [1, 3]:  # c, d
                U[row_u][1] += 1
            V[0][1] += 1  # e
            W[2][1] += 1  # C[1][0]
            
            # M3 = a(f-h)
            # a: U[0]=1
            # f-h: V[1]=1, V[3]=-1
            # C[0][1] = M3 + M5, so W[1] += M3
            U[0][2] += 1  # a
            V[1][2] += 1  # f
            V[3][2] -= 1  # -h
            W[1][2] += 1  # C[0][1]
            
            # M4 = d(g-e)
            # d: U[3]=1
            # g-e: V[2]=1, V[0]=-1
            # C[0][0] += M4, C[1][0] += M4
            U[3][3] += 1  # d
            V[2][3] += 1  # g
            V[0][3] -= 1  # -e
            W[0][3] += 1  # C[0][0]
            W[2][3] += 1  # C[1][0]
            
            # M5 = (a+b)h
            # a+b: U[0]=1, U[1]=1
            # h: V[3]=1
            # C[0][0] -= M5, C[0][1] += M5
            for row_u in [0, 1]:  # a, b
                U[row_u][4] += 1
            V[3][4] += 1  # h
            W[0][4] -= 1  # C[0][0]
            W[1][4] += 1  # C[0][1]
            
            # M6 = (c-d)(f+g)
            # c-d: U[1]=1, U[3]=-1
            # f+g: V[1]=1, V[2]=1
            # C[1][1] += M6
            U[1][5] += 1  # c
            U[3][5] -= 1  # -d
            V[1][5] += 1  # f
            V[2][5] += 1  # g
            W[3][5] += 1  # C[1][1]
            
            # M7 = (b-d)(e+h)
            # b-d: U[1]=1, U[3]=-1
            # e+h: V[0]=1, V[3]=1
            # C[0][0] += M7
            U[1][6] += 1  # b
            U[3][6] -= 1  # -d
            V[0][6] += 1  # e
            V[3][6] += 1  # h
            W[0][6] += 1  # C[0][0]
            
            return U, V, W
        
        # Recursive case: n = 2 * n_half
        n_half = n // 2
        U_half, V_half, W_half = strassen_decomp(n_half)
        rank_half = len(U_half[0])
        total_rank = 7 * rank_half
        
        U = [[0] * total_rank for _ in range(n * n)]
        V = [[0] * total_rank for _ in range(n * n)]
        W = [[0] * total_rank for _ in range(n * n)]
        
        # Define the 7 Strassen combinations at the block level
        # Each block is n_half x n_half
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        #
        # Block indices in the full matrix:
        # A11: rows 0..n_half-1, cols 0..n_half-1
        # A12: rows 0..n_half-1, cols n_half..2*n_half-1
        # A21: rows n_half..2*n_half-1, cols 0..n_half-1
        # A22: rows n_half..2*n_half-1, cols n_half..2*n_half-1
        # Similarly for B and C.
        
        # For each of the 7 terms, we have a combination of blocks
        # term_t has U contribution: for each block (bi, bj) of A, coefficient c
        # We need to map block-level U to element-level U
        
        # The 7 terms at block level:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A22)(B12+B21)
        # M7 = (A12-A22)(B11+B22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For each term t (0..6), we define:
        # U_blocks[t]: list of (block_row, block_col, coeff) for A
        # V_blocks[t]: list of (block_row, block_col, coeff) for B
        # W_blocks[t]: list of (block_row, block_col, coeff) for C
        
        # Block (br, bc) of A corresponds to rows br*n_half..(br+1)*n_half-1, cols bc*n_half..(bc+1)*n_half-1
        # Element index in U: (br*n_half + i) * n + (bc*n_half + j) where i,j are within the block
        
        U_blocks = [
            [(0,0,1), (1,1,1)],       # M1: A11+A22
            [(1,0,1), (1,1,1)],       # M2: A21+A22
            [(0,0,1)],                # M3: A11
            [(1,1,1)],                # M4: A22
            [(0,0,1), (0,1,1)],       # M5: A11+A12
            [(1,0,1), (1,1,-1)],      # M6: A21-A22
            [(0,1,1), (1,1,-1)],      # M7: A12-A22
        ]
        
        V_blocks = [
            [(0,0,1), (1,1,1)],       # M1: B11+B22
            [(0,0,1)],                # M2: B11
            [(0,1,1), (1,1,-1)],      # M3: B12-B22
            [(1,0,1), (0,0,-1)],      # M4: B21-B11
            [(1,1,1)],                # M5: B22
            [(0,1,1), (1,0,1)],       # M6: B12+B21
            [(0,0,1), (1,1,1)],       # M7: B11+B22
        ]
        
        W_blocks = [
            [(0,0,1), (1,1,1)],       # M1: C11+C22
            [(1,0,1)],                # M2: C21
            [(0,1,1)],                # M3: C12
            [(0,0,1), (1,0,1)],       # M4: C11+C21
            [(0,0,-1), (0,1,1)],      # M5: -C11+C12
            [(1,1,1)],                # M6: C22
            [(0,0,1)],                # M7: C11
        ]
        
        for t in range(7):
            # For each block combination, apply the half-size Strassen decomposition
            # The rank offset for term t is t * rank_half
            
            rank_offset = t * rank_half
            
            # Process U contributions
            for (br, bc, coeff) in U_blocks[t]:
                for i in range(n_half):
                    for j in range(n_half):
                        u_row = (br * n_half + i) * n + (bc * n_half + j)
                        for r in range(rank_half):
                            U[u_row][rank_offset + r] += coeff * U_half[i * n_half + j][r]
            
            # Process V contributions
            for (br, bc, coeff) in V_blocks[t]:
                for i in range(n_half):
                    for j in range(n_half):
                        v_row = (br * n_half + i) * n + (bc * n_half + j)
                        for r in range(rank_half):
                            V[v_row][rank_offset + r] += coeff * V_half[i * n_half + j][r]
            
            # Process W contributions
            for (br, bc, coeff) in W_blocks[t]:
                for i in range(n_half):
                    for j in range(n_half):
                        # W is indexed by (k, i) where k is col of C, i is row of C
                        # Block (br, bc) of C: rows br*n_half.., cols bc*n_half..
                        # So element (i,j) in block (br,bc) of C is at row br*n_half+i, col bc*n_half+j
                        # W index: col * n + row = (bc*n_half+j)*n + (br*n_half+i)
                        w_row = (bc * n_half + j) * n + (br * n_half + i)
                        for r in range(rank_half):
                            W[w_row][rank_offset + r] += coeff * W_half[i * n_half + j][r]
        
        return U, V, W
    
    def add_decomps(U1, V1, W1, U2, V2, W2):
        """Add two decompositions (concatenate columns)."""
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        total_rank = rank1 + rank2
        U = [row[:rank1] + [0] * rank2 for row in U1]
        V = [row[:rank1] + [0] * rank2 for row in V1]
        W = [row[:rank1] + [0] * rank2 for row in W1]
        for i in range(len(U2)):
            for r in range(rank2):
                U[i][rank1 + r] = U2[i][r]
                V[i][rank1 + r] = V2[i][r]
                W[i][rank1 + r] = W2[i][r]
        return U, V, W
    
    def shift_decomp(U, V, W, row_offset, rank_offset):
        """
        Place a sub-decomposition into the correct position in the full matrix.
        row_offset: offset in the row indices of U, V, W
        rank_offset: offset in the rank (column) indices
        """
        rows_U = len(U)
        rows_V = len(V)
        rows_W = len(W)
        rank = len(U[0])
        
        U_full = [[0] * (rank + rank_offset) for _ in range(rows_U + row_offset)]
        V_full = [[0] * (rank + rank_offset) for _ in range(rows_V + row_offset)]
        W_full = [[0] * (rank + rank_offset) for _ in range(rows_W + row_offset)]
        
        # This approach is getting complex. Let me instead build the full matrices directly.
        pass
    
    # Let me take a cleaner approach: build the full 5x5x5 decomposition directly.
    
    # The 5x5 matrix A is partitioned into:
    # A = [[A44, A41], [A14, A11]] where A44 is 4x4, A41 is 4x1, A14 is 1x4, A11 is 1x1
    # B = [[B44, B41], [B14, B11]]
    # C = [[C44, C41], [C14, C11]] = A * B
    
    # C44 = A44*B44 + A41*B14  (4x4)
    # C41 = A44*B41 + A41*B11  (4x1)
    # C14 = A14*B44 + A11*B14  (1x4)
    # C11 = A14*B41 + A11*B11  (1x1)
    
    # For each product term, we use a decomposition.
    
    # Let's build U, V, W for the full 5x5 problem.
    # U has shape (25, total_rank)
    # V has shape (25, total_rank)
    # W has shape (25, total_rank)
    
    # A is 5x5, indexed 0..4. A[i,j] with i=row, j=col
    # B is 5x5, B[j,k] with j=row, k=col
    # C is 5x5, C[i,k] with i=row, k=col
    # U[i*5+j, r], V[j*5+k, r], W[k*5+i, r]
    
    # Block 4x4 of A: rows 0-3, cols 0-3
    # Block 4x1 of A: rows 0-3, col 4
    # Block 1x4 of A: row 4, cols 0-3
    # Block 1x1 of A: row 4, col 4
    
    # Similarly for B and C.
    
    # Product terms:
    # 1. A44 * B44 -> contributes to C44 (4x4)
    # 2. A41 * B14 -> contributes to C44 (4x4)
    # 3. A44 * B41 -> contributes to C41 (4x1)
    # 4. A41 * B11 -> contributes to C41 (4x1)
    # 5. A14 * B44 -> contributes to C14 (1x4)
    # 6. A11 * B14 -> contributes to C14 (1x4)
    # 7. A14 * B41 -> contributes to C11 (1x1)
    # 8. A11 * B11 -> contributes to C11 (1x1)
    
    # Get decompositions for each:
    # 1. M_{4,4,4}: Strassen, rank 49
    # 2. M_{4,1,4}: schoolbook, rank 16
    # 3. M_{4,4,1}: schoolbook, rank 16
    # 4. M_{4,1,1}: schoolbook, rank 4
    # 5. M_{1,4,4}: schoolbook, rank 16
    # 6. M_{1,1,4}: schoolbook, rank 4
    # 7. M_{1,4,1}: schoolbook, rank 4
    # 8. M_{1,1,1}: schoolbook, rank 1
    
    U44, V44, W44 = strassen_decomp(4)
    U41_14, V41_14, W41_14 = schoolbook(4, 1, 4)
    U44_41, V44_41, W44_41 = schoolbook(4, 4, 1)
    U41_11, V41_11, W41_11 = schoolbook(4, 1, 1)
    U14_44, V14_44, W14_44 = schoolbook(1, 4, 4)
    U11_14, V11_14, W11_14 = schoolbook(1, 1, 4)
    U14_41, V14_41, W14_41 = schoolbook(1, 4, 1)
    U11_11, V11_11, W11_11 = schoolbook(1, 1, 1)
    
    # Now we need to place each sub-decomposition into the full 25x25 matrices.
    # The sub-decomposition U has shape (n*m, rank) where n, m are the dimensions of the sub-problem.
    # We need to map the sub-problem indices to the full 5x5 indices.
    
    def place_decomp(U_sub, V_sub, W_sub, 
                     A_row_start, A_col_start,
                     B_row_start, B_col_start,
                     C_row_start, C_col_start,
                     rank_offset):
        """
        Place a sub-decomposition into the full matrices.
        U_sub has shape (n*m, rank) where A is n x m
        V_sub has shape (m*p, rank) where B is m x p
        W_sub has shape (p*n, rank) where C is p x n
        
        A_row_start: starting row index of A in the full matrix
        A_col_start: starting col index of A in the full matrix
        B_row_start: starting row index of B in the full matrix
        B_col_start: starting col index of B in the full matrix
        C_row_start: starting row index of C in the full matrix
        C_col_start: starting col index of C in the full matrix
        rank_offset: starting column index in the full U, V, W
        """
        n = A_row_start  # This is wrong, need to pass n, m, p
        # Actually, let me just do this manually for each term.
        pass
    
    # Let me just do it manually.
    
    total_rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    # Helper to place a sub-decomp
    def place(U_sub, V_sub, W_sub, A_rs, A_cs, B_rs, B_cs, C_rs, C_cs, rank_off):
        """
        Place sub-decomp into full matrices.
        A is n x m: rows A_rs..A_rs+n-1, cols A_cs..A_cs+m-1
        B is m x p: rows B_rs..B_rs+m-1, cols B_cs..B_cs+p-1
        C is p x n: rows C_rs..C_rs+p-1, cols C_cs..C_cs+n-1
        """
        n = len(U_sub) // len(U_sub[0]) * 0  # Can't determine n,m,p from shapes alone easily
        # U_sub shape: (n*m, rank)
        # V_sub shape: (m*p, rank)
        # W_sub shape: (p*n, rank)
        # We need to figure out n, m, p from the context.
        # Let me just pass n, m, p explicitly.
        pass
    
    # I'll just do each one explicitly.
    
    # Term 1: A44 * B44 -> C44
    # A: rows 0-3, cols 0-3 (4x4)
    # B: rows 0-3, cols 0-3 (4x4)
    # C: rows 0-3, cols 0-3 (4x4)
    # U44 shape: (16, 49), V44 shape: (16, 49), W44 shape: (16, 49)
    for i in range(16):
        for r in range(49):
            # U44[i] corresponds to A[?][?]
            # U is indexed by (a_row, a_col) -> i = a_row * 4 + a_col
            a_row = i // 4
            a_col = i % 4
            U[a_row + 0][r + 0] += U44[i][r]
            
            # V44[i] corresponds to B[?][?]
            # V is indexed by (b_row, b_col) -> i = b_row * 4 + b_col
            b_row = i // 4
            b_col = i % 4
            V[b_row + 0][r + 0] += V44[i][r]
            
            # W44[i] corresponds to C[?][?]
            # W is indexed by (c_col, c_row) -> i = c_col * 4 + c_row
            c_col = i // 4
            c_row = i % 4
            W[(c_col + 0) * 5 + (c_row + 0)][r + 0] += W44[i][r]
    
    # Term 2: A41 * B14 -> C44
    # A: rows 0-3, col 4 (4x1)
    # B: row 4, cols 0-3 (1x4)
    # C: rows 0-3, cols 0-3 (4x4)
    # U41_14 shape: (4, 16), V41_14 shape: (4, 16), W41_14 shape: (16, 16)
    for i in range(4):
        for r in range(16):
            # U41_14[i] corresponds to A[i][0] in the 4x1 sub-matrix
            a_row = i
            a_col = 0
            U[a_row + 0][a_col + 4 * 5 + r + 49] += U41_14[i][r]
            # Wait, U is indexed by (a_row * 5 + a_col, rank)
            # So U[a_row * 5 + a_col]
            # Let me redo this.
            
    # Let me restart with a cleaner helper function.
    
    def place_term(U_sub, V_sub, W_sub, A_rs, A_cs, B_rs, B_cs, C_rs, C_cs, n, m, p, rank_off):
        """
        Place sub-decomposition into full 5x5 matrices.
        U_sub shape: (n*m, rank), V_sub shape: (m*p, rank), W_sub shape: (p*n, rank)
        A is at rows A_rs..A_rs+n-1, cols A_cs..A_cs+m-1
        B is at rows B_rs..B_rs+m-1, cols B_cs..B_cs+p-1
        C is at rows C_rs..C_rs+p-1, cols C_cs..C_cs+n-1
        """
        rank = len(U_sub[0])
        for i in range(n * m):
            a_row = i // m
            a_col = i % m
            full_U_row = (A_rs + a_row) * 5 + (A_cs + a_col)
            for r in range(rank):
                U[full_U_row][rank_off + r] += U_sub[i][r]
        
        for i in range(m * p):
            b_row = i // p
            b_col = i % p
            full_V_row = (B_rs + b_row) * 5 + (B_cs + b_col)
            for r in range(rank):
                V[full_V_row][rank_off + r] += V_sub[i][r]
        
        for i in range(p * n):
            c_col = i // n
            c_row = i % n
            full_W_row = (C_cs + c_col) * 5 + (C_rs + c_row)
            for r in range(rank):
                W[full_W_row][rank_off + r] += W_sub[i][r]
    
    # Term 1: A44 * B44 -> C44
    place_term(U44, V44, W44, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0)
    
    # Term 2: A41 * B14 -> C44
    # A: 4x1 at rows 0-3, col 4
    # B: 1x4 at row 4, cols 0-3
    # C: 4x4 at rows 0-3, cols 0-3
    place_term(U41_14, V41_14, W41_14, 0, 4, 4, 0, 0, 0, 4, 1, 4, 49)
    
    # Term 3: A44 * B41 -> C41
    # A: 4x4 at rows 0-3, cols 0-3
    # B: 4x1 at rows 0-3, col 4
    # C: 4x1 at rows 0-3, col 4
    place_term(U44_41, V44_41, W44_41, 0, 0, 0, 4, 0, 4, 4, 4, 1, 65)
    
    # Term 4: A41 * B11 -> C41
    # A: 4x1 at rows 0-3, col 4
    # B: 1x1 at row 4, col 4
    # C: 4x1 at rows 0-3, col 4
    place_term(U41_11, V41_11, W41_11, 0, 4, 4, 4, 0, 4, 4, 1, 1, 81)
    
    # Term 5: A14 * B44 -> C14
    # A: 1x4 at row 4, cols 0-3
    # B: 4x4 at rows 0-3, cols 0-3
    # C: 1x4 at row 4, cols 0-3
    place_term(U14_44, V14_44, W14_44, 4, 0, 0, 0, 4, 0, 1, 4, 4, 85)
    
    # Term 6: A11 * B14 -> C14
    # A: 1x1 at row 4, col 4
    # B: 1x4 at row 4, cols 0-3
    # C: 1x4 at row 4, cols 0-3
    place_term(U11_14, V11_14, W11_14, 4, 4, 4, 0, 4, 0, 1, 1, 4, 101)
    
    # Term 7: A14 * B41 -> C11
    # A: 1x4 at row 4, cols 0-3
    # B: 4x1 at rows 0-3, col 4
    # C: 1x1 at row 4, col 4
    place_term(U14_41, V14_41, W14_41, 4, 0, 0, 4, 4, 4, 1, 4, 1, 105)
    
    # Term 8: A11 * B11 -> C11
    # A: 1x1 at row 4, col 4
    # B: 1x1 at row 4, col 4
    # C: 1x1 at row 4, col 4
    place_term(U11_11, V11_11, W11_11, 4, 4, 4, 4, 4, 4, 1, 1, 1, 109)
    
    return {"U": U, "V": V, "W": W}
