"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll build U, V, W matrices for the tensor decomposition
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta for matrix mult
    
    # Use Strassen's algorithm for 2x2 blocks
    # Partition A (4x4) into 2x2 blocks, B (4x7) into 2x2 and 2x3 blocks
    
    # We'll accumulate the decomposition
    U_rows = {}  # (i, j) -> list of contributions
    V_rows = {}  # (j, k) -> list of contributions  
    W_rows = {}  # (k, i) -> list of contributions
    
    # Instead, let's build the full matrices directly
    # Start with empty lists, we'll add columns as we go
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(i, j, k, ui, vj, wk):
        """Add a term to the decomposition"""
        U_cols.append([0] * (n * m))
        V_cols.append([0] * (m * p))
        W_cols.append([0] * (p * n))
        U_cols[-1][i * m + j] = ui
        V_cols[-1][j * p + k] = vj
        W_cols[-1][k * n + i] = wk
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4 x 4x7, we partition:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11,B21 are 2x2, B12,B22 are 2x2, B13,B23 are 2x3
    
    # C11 = A11*B11 + A12*B21 (2x2 x 2x2 + 2x2 x 2x2)
    # C12 = A11*B12 + A12*B22 (2x2 x 2x2 + 2x2 x 2x2)
    # C13 = A11*B13 + A12*B23 (2x2 x 2x3 + 2x2 x 2x3)
    # C21 = A21*B11 + A22*B21 (2x2 x 2x2 + 2x2 x 2x2)
    # C22 = A21*B12 + A22*B22 (2x2 x 2x2 + 2x2 x 2x2)
    # C23 = A21*B13 + A22*B23 (2x2 x 2x3 + 2x2 x 2x3)
    
    # For the 2x2 x 2x2 parts, use Strassen (7 multiplications each)
    # For the 2x2 x 2x3 parts, use schoolbook (6 multiplications each)
    
    # Let's define helper functions to get block indices
    # A block (bi, bj) where bi, bj in {0,1} covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bi, bj) where bi in {0,1}, bj in {0,1,2}
    #   if bj < 2: cols 2*bj to 2*bj+1 (2x2)
    #   if bj == 2: cols 4 to 6 (2x3)
    
    # Strassen for 2x2 x 2x2: given block indices for A (ai, aj) and B (bi, bj) where both are 2x2
    # This contributes to C block (ai, bj)
    
    def strassen_2x2(ai, aj, bi, bj):
        """Add Strassen decomposition for A[ai,aj] * B[bi,bj] contributing to C[ai,bj]"""
        # A blocks: A11=(ai,aj), A12=(ai,aj+1), A21=(ai+1,aj), A22=(ai+1,aj+1)
        # B blocks: B11=(bi,bj), B12=(bi,bj+1), B21=(bi+1,bj), B22=(bi+1,bj+1)
        # C block: (ai,bj)
        
        a11, a12, a21, a22 = (ai, aj), (ai, aj+1), (ai+1, aj), (ai+1, aj+1)
        b11, b12, b21, b22 = (bi, bj), (bi, bj+1), (bi+1, bj), (bi+1, bj+1)
        
        # M1 = (A11 + A22) * (B11 + B22)
        # This is a sum of 4 products, but in Strassen we treat it as one "multiplication"
        # In tensor decomposition, each "multiplication" Mx corresponds to a rank-1 term
        # M1 = (A11 + A22) * (B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But Strassen's trick is that we can compute this with 7 "multiplications" total
        
        # Actually, in the tensor decomposition, each of the 7 Strassen multiplications
        # corresponds to a set of rank-1 terms. Let me think about this more carefully.
        
        # The Strassen decomposition for 2x2 x 2x2 has rank 7.
        # Each of the 7 "multiplications" M1..M7 is a bilinear form.
        # M1 = (A11+A22)*(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is not rank-1 in the tensor sense; it's a sum of 4 rank-1 terms.
        
        # I think I'm confusing myself. Let me reconsider.
        
        # In the matrix multiplication tensor, we have:
        # C_kl = sum_{ij} A_ik * B_jl
        # The tensor is T[i,j,k,l] = delta_{i,k} * delta_{j,l} (in some indexing)
        
        # Strassen's algorithm shows that the 2x2 matrix multiplication tensor
        # can be decomposed as a sum of 7 rank-1 tensors.
        
        # Let me look up the explicit Strassen decomposition in terms of the tensor.
        
        # Actually, I think the cleanest way is to just use the known fact that
        # 2x2 x 2x2 has rank 7, and implement it by directly specifying the 7 terms.
        
        # The 7 terms for Strassen (in terms of which (i,j,k) triples are nonzero):
        # Using 1-based indexing for A (2x2), B (2x2), C (2x2):
        # M1: A(1,1),A(2,2) with B(1,1),B(2,2) -> contributes to C(1,1),C(2,2)
        # This is getting complicated. Let me use a different approach.
        
        pass
    
    # Let me use a simpler but still better approach: 
    # Use Strassen for the overall 4x4 x 4x4 part and schoolbook for the 4x4 x 4x3 part
    # 4x4 x 4x4 with Strassen: rank 49
    # 4x4 x 4x3 with schoolbook: rank 48
    # Total: 97 vs schoolbook 112
    
    # Actually, let me just implement Strassen recursively for the square part
    
    # For now, let me implement a clean Strassen-based decomposition for 4x4 x 4x4
    # and schoolbook for 4x4 x 4x3
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(i, j, k, ui, vj, wk):
        """Add a rank-1 term to the decomposition"""
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[i * m + j] = ui
        V_row[j * p + k] = vj
        W_row[k * n + i] = wk
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # Part 1: 4x4 x 4x4 (columns 0-3 of B)
    # Use Strassen's algorithm recursively
    # 4x4 = 2x2 blocks, each 2x2 x 2x2 uses Strassen (rank 7)
    # Total for 4x4 x 4x4: 4 * 7 = 28? No, that's not right.
    
    # Actually, Strassen for 4x4 x 4x4:
    # Split into 2x2 blocks: 4 subproblems of 2x2 x 2x2
    # Each 2x2 x 2x2 uses Strassen with 7 multiplications
    # But the 7 multiplications involve sums of the 2x2 blocks, so we need to be careful.
    
    # Let me implement Strassen properly for 2x2 blocks.
    
    # For a 2x2 x 2x2 multiplication at block position (ai, bj) in the 4x4 x 4x4:
    # A blocks: A11=(ai,aj), A12=(ai,aj+1), A21=(ai+1,aj), A22=(ai+1,aj+1)
    # B blocks: B11=(bi,bj), B12=(bi,bj+1), B21=(bi+1,bj), B22=(bi+1,bj+1)
    # C block: (ai,bj)
    
    # The 7 Strassen multiplications for this 2x2 x 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each Mx is a product of two 2x2 matrices, which itself requires 4 scalar multiplications
    # in the schoolbook sense, but in the tensor decomposition, each Mx corresponds to
    # a specific pattern of rank-1 terms.
    
    # I think the cleanest implementation is to directly enumerate the rank-1 terms
    # for Strassen's 2x2 x 2x2 decomposition.
    
    # Let me use the explicit Strassen tensor decomposition.
    # After some thought, let me just implement schoolbook for now but with a twist:
    # I'll use the fact that we can share some computations.
    
    # Actually, let me just do schoolbook for the entire 4x4 x 4x7 to be safe,
    # but that gives rank 112 which is the baseline.
    
    # Let me try a different approach: use the known minimal rank for small cases.
    # For 2x2 x 2x2, rank is 7.
    # For 4x4 x 4x4, the best known is 49 (Strassen) or possibly 44 (other methods).
    # For 4x4 x 4x7, I don't know the exact minimal rank.
    
    # Let me implement Strassen for 2x2 x 2x2 and apply it to the 4x4 x 4x4 part.
    
    # Clear previous work
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define the 7 Strassen terms for 2x2 x 2x2 in terms of scalar (i,j,k) indices
    # where i,j are indices in A (0-1), j,k in B (0-1), k,i in C (0-1)
    # Actually, let me use local indices for the 2x2 blocks.
    
    # For a 2x2 x 2x2 block at global position:
    # A rows: 2*ai, 2*ai+1; A cols: 2*aj, 2*aj+1
    # B rows: 2*bi, 2*bi+1; B cols: 2*bj, 2*bj+1
    # C rows: 2*ai, 2*ai+1; C cols: 2*bj, 2*bj+1
    
    # The Strassen decomposition for this 2x2 x 2x2 involves 7 terms.
    # Each term is a sum of products of linear combinations of A and B entries
    # that contribute to specific C entries.
    
    # Let me define the 7 terms explicitly.
    # Using local indices (0,1) for the 2x2 blocks:
    
    # M1 = (A00 + A11) * (B00 + B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   Contributes to: C00 += M1, C11 += M1 (with appropriate signs from other terms)
    
    # This is getting very complex. Let me just use schoolbook for now.
    
    # Schoolbook decomposition for 4x4 x 4x7:
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
# EVOLVE-BLOCK-END
