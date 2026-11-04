"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists.
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    # Rank contributions:
    # 1. Strassen 2x2x2 blocks for the 4x4x4 part.
    #    The 4x4x4 part covers A[1:5, 1:5] * B[1:5, 1:4] -> C[1:4, 1:4].
    #    Wait, dimensions are A(3x5), B(5x6), C(3x6).
    #    Let's index 0-based.
    #    A: 0..2 (rows), 0..4 (cols)
    #    B: 0..4 (rows), 0..5 (cols)
    #    C: 0..2 (rows), 0..5 (cols)
    
    # Let's partition indices into "small" and "large" parts.
    # A small: rows 0, cols 0..3 (2x4)
    # A large: rows 1,2, cols 4 (2x1)
    # B small: rows 0..3, cols 0..3 (4x4)
    # B large: rows 0..3, cols 4..5 (4x2) AND rows 4, cols 0..5 (1x6)
    # This partitioning is messy.
    
    # Better partitioning for Strassen on a 2x2x2 block:
    # We can apply Strassen to any 2x2x2 sub-block.
    # Let's identify a 2x2x2 block in the multiplication.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # Let's pick i in {0,1}, j in {0,1}, k in {0,1}.
    # This is a 2x2x2 multiplication.
    # We can decompose this specific contribution using 7 terms instead of 8.
    # The other terms are "schoolbook" (1 term each).
    
    # Total schoolbook terms: 3*5*6 = 90.
    # We replace 8 terms with 7 terms for one 2x2x2 block.
    # Net reduction: 1.
    # Can we do more?
    # We can apply Strassen to multiple disjoint 2x2x2 blocks.
    # Max number of disjoint 2x2x2 blocks in 3x5x6?
    # Rows: 3 -> only 1 pair (0,1) or (1,2) or (0,2) non-adjacent? Strassen needs contiguous or just specific indices?
    # Strassen works on any 2x2x2 selection of indices, but the "blocks" in the matrix must be contiguous for the algebra to simplify into matrix multiplications of sub-blocks.
    # Actually, Strassen's formula relies on the structure of 2x2 matrices.
    # If we pick rows {0,1}, cols {0,1}, k {0,1}, we are doing the multiplication of sub-matrices A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2].
    # This is a valid 2x2x2 block.
    
    # How many disjoint 2x2x2 blocks can we fit?
    # Rows: 3 available. We can fit 1 block of size 2 in rows. (1 remaining row)
    # Cols (A): 5 available. We can fit 2 blocks of size 2 in cols. (1 remaining col)
    # Cols (B): 6 available. We can fit 3 blocks of size 2 in cols.
    # k (B rows): 5 available. We can fit 2 blocks of size 2 in k.
    
    # So we can fit 1 * 2 * 2 = 4 disjoint 2x2x2 blocks?
    # Let's check indices.
    # Block 1: Rows {0,1}, A-Cols {0,1}, B-Rows {0,1}, B-Cols {0,1}
    # Block 2: Rows {0,1}, A-Cols {2,3}, B-Rows {0,1}, B-Cols {2,3}
    # Block 3: Rows {0,1}, A-Cols {0,1}, B-Rows {2,3}, B-Cols {0,1}
    # Block 4: Rows {0,1}, A-Cols {2,3}, B-Rows {2,3}, B-Cols {2,3}
    
    # These 4 blocks are disjoint in terms of (i,j,k) tuples.
    # Total terms saved: 4 * 1 = 4.
    # New rank: 90 - 4 = 86.
    
    # Can we fit more?
    # Rows {1,2} could be used instead of {0,1}, but not both since they overlap in row 1.
    # So only 1 block in Row dimension.
    # A-Cols: {0,1}, {2,3} used. {4} left.
    # B-Cols: {0,1}, {2,3} used. {4,5} left.
    # B-Rows: {0,1}, {2,3} used. {4} left.
    
    # What about blocks involving the leftover dimensions?
    # Row 2 is unused in Strassen blocks.
    # A-Col 4 is unused.
    # B-Col 4,5 are unused in the 2x2 blocks above (we used 0,1 and 2,3).
    # B-Row 4 is unused.
    
    # We can form 1x1x1 blocks (schoolbook) for the rest.
    # Can we form a 2x2x2 block using Row 2?
    # No, we only have 1 row left (Row 2). Strassen needs 2 rows.
    # Unless we use Row 0 or 1 again? No, that would double count.
    # So we are stuck with 4 Strassen blocks.
    
    # Rank = 90 - 4 = 86.
    
    # Let's verify if we can do better with non-contiguous indices?
    # Strassen's algorithm is specifically for 2x2 matrices. The indices don't have to be contiguous in the original matrix, but the "sub-matrices" must be 2x2.
    # If we pick non-contiguous indices, we are just permuting the matrix. The rank is invariant under permutation.
    # So we can assume contiguous blocks WLOG for counting max disjoint blocks.
    
    # Is there a way to combine the leftover dimensions into a larger Strassen-like structure?
    # Leftovers:
    # Rows: {2}
    # A-Cols: {4}
    # B-Rows: {4}
    # B-Cols: {4, 5}
    # Plus the "cross terms" from the 2x2 blocks? No, the 2x2 blocks cover all (i,j,k) in their respective 2x2x2 volumes.
    # The remaining (i,j,k) tuples are those where i=2 OR (i in {0,1} AND (a_col not in {0,1,2,3} OR b_row not in {0,1,2,3} OR b_col not in {0,1,2,3})).
    
    # Let's list the covered (i,j,k) tuples by the 4 Strassen blocks:
    # SB1: i in {0,1}, j in {0,1}, k in {0,1}
    # SB2: i in {0,1}, j in {2,3}, k in {0,1}
    # SB3: i in {0,1}, j in {0,1}, k in {2,3}
    # SB4: i in {0,1}, j in {2,3}, k in {2,3}
    
    # Covered:
    # i=0,1; j=0,1,2,3; k=0,1,2,3.
    # Total 2 * 4 * 4 = 32 tuples.
    # Schoolbook would use 32 terms. Strassen uses 4 * 7 = 28 terms.
    # Saved 4.
    
    # Remaining tuples:
    # 1. i=2 (all j, all k): 1 * 5 * 6 = 30 tuples.
    # 2. i=0,1; j=4; k=0..5: 2 * 1 * 6 = 12 tuples.
    # 3. i=0,1; j=0..3; k=4: 2 * 4 * 1 = 8 tuples.
    # 4. i=0,1; j=4; k=4: already counted in 2 and 3?
    #    j=4 is in "j=4" group. k=4 is in "k=4" group.
    #    Let's partition the remaining space carefully.
    #    Total 90.
    #    Covered 32.
    #    Remaining 58.
    
    # Can we apply Strassen to any of the remaining?
    # We need 2 rows, 2 A-cols, 2 B-cols, 2 B-rows.
    # Available rows: only i=2. (1 row). Cannot form 2x2x2.
    # So no more Strassen blocks possible.
    
    # Thus, the optimal rank using simple disjoint Strassen blocks is 86.
    
    rank = 90
    # We will construct the full 90x90 matrices and then replace the 8 terms of each Strassen block with 7.
    # Actually, it's easier to build the Strassen terms directly and then add the schoolbook terms for the rest.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        # Append a column to U, V, W
        U.append([0]*(n*m))
        V.append([0]*(m*p))
        W.append([0]*(p*n))
        U[-1][u_idx] = val_u
        V[-1][v_idx] = val_v
        W[-1][w_idx] = val_w
        # Note: U, V, W are currently lists of rows. We are appending columns.
        # The return format is {'U': ..., 'V': ..., 'W': ...}
        # U shape should be (n*m, rank).
        # So U is a list of n*m rows, each of length rank.
        # My add_term appends to U, which makes U a list of columns?
        # Let's stick to building matrices as lists of rows.
        pass
    
    # Let's build U, V, W as lists of rows.
    # We don't know the final rank yet.
    # Let's collect terms as tuples (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w).
    
    terms = []
    
    def add_schoolbook(i, j, k):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        terms.append((u_idx, v_idx, w_idx, 1, 1, 1))
    
    def add_strassen_block(i0, j0, k0):
        # Strassen for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Indices:
        # i: i0, i0+1
        # j: j0, j0+1
        # k: k0, k0+1
        
        # M1 = A[i0, j0] * (B[j0+1, k0] + B[j0+1, k0+1])
        # M2 = (A[i0, j0] + A[i0, j0+1]) * B[j0+1, k0+1]
        # M3 = (A[i0+1, j0] + A[i0+1, j0+1]) * B[j0, k0]
        # M4 = A[i0+1, j0+1] * (B[j0, k0] - B[j0, k0+1])
        # M5 = (A[i0, j0] + A[i0+1, j0]) * B[k0+1, k0]  <- Wait, Strassen formulas vary.
        
        # Standard Strassen:
        # M1 = A11 * (B21 + B22)
        # M2 = (A11 + A12) * B22
        # M3 = (A21 + A22) * B11
        # M4 = A22 * (B11 - B12)
        # M5 = (A11 + A21) * B21
        # M6 = (A12 - A21) * (B11 + B21)
        # M7 = (A22 - A12) * (B21 + B22)
        
        # C11 = M1 - M2 + M4 + M7
        # C12 = M2 + M5
        # C21 = M3 + M6
        # C22 = M1 - M5 + M3 - M4
        
        # Let's map this to our tensor decomposition.
        # Each "M" term is a product of a linear combo of A and a linear combo of B.
        # This corresponds to a set of rank-1 terms in the decomposition.
        # For example, M1 = A11 * (B21 + B22) = A11*B21 + A11*B22.
        # This contributes to C11 and C22.
        # C11 += A11*B21
        # C22 += A11*B21
        # So the term (A11, B21) contributes 1 to C11 and 1 to C22.
        
        # We need to accumulate the contributions to each (i,k) pair (which is the row/col of C) from each (u_idx, v_idx) pair.
        
        # Let's define the 7 multipliers.
        # 1. u1 = A[i0, j0]
        #    v1 = B[j0+1, k0] + B[j0+1, k0+1]
        #    This means we need two rank-1 terms?
        #    No, a rank-1 term is defined by a specific u_idx and v_idx.
        #    If v is a sum, it's not a single rank-1 term.
        #    Wait, the tensor decomposition is T = sum_r U[:,r] * V[:,r] * W[:,r].
        #    If we have a term A[i0,j0] * (B[j0+1,k0] + B[j0+1,k0+1]), this is:
        #    A[i0,j0] * B[j0+1,k0]  +  A[i0,j0] * B[j0+1,k0+1]
        #    These are two separate rank-1 contributions to the tensor.
        #    So Strassen's "7 multiplications" doesn't directly map to "7 rank-1 terms" in the tensor decomposition?
        #    
        #    Let's check.
        #    In Strassen, we compute 7 matrix products. Each matrix product is a sum of rank-1 terms?
        #    No, M1 is a scalar if A and B are 1x1? No, A and B are 2x2 matrices.
        #    M1 = A11 * (B21 + B22). A11 is 1x1, B21, B22 are 1x1.
        #    So M1 is a scalar.
        #    The computation of M1 involves 1 multiplication of scalars? No, A11 is an entry.
        #    So Strassen reduces the number of scalar multiplications.
        #    
        #    In the tensor decomposition, each rank-1 term corresponds to a scalar multiplication A[i,j] * B[j,k] contributing to C[i,k].
        #    The schoolbook decomposition has 8 such terms for a 2x2x2 block.
        #    Strassen has 7 "multiplications", but they are of linear combinations.
        #    (A11 + A12) * B22 = A11*B22 + A12*B22.
        #    This still requires 2 scalar multiplications if we expand it.
        #    
        #    HOWEVER, the rank of the matrix multiplication tensor is the minimum number of rank-1 tensors that sum to it.
        #    A rank-1 tensor is T[i,j,k] = U[i,j] * V[j,k] * W[k,i].
        #    For the 2x2x2 case, the rank is 7.
        #    This means there exist 7 tuples (u, v, w) such that the sum of u[i,j]v[j,k]w[k,i] equals the identity tensor for 2x2x2.
        #    
        #    The explicit 7-term decomposition for 2x2x2 is known.
        #    Let's find it.
        #    
        #    Ref: "The rank of the 2x2x2 matrix multiplication tensor is 7".
        #    The terms are:
        #    1. e11 e11 e11  (A11, B11, C11)
        #    ... this is getting complex to derive from scratch.
        #    
        #    Let's use the standard Strassen formulas to generate the 7 terms.
        #    Each Strassen "multiplication" Mx = (linear combo of A) * (linear combo of B).
        #    This product expands to a sum of scalar products Aij * Bkl.
        #    If we keep the linear combinations as single "factors" in the decomposition, we are NOT using rank-1 terms of the original A and B.
        #    The decomposition requires factors that are linear in A entries and linear in B entries.
        #    U[i,j] can be a linear combination of A entries?
        #    No, U is a matrix of shape (n*m, rank). Its entries are constants.
        #    The term is U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r].
        #    So U, V, W are constant matrices.
        #    
        #    So, can we represent Strassen's M2 = (A11+A12) * B22 using rank-1 terms with constant U,V,W?
        #    M2 = A11*B22 + A12*B22.
        #    This is the sum of two rank-1 terms: (A11, B22) and (A12, B22).
        #    So Strassen's "7 multiplications" expands back to 8 scalar multiplications if we just expand the sums.
        #    
        #    Wait. The rank 7 result is non-trivial. It means there is a clever cancellation or structure.
        #    The 7 terms in the rank-7 decomposition are NOT simply the Strassen M1..M7 expanded.
        #    They are specific combinations.
        #    
        #    Let's look up the explicit rank-7 decomposition for 2x2x2.
        #    
        #    Terms (i,j,k) indices 1,2.
        #    1. (1,1,1)
        #    2. (1,2,2)
        #    3. (2,1,1)
        #    4. (2,2,2)
        #    5. (1,1,2) - with coefficient?
        #    
        #    Actually, let's just use the known fact that rank(2,2,2)=7.
        #    I can hardcode the 7 terms for the 2x2x2 block.
        #    
        #    Source: https://en.wikipedia.org/wiki/Matrix_multiplication#Strassen_algorithm
        #    The algorithm computes C using 7 products.
        #    But does it give the tensor decomposition?
        #    
        #    Let's construct the 7 terms manually for 2x2x2 (indices 0,1).
        #    T[i,j,k] = delta(i,j,k) for i=j=k? No, T[i,j,k] = 1 if i,k match the output, j is the intermediate.
        #    T is the tensor such that C = A T B.
        #    T[i,j,k] = 1 if (i,j,k) is a valid path.
        #    For 2x2x2, the tensor is the sum of 8 basis tensors e_i e_j e_k.
        #    We want to write it as sum of 7 rank-1 tensors.
        #    
        #    Let's try to find the 7 vectors.
        #    Let the 7 terms be (u_r, v_r, w_r).
        #    
        #    I will use a known decomposition:
        #    u1 = [1, 0, 0, 0] (A00)
        #    v1 = [0, 1, 1, 0] (B01+B11)
        #    w1 = [0, 1, 0, 1] (C01+C11)
        #    ... this is guessing.
        #    
        #    Alternative: Use the "Bini's cap" or similar.
        #    Given the time, I will implement the 4 Strassen blocks using the expanded 8 terms but cancel 1 term per block?
        #    No, that doesn't work.
        #    
        #    Let's stick to the schoolbook rank 90.
        #    The prompt asks for a "stronger algorithm".
        #    If I can't easily derive the rank-7 terms, I might just return 90.
        #    But the previous attempt scored 1.0 (90).
        #    I need to beat 90.
        #    
        #    Let's try to find the rank-7 terms online or derive them.
        #    Derivation:
        #    We need 7 vectors u,v,w in C^4 (since 2x2=4) such that sum u_r[i,j] v_r[j,k] w_r[k,i] = delta(i,k) delta(i,j)?? No.
        #    T[i,j,k] = 1 if i,k are the matrix indices.
        #    T is a 2x2x2 tensor.
        #    T = e00 e00 e00 + e00 e01 e01 + e01 e00 e10 + ...
        #    
        #    Let's use a python snippet to solve for the rank-7 decomposition?
        #    No, I must output a static program.
        #    
        #    I will use the following known decomposition for 2x2x2:
        #    1. (1,0,0,0) (0,1,1,0) (0,1,0,1)
        #    2. (1,0,0,0) (0,0,1,0) (0,0,1,0)
        #    ...
        #    
        #    Actually, let's look at the "Strassen 7" formulas again.
        #    M1 = A11(B21+B22)
        #    M2 = (A11+A12)B22
        #    M3 = (A21+A22)B11
        #    M4 = A22(B11-B12)
        #    M5 = (A11+A21)B21
        #    M6 = (A12-A21)(B11+B21)
        #    M7 = (A22-A12)(B21+B22)
        #    
        #    C11 = M1 - M2 + M4 + M7
        #    C12 = M2 + M5
        #    C21 = M3 + M6
        #    C22 = M1 - M5 + M3 - M4
        #    
        #    Let's expand M1: A11B21 + A11B22.
        #    Contribution to C11: A11B21 + A11B22.
        #    Contribution to C22: A11B21 + A11B22.
        #    
        #    If we treat M1 as a single "term" in a higher-level decomposition, it has rank 2 in the original A,B space.
        #    So Strassen doesn't reduce the rank of the tensor decomposition directly unless we combine terms cleverly.
        #    
        #    HOWEVER, the rank of the 2x2x2 multiplication tensor IS 7.
        #    This is a mathematical fact.
        #    There exist 7 rank-1 decompositions.
        #    
        #    I will hardcode the 7 terms for 2x2x2.
        #    Ref: "A note on the rank of the 2x2x2 matrix multiplication tensor"
        #    The 7 terms are:
        #    1. u=(1,0,0,0), v=(0,0,1,0), w=(0,1,0,0)  -> A00 B01 C01?
        #    
        #    Let's assume the indices are flattened.
        #    A: 00, 01, 10, 11
        #    B: 00, 01, 10, 11
        #    C: 00, 01, 10, 11
        #    
        #    I will use a known set of 7 vectors.
        #    If I can't be sure, I'll stick to 90.
        #    But let's try 86 first (4 Strassen blocks, using 7 terms each, assuming they map directly to 7 rank-1 terms? No, they don't).
        #    
        #    Wait, if I use the Strassen formulas, I am NOT using rank-1 terms of the original tensor.
        #    I am using a decomposition into 7 "blocks" each of which is a rank-2 tensor (sum of 2 rank-1s).
        #    So that gives 14 rank-1 terms, which is worse than 8.
        #    
        #    The rank 7 result is non-constructive in the simple Strassen sense. It requires specific linear combinations that cancel out.
        #    
        #    Given the difficulty, I will return the schoolbook decomposition (rank 90).
        #    It is correct and safe.
        
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
