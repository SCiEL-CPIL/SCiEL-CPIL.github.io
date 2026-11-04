import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _build_strassen_2x2():
    """
    Returns (U, V, W) as numpy arrays of shape (4, 7) for 2x2x2 matrix mult.
    U[a_idx, t], V[b_idx, t], W[c_idx, t]
    where a_idx = i*2+j for A[i,j], b_idx = j*2+k for B[j,k], c_idx = k*2+i for C[k,i]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices (k*n+i): C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # Term 1: (A10+A11)B00 -> C10+=1, C11-=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # Term 2: A00(B01-B11) -> C01+=1, C11+=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # Term 3: A11(B10-B00) -> C10+=1, C00+=1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[2,3] = 1; W[0,3] = 1
    
    # Term 4: (A00+A01)B11 -> C01-=1, C11+=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = -1; W[3,4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11+=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00+=1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, max_checks=50):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[k*n+i] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    # For each t: contribution to C is W_col_t * outer(U_col_t, V_col_t)
    # C = sum_t W[:,t] * (U[:,t]^T A) * B V[:,t] ... let's just do direct check
    
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for _ in range(max_checks):
        A = np.random.randint(-5, 5, size=(n, m), dtype=float)
        B = np.random.randint(-5, 5, size=(m, p), dtype=float)
        C_expected = A @ B
        
        # Compute via decomposition
        # For each term t:
        #   a_t = U[:, t] reshaped as (n, m) dotted with A -> scalar
        #   b_t = V[:, t] reshaped as (m, p) dotted with B -> scalar
        #   C += W[:, t] * a_t * b_t (W[:, t] reshaped as (p, n))
        
        C_computed = np.zeros((p, n), dtype=float)
        for t in range(rank):
            a_vec = U[:, t].reshape(n, m)
            b_vec = V[:, t].reshape(m, p)
            w_vec = W[:, t].reshape(p, n)
            a_scalar = np.sum(a_vec * A)
            b_scalar = np.sum(b_vec * B)
            C_computed += w_vec * a_scalar * b_scalar
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _build_block_circulant_decomposition():
    """
    Build a rank-56 decomposition for 4x4 x 4x5 matrix multiplication
    using 2x2 Strassen blocks with cyclic shifts for the 5 output columns.
    
    Strategy:
    - Split A into 4 blocks of 2x2: A00, A01, A10, A11 (rows 0-1,2-3; cols 0-1,2-3)
    - Split B into 2 blocks of 2x5: B_top (rows 0-1), B_bot (rows 2-3)
    - For each of the 4 output column pairs (k-pairs), use Strassen on the 2x2 A blocks
      combined with appropriate 2x5 B sub-blocks via cyclic convolution.
    
    Actually, let me use a simpler approach:
    - Use 7 Strassen terms for the 2x2x2 core
    - For the 4x4 x 4x5 case, we have 2x2 block structure in rows/cols of A
    - B has 4 rows, 5 cols. Split B into 2x2 blocks where possible.
    
    Let me try: rank = 4 * 7 = 28? No, that's too optimistic.
    
    Better approach: Use the fact that we can do 2x2 Strassen for each pair of 
    output columns. We have 5 output columns, so we need ceil(5/2) = 3 column groups.
    But the coupling makes this tricky.
    
    Let me just build a structured rank-56 solution:
    - 7 Strassen terms per 2x2x2 block
    - 4 such blocks (for the 4x4 A)
    - But B has 5 columns, so we handle it with 3 groups: cols 0-1, 2-3, 4
    - Total: 7 * (4 + 4 + 2) = 7 * 10 = 70? Still not 56.
    
    Let me try a different structured approach:
    - Use 2x2 Strassen for the row dimension (4 rows = 2x2 blocks)
    - For each Strassen term, we get a 2x5 result from 2x2 A block and 2x5 B block
    - That's 7 terms * 4 column groups... 
    
    Actually, let me just do: for each of the 4 row-block-pairs and handle columns.
    
    Simplest correct approach with rank < 80:
    - Treat as 2x2 block matrix multiplication where A is 2x2 blocks of 2x2,
      B is 2x2 blocks of 2x5 (last block is 2x5 not 2x2)
    - Use Strassen for the 2x2 block structure
    - Each Strassen term multiplies 2x2 by 2x5 -> 2x5 result
    - Rank per term: 2*5 = 10? No, that's schoolbook within each term.
    
    Let me just build rank-56 directly with a verification.
    """
    n, m, p = 4, 4, 5
    rank = 56
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Use 2x2 Strassen for the 4x4 A matrix (2x2 blocks of 2x2)
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01, B02, B03, B04], [B10, B11, B12, B13, B14]] where each Bj is 2x5
    # C = A*B, C is 4x5
    
    # Split B into pairs of columns: (0,1), (2,3), (4)
    # For cols (0,1): B_top = B[0:2, 0:2], B_bot = B[2:4, 0:2]
    # For cols (2,3): B_top = B[0:2, 2:4], B_bot = B[2:4, 2:4]
    # For col 4: B_top = B[0:2, 4:5], B_bot = B[2:4, 4:5]
    
    # Strassen terms for 2x2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 - M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each column group, we apply Strassen to get 2x2 block of C
    # Then expand to 2x5
    
    # Column group 0: cols 0,1 -> C[:, 0:2] is 4x2
    #   C_top = C[0:2, 0:2] (2x2), C_bot = C[2:4, 0:2] (2x2)
    #   Using Strassen with B_top = B[0:2, 0:2], B_bot = B[2:4, 0:2]
    
    # Column group 1: cols 2,3 -> C[:, 2:4] is 4x2
    #   C_top = C[0:2, 2:4] (2x2), C_bot = C[2:4, 2:4] (2x2)
    #   Using Strassen with B_top = B[0:2, 2:4], B_bot = B[2:4, 2:4]
    
    # Column group 2: col 4 -> C[:, 4:5] is 4x1
    #   This is trickier - we need a 2x2x1 decomposition
    
    # Let me handle col 4 separately with rank 4 (schoolbook for 2x2x1)
    # Total rank: 7 + 7 + 4 = 18? That's too low, something's wrong.
    
    # Wait - the Strassen terms multiply 2x2 by 2x2 to get 2x2.
    # For col group 0: we multiply A_blocks (2x2 of 2x2) by B_cols_0_1 (2x2 of 2x2)
    #   This gives 2x2 of 2x2 = 4x2 result. Rank = 7.
    # For col group 1: same, rank = 7.
    # For col group 2 (single col): multiply A_blocks by B_col_4 (2x1 of 2x1)
    #   This is 2x2 x 2x1 = 2x2 of 2x1 = 4x1. 
    #   We can use 4 schoolbook terms: rank = 4.
    # Total: 7 + 7 + 4 = 18.
    
    # But wait, this isn't right either. The Strassen decomposition of 2x2x2
    # has rank 7, but when we extend to 2x2x2x2 (i.e., 4x4x2), we need
    # to apply Strassen to the outer 2x2 structure, and each inner multiplication
    # is 2x2 x 2x2 = rank 4 schoolbook.
    
    # So for 4x4 x 4x2: rank = 7 * 4 = 28
    # For 4x4 x 4x1: rank = 4 * 4 = 16 (schoolbook for each of 4 A-blocks)
    # Total for 4x4 x 4x5: 28 + 28 + 16 = 72
    
    # Hmm, let me reconsider. Let me just build it carefully.
    
    # A is 4x4, split into 2x2 blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # B is 4x5. Split rows into 2x2 blocks:
    # B0 = B[0:2, :], B1 = B[2:4, :]
    
    # C = A*B is 4x5. Split into:
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C02 = C[0:2, 4:5]
    # C10 = C[2:4, 0:2], C11 = C[2:4, 2:4], C12 = C[2:4, 4:5]
    
    # C00 = A00*B0[:,0:2] + A01*B1[:,0:2]
    # C01 = A00*B0[:,2:4] + A01*B1[:,2:4]
    # C02 = A00*B0[:,4:5] + A01*B1[:,4:5]
    # C10 = A10*B0[:,0:2] + A11*B1[:,0:2]
    # C11 = A10*B0[:,2:4] + A11*B1[:,2:4]
    # C12 = A10*B0[:,4:5] + A11*B1[:,4:5]
    
    # For C00, C10, C11: use Strassen on the 2x2 block structure
    # with B having 2x2 sub-blocks for cols 0-1 and 2-3.
    
    # Let me define for cols 0-1:
    # B0_01 = B[0:2, 0:2], B1_01 = B[2:4, 0:2]
    # Strassen terms:
    # M0 = (A00+A11) * (B0_01 + B1_01)
    # M1 = (A10+A11) * B0_01
    # M2 = A00 * (B1_01 - B0_01)
    # M3 = A11 * (B0_01 - B1_01)
    # M4 = (A00+A01) * B1_01
    # M5 = (A10-A00) * (B0_01 + B1_01)
    # M6 = (A01-A11) * (B0_01 + B1_01)
    
    # C00 = M0 + M3 - M4 - M5  (each Mi is 2x2)
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5  (wait, this is for cols 0-1)
    
    # Actually for cols 0-1, the "C11" in Strassen is C[2:4, 0:2] = C10 in my notation.
    # Let me redo: Strassen gives:
    # C_top_left = M0 + M3 - M4 - M5
    # C_top_right = M2 + M4
    # C_bot_left = M1 + M3
    # C_bot_right = M0 - M1 + M2 + M5
    
    # But here "right" means cols 2-3, not cols 1. So this Strassen applies to
    # the 2x2 block structure where each block is 2x2, giving a 4x4 result for
    # the first 4 columns.
    
    # So for cols 0-3:
    # C[0:2, 0:2] = M0 + M3 - M4 - M5
    # C[0:2, 2:4] = M2 + M4
    # C[2:4, 0:2] = M1 + M3
    # C[2:4, 2:4] = M0 - M1 + M2 + M5
    
    # Where each Mi is a 2x2 matrix product computed via schoolbook (rank 4 each).
    # Total rank for cols 0-3: 7 * 4 = 28.
    
    # For col 4:
    # C[0:2, 4] = A00 * B[0:2, 4] + A01 * B[2:4, 4]
    # C[2:4, 4] = A10 * B[0:2, 4] + A11 * B[2:4, 4]
    # Schoolbook: 4 terms, each rank 2 (2x2 x 2x1 = 2x1, needs 2 mults)
    # Total rank for col 4: 4 * 2 = 8.
    
    # Grand total: 28 + 8 = 36.
    
    # Let me build this!
    
    col = 0
    # Strassen terms for cols 0-3, each term is a 2x2 x 2x2 product (rank 4)
    # A blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B blocks for cols 0-1: B00 = B[0:2,0:2], B10 = B[2:4,0:2]
    # B blocks for cols 2-3: B01 = B[0:2,2:4], B11 = B[2:4,2:4]
    
    # Strassen term 0: (A00+A11)(B00+B10)
    # A part: A00 + A11
    # B part: B00 + B10
    for i in range(2):
        for j in range(2):
            # A00[i,j]
            U[(0)*4 + i*2 + j, col] = 1
            # A11[i,j]
            U[(2)*4 + i*2 + j, col] = 1
            # B00[i,j]
            V[(0)*5 + i*2 + j, col] = 1
            # B10[i,j]
            V[(2)*5 + i*2 + j, col] = 1
            # C[0:2, 0:2][i,j] += 1
            W[(0)*4 + i*2 + j, col] = 1
            # C[2:4, 2:4][i,j] += 1
            W[(2)*4 + i*2 + j, col] = 1
    col += 1
    
    # Wait, I'm confusing myself. Each Strassen term is ONE scalar multiplication
    # of two 2x2 matrices, which requires 4 scalar multiplications (schoolbook).
    # So each Strassen term contributes 4 columns to the decomposition.
    
    # Let me restart with a cleaner approach.
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper: add a schoolbook 2x2 x 2x2 product term
    # a_rows, a_cols: row/col offsets of the 2x2 A sub-block
    # b_rows, b_cols: row/col offsets of the 2x2 B sub-block
    # c_rows, c_cols: row/col offsets of the 2x2 C sub-block
    # coeff_a, coeff_b, coeff_c: coefficients
    def add_2x2_term(a_r, a_c, b_r, b_c, c_r, c_c, ca, cb, cc):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[(a_r+i)*4 + (a_c+j), col] += ca
                V[(b_r+i)*5 + (b_c+j), col] += cb
                W[(c_r+i)*4 + (c_c+j), col] += cc
        col += 1
    
    # Strassen terms for 4x4 x 4x4 (cols 0-3):
    # M0 = (A00+A11)(B00+B10)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)
    # M3 = A11(B00-B10)
    # M4 = (A00+A01)B10
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B00+B10)
    
    # But B has 5 columns, so I need to handle cols 0-1 and 2-3 separately
    # for the Strassen structure.
    
    # For cols 0-1:
    # B_top_01 = B[0:2, 0:2], B_bot_01 = B[2:4, 0:2]
    # M0_01 = (A00+A11)(B_top_01 + B_bot_01)
    # ... etc
    
    # This is getting complex. Let me just build rank-56 with a simpler structure.
    
    # Approach: Use 2x2 Strassen for the row dimension only.
    # A = [[A_top], [A_bot]] where A_top is 2x4, A_bot is 2x4
    # B = [[B_top], [B_bot]] where B_top is 2x5, B_bot is 2x5
    # C = A_top * B_top + A_bot * B_bot... no that's not right.
    
    # C = A * B where A is 4x4, B is 4x5
    # C[0:2, :] = A[0:2, :] * B
    # C[2:4, :] = A[2:4, :] * B
    
    # This doesn't decompose nicely with Strassen.
    
    # Let me just use a known result: the rank of 4x4x5 matrix multiplication
    # is at most 4*4*5 = 80 (schoolbook). A better bound exists.
    # 
    # Let me try: rank = 56 using a structured approach.
    # 
    # I'll use the following: for each of the 4 rows of C, compute using
    # 14 terms (4*14 = 56). Each term is a product of a linear form in A
    # and a linear form in B.
    #
    # Actually, let me just verify that a rank-56 solution works by constructing
    # one carefully.
    
    # New approach: Use 7 Strassen terms, each computing a 2x2 x 2x5 product
    # via schoolbook (10 scalar mults each). Total: 7*10 = 70.
    # Then for the remaining column, add 4*1 = 4 more. Total: 74.
    
    # Or: 7 Strassen terms for 2x2x2 (rank 7 each, total 49) + 
    #     7 terms for the extra column (rank 7) = 56!
    
    # Let me try this:
    # For cols 0-1: use Strassen 2x2x2, rank 7
    # For cols 2-3: use Strassen 2x2x2, rank 7  
    # For col 4: use 4x2x1 = 8 terms? Or 4x1x1 = 4 terms?
    
    # 7 + 7 + 4 = 18? That can't be right for 4x4x5.
    
    # I think the issue is that each "Strassen 2x2x2" term when extended to
    # 4x4x2 requires 7*4 = 28 scalar multiplications.
    
    # Let me just go with rank = 56 and construct it via a different method.
    
    # Method: Use the fact that we can compute 4x4 x 4x5 as:
    # - 2 Strassen 4x4x2 computations (for cols 0-1 and 2-3): rank 2*28 = 56
    # - Plus 4x4x1 for col 4: rank 16
    # Total: 72
    
    # Or:
    # - 1 Strassen 4x4x4 (for cols 0-3): rank 28
    # - 1 schoolbook 4x4x1 (for col 4): rank 16
    # Total: 44
    
    # Let me build rank 44!
    
    rank = 44
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Part 1: Strassen for 4x4 x 4x4 (cols 0-3), rank 28
    # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    
    # M0 = (A00+A11)(B00+B10)
    # Each Mi is a 2x2 x 2x2 product = 4 scalar mults
    
    def add_block_product(a_block, b_block, c_block, coeff=1):
        """Add a schoolbook 2x2x2 product.
        a_block: (row_offset, col_offset) of 2x2 A block
        b_block: (row_offset, col_offset) of 2x2 B block  
        c_block: (row_offset, col_offset) of 2x2 C block
        """
        nonlocal col
        ar, ac = a_block
        br, bc = b_block
        cr, cc = c_block
        for i in range(2):
            for j in range(2):
                U[(ar+i)*4 + (ac+j), col] = coeff
                V[(br+i)*5 + (bc+j), col] = coeff
                W[(cr+i)*4 + (cc+j), col] = coeff
                col += 1
    
    # M0 = (A00+A11)(B00+B10)
    # = A00*B00 + A00*B10 + A11*B00 + A11*B10
    # But Strassen combines them into one "multiplication" of sums.
    # In the tensor decomposition, each scalar mult is separate.
    
    # Actually, in a tensor decomposition, each rank-1 term is:
    # C += u * v * w where u is a vector in A-space, v in B-space, w in C-space.
    # For schoolbook 2x2x2, we have 4 terms:
    # C[i,j] += A[i,j] * B[j,k] ... wait, for 2x2x2:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 4 terms: (i,j,k) for i,j,k in {0,1}
    
    # For Strassen, we have 7 terms where each term is a product of
    # linear combinations. In the tensor decomposition, each Strassen term
    # expands to multiple rank-1 terms.
    
    # Let me just do schoolbook for everything and see if I can get below 80.
    
    # Schoolbook 4x4x5: 4*4*5 = 80 terms.
    
    # Let me try a hybrid: Strassen for the 2x2 block structure.
    # The 7 Strassen terms each involve products of 2x2 matrices.
    # Each 2x2 x 2x2 product = 4 scalar mults.
    # Total: 7 * 4 = 28 for the 4x4x4 part.
    
    # For col 4: 4x4x1 = 16 scalar mults.
    
    # Total: 44.
    
    # Let me build this properly.
    
    col = 0
    
    # Strassen term 0: (A00+A11)(B00+B10)
    # A00 = A[0:2, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B10 = B[2:4, 0:2]
    # Result adds to C[0:2, 0:2] and C[2:4, 2:4]
    
    # The product (A00+A11)(B00+B10) is a 2x2 matrix.
    # Let P = A00+A11, Q = B00+B10
    # P[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0,1
    # Q[i,j] = B[i,j] + B[i+2,j] for i,j in 0,1 (B rows 0-1 and 2-3, cols 0-1)
    # R = P*Q, R[i,k] = sum_j P[i,j]*Q[j,k]
    
    # R contributes to C[0:2, 0:2] with +1 and C[2:4, 2:4] with +1
    
    # For each (i,j,k) in 2x2x2:
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i,j] + A[i+2, j+2]
                U[i*4 + j, col] = 1
                U[(i+2)*4 + (j+2), col] = 1
                # B part: B[j,k] + B[j+2, k]
                V[j*5 + k, col] = 1
                V[(j+2)*5 + k, col] = 1
                # C part: C[i,k] += 1, C[i+2, k+2] += 1
                W[i*4 + k, col] = 1
                W[(i+2)*4 + (k+2), col] = 1
                col += 1
    
    # Strassen term 1: (A10+A11)B00
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2]
    # P = A10+A11, P[i,j] = A[i+2,j] + A[i+2,j+2]
    # Q = B00, Q[i,j] = B[i,j]
    # R = P*Q
    # R contributes to C[2:4, 0:2] with +1 and C[2:4, 2:4] with -1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i+2,j] + A[i+2,j+2]
                U[(i+2)*4 + j, col] = 1
                U[(i+2)*4 + (j+2), col] = 1
                # B part: B[j,k]
                V[j*5 + k, col] = 1
                # C part: C[i+2,k] += 1, C[i+2,k+2] -= 1
                W[(i+2)*4 + k, col] = 1
                W[(i+2)*4 + (k+2), col] = -1
                col += 1
    
    # Strassen term 2: A00(B10-B00)
    # A00 = A[0:2, 0:2]
    # B10 = B[2:4, 0:2], B00 = B[0:2, 0:2]
    # P = A00, Q = B10-B00
    # Q[i,j] = B[i+2,j] - B[i,j]
    # R = P*Q
    # R contributes to C[0:2, 2:4] with +1 and C[2:4, 2:4] with +1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i,j]
                U[i*4 + j, col] = 1
                # B part: B[j+2,k] - B[j,k]
                V[(j+2)*5 + k, col] = 1
                V[j*5 + k, col] = -1
                # C part: C[i,k+2] += 1, C[i+2,k+2] += 1
                W[i*4 + (k+2), col] = 1
                W[(i+2)*4 + (k+2), col] = 1
                col += 1
    
    # Strassen term 3: A11(B00-B10)
    # A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B10 = B[2:4, 0:2]
    # P = A11, Q = B00-B10
    # Q[i,j] = B[i,j] - B[i+2,j]
    # R = P*Q
    # R contributes to C[2:4, 0:2] with +1 and C[0:2, 0:2] with +1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i+2,j+2]
                U[(i+2)*4 + (j+2), col] = 1
                # B part: B[j,k] - B[j+2,k]
                V[j*5 + k, col] = 1
                V[(j+2)*5 + k, col] = -1
                # C part: C[i+2,k] += 1, C[i,k] += 1
                W[(i+2)*4 + k, col] = 1
                W[i*4 + k, col] = 1
                col += 1
    
    # Strassen term 4: (A00+A01)B10
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # B10 = B[2:4, 0:2]
    # P = A00+A01, P[i,j] = A[i,j] + A[i,j+2]
    # Q = B10, Q[i,j] = B[i+2,j]
    # R = P*Q
    # R contributes to C[0:2, 2:4] with -1 and C[2:4, 2:4] with +1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i,j] + A[i,j+2]
                U[i*4 + j, col] = 1
                U[i*4 + (j+2), col] = 1
                # B part: B[j+2,k]
                V[(j+2)*5 + k, col] = 1
                # C part: C[i,k+2] -= 1, C[i+2,k+2] += 1
                W[i*4 + (k+2), col] = -1
                W[(i+2)*4 + (k+2), col] = 1
                col += 1
    
    # Strassen term 5: (A10-A00)(B00+B10)
    # A10 = A[2:4, 0:2], A00 = A[0:2, 0:2]
    # B00 = B[0:2, 0:2], B10 = B[2:4, 0:2]
    # P = A10-A00, P[i,j] = A[i+2,j] - A[i,j]
    # Q = B00+B10, Q[i,j] = B[i,j] + B[i+2,j]
    # R = P*Q
    # R contributes to C[2:4, 2:4] with +1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i+2,j] - A[i,j]
                U[(i+2)*4 + j, col] = 1
                U[i*4 + j, col] = -1
                # B part: B[j,k] + B[j+2,k]
                V[j*5 + k, col] = 1
                V[(j+2)*5 + k, col] = 1
                # C part: C[i+2,k+2] += 1
                W[(i+2)*4 + (k+2), col] = 1
                col += 1
    
    # Strassen term 6: (A01-A11)(B10+B00)
    # A01 = A[0:2, 2:4], A11 = A[2:4, 2:4]
    # B10 = B[2:4, 0:2], B00 = B[0:2, 0:2]
    # P = A01-A11, P[i,j] = A[i,j+2] - A[i+2,j+2]
    # Q = B10+B00, Q[i,j] = B[i+2,j] + B[i,j]
    # R = P*Q
    # R contributes to C[0:2, 0:2] with +1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A part: A[i,j+2] - A[i+2,j+2]
                U[i*4 + (j+2), col] = 1
                U[(i+2)*4 + (j+2), col] = -1
                # B part: B[j+2,k] + B[j,k]
                V[(j+2)*5 + k, col] = 1
                V[j*5 + k, col] = 1
                # C part: C[i,k] += 1
                W[i*4 + k, col] = 1
                col += 1
    
    # Now handle col 4 with schoolbook: 4x4x1 = 16 terms
    for i in range(4):
        for j in range(4):
            for k in range(2):  # k is the row of B, 0-1 for top, 2-3 for bot
                pass  # This is wrong, let me redo
    
    # For col 4: C[i,4] = sum_j A[i,j] * B[j,4]
    # 16 terms:
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1  # C[4,i] in the p*n indexing? 
            # Wait, C is 4x5, so C[k,n+i] where k is col, i is row
            # C[4, i] means col 4, row i
            col += 1
    
    # Verify
    assert col <= rank, f"Used {col} columns, expected {rank}"
    
    # Trim if needed
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return U.tolist(), V.tolist(), W.tolist(), col

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        U_list, V_list, W_list, actual_rank = _build_block_circulant_decomposition()
        
        # Verify
        if time.time() < deadline - 5:
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, max_checks=20):
                return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook
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
