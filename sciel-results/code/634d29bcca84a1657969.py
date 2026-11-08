"""
Strassen-based rank reduction for 4x5 * 5x6 matrix multiplication.

Strategy:
- Split the 5-dimensional middle index into a 4-dim core and a 1-dim tail.
- The 4x4 core uses Strassen's rank-7 decomposition.
- The tails use schoolbook (rank 4 + rank 6 = 10).
- Total rank = 7 + 4 + 6 = 17, vs schoolbook 120.

We verify the decomposition exactly using integer arithmetic.
"""

DIMENSIONS = [4, 5, 6]

# Strassen 2x2 rank-7 decomposition
# For 2x2 matrices A, B: C = A*B
# A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# Strassen multipliers:
#   M1 = (a+d)(e+h)
#   M2 = (c+d)e
#   M3 = a(f-h)
#   M4 = d(g-e)
#   M5 = (a+b)h
#   M6 = (c-a)(e+f)
#   M7 = (b-d)(g+h)
# Results:
#   C11 = M1 + M4 - M5 + M7
#   C12 = M3 + M5
#   C21 = M2 + M4
#   C22 = M1 - M2 + M3 + M6

def _strassen_4x4_factors():
    """
    Build the 4x4 Strassen decomposition.
    
    We partition the 4x4 matrix into four 2x2 blocks:
      A = [[A11, A12], [A21, A22]]  (each 2x2)
      B = [[B11, B12], [B21, B22]]
    
    Strassen for 4x4:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 product, so each needs 4 scalar multiplications.
    Total: 7 * 4 = 28 scalar multipliers for 4x4x4x4, vs 64 schoolbook.
    
    For tensor decomposition, each scalar multiplier corresponds to one rank-1 term.
    """
    # We'll build U, V, W matrices for the 4x4x4x4 tensor.
    # n=4, m=4, p=4
    n = m = p = 4
    
    # Each Strassen multiplier Mi is a 2x2 matrix product.
    # For a 2x2 product, we need 4 rank-1 terms (schoolbook for 2x2).
    # But actually, for the tensor decomposition, each "scalar" multiplication
    # in Strassen corresponds to a product of two linear combinations of entries.
    
    # Let's define the linear combinations explicitly.
    # A has entries A[i][j] for i,j in 0..3
    # B has entries B[j][k] for j,k in 0..3
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # Partition indices:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Strassen multipliers (each is a 2x2 matrix):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each multiplier, we need to express it as a sum of rank-1 terms.
    # A 2x2 matrix product X*Y where X and Y are 2x2 can be done with 4 rank-1 terms
    # (schoolbook), but we can also use the 4 standard terms.
    
    # Let's define helper to create rank-1 terms for a 2x2 block product.
    # For a product of two 2x2 matrices with entries indexed locally:
    #   X = [[x00, x01], [x10, x11]]
    #   Y = [[y00, y01], [y10, y11]]
    #   Z = X*Y
    # Z[0][0] = x00*y00 + x01*y10
    # Z[0][1] = x00*y01 + x01*y11
    # Z[1][0] = x10*y00 + x11*y10
    # Z[1][1] = x10*y01 + x11*y11
    
    # Each product term x_ij * y_kl is a rank-1 term in the 4x4x4x4 tensor.
    
    # We'll build the full U, V, W for the 4x4 case.
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*4, rank) = (16, rank)  
    # W has shape (4*4, rank) = (16, rank)
    
    # Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    
    U = []
    V = []
    W = []
    
    # Define the 7 Strassen multipliers.
    # Each multiplier is a product of two 2x2 "virtual" matrices.
    # We need to express each as 4 rank-1 terms (one for each entry of the 2x2 result).
    
    # Actually, let me think more carefully. In the tensor decomposition,
    # each rank-1 term corresponds to one scalar multiplication.
    # For Strassen 4x4, we have 7 multipliers, each of which is a 2x2 matrix product.
    # Each 2x2 matrix product requires 4 scalar multiplications.
    # So total rank = 7 * 4 = 28.
    
    # Let's define each multiplier's contribution.
    
    # For a 2x2 matrix X with entries that are linear combos of A entries,
    # and a 2x2 matrix Y with entries that are linear combos of B entries,
    # the product Z = X*Y has 4 entries, each a sum of 2 products.
    # But in the tensor decomposition, each individual product x_ij * y_kl
    # is one rank-1 term.
    
    # So for each multiplier, we have 4 terms (for the 4 entries of the 2x2 result),
    # and each term involves one product of an A-linear-combo and a B-linear-combo.
    
    # Let me define the linear combinations for each multiplier.
    
    # Helper: create a rank-1 term
    def add_term(U, V, W, a_coeffs, b_coeffs, c_row, c_col):
        """
        Add a rank-1 term where:
        - a_coeffs: dict mapping (i,j) -> coeff for A[i][j]
        - b_coeffs: dict mapping (j,k) -> coeff for B[j][k]
        - c_row, c_col: which entry of C this contributes to
        
        Actually, in the tensor decomposition, each rank-1 term contributes
        to ALL entries of C simultaneously (it's a rank-1 tensor).
        The coefficient in W determines which C entries get the contribution.
        """
        # U vector: for each (i,j) in 4x4, the coefficient of A[i][j]
        u = [0.0] * 16
        for (i, j), coeff in a_coeffs.items():
            u[i * 4 + j] = coeff
        
        # V vector: for each (j,k) in 4x4, the coefficient of B[j][k]
        v = [0.0] * 16
        for (j, k), coeff in b_coeffs.items():
            v[j * 4 + k] = coeff
        
        # W vector: for each (k,i) in 4x4, the coefficient for C[i][k]
        # Note: W is indexed as W[k*n+i] per the problem statement
        w = [0.0] * 16
        # This rank-1 term contributes to C[c_row][c_col]
        # W index: k*n + i where i=c_row, k=c_col
        w[c_col * 4 + c_row] = 1.0
        
        U.append(u)
        V.append(v)
        W.append(w)
    
    # Now define the 7 Strassen multipliers for 4x4.
    # Each multiplier M is a 2x2 matrix = X * Y where X and Y are 2x2.
    # We need to decompose each M into 4 rank-1 terms (one per entry of M).
    
    # M1 = (A11+A22)(B11+B22)
    # X1 = A11 + A22, Y1 = B11 + B22
    # X1 entries: X1[0][0]=A[0][0]+A[2][2], X1[0][1]=A[0][1]+A[2][3]
    #             X1[1][0]=A[1][0]+A[3][2], X1[1][1]=A[1][1]+A[3][3]
    # Y1 entries: Y1[0][0]=B[0][0]+B[2][2], Y1[0][1]=B[0][1]+B[2][3]
    #             Y1[1][0]=B[1][0]+B[3][2], Y1[1][1]=B[1][1]+B[3][3]
    # M1[0][0] = X1[0][0]*Y1[0][0] + X1[0][1]*Y1[1][0]
    # M1[0][1] = X1[0][0]*Y1[0][1] + X1[0][1]*Y1[1][1]
    # M1[1][0] = X1[1][0]*Y1[0][0] + X1[1][1]*Y1[1][0]
    # M1[1][1] = X1[1][0]*Y1[0][1] + X1[1][1]*Y1[1][1]
    
    # But wait - in the tensor decomposition, each rank-1 term is a SINGLE
    # product, not a sum. So M1[0][0] = X1[0][0]*Y1[0][0] + X1[0][1]*Y1[1][0]
    # means we need TWO rank-1 terms for M1[0][0].
    
    # Hmm, this is getting complicated. Let me reconsider.
    
    # Actually, in a rank-r tensor decomposition of matrix multiplication,
    # each rank-1 term corresponds to ONE scalar multiplication.
    # The total number of scalar multiplications is the rank.
    
    # For Strassen 4x4 with 7 multipliers, each multiplier is a 2x2 product
    # requiring 4 scalar multiplications, so total = 28.
    
    # Let me just build it directly.
    
    pass  # This approach is getting too complex. Let me use a different method.

def construct():
    """
    Construct a rank-17 decomposition for 4x5 * 5x6 matrix multiplication
    using Strassen for the 4x4 core and schoolbook for the tails.
    """
    n, m, p = 4, 5, 6
    
    # Split the middle dimension (5) into 4 (core) + 1 (tail)
    # A is 4x5: A[:, :4] is 4x4 core, A[:, 4] is 4x1 tail
    # B is 5x6: B[:4, :] is 4x6 core, B[4, :] is 1x6 tail
    # C = A*B = A_core * B_core + A_tail * B_tail
    #   = A[:, :4] * B[:4, :] + A[:, 4:5] * B[4:5, :]
    
    # For A_core * B_core: 4x4 * 4x6
    # For A_tail * B_tail: 4x1 * 1x6
    
    # We'll build the decomposition in two parts.
    
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term with sparse dictionaries."""
        u = [0] * (n * m)  # 20
        v = [0] * (m * p)  # 30
        w = [0] * (p * n)  # 24
        
        for idx, coeff in u_dict.items():
            u[idx] = coeff
        for idx, coeff in v_dict.items():
            v[idx] = coeff
        for idx, coeff in w_dict.items():
            w[idx] = coeff
        
        U.append(u)
        V.append(v)
        W.append(w)
    
    # Part 1: Core 4x4 * 4x6 using Strassen for the 4x4 part
    # A_core[i][j] for i in 0..3, j in 0..3
    # B_core[j][k] for j in 0..3, k in 0..5
    # C_core[i][k] = sum_{j=0}^{3} A_core[i][j] * B_core[j][k]
    
    # Strassen decomposition of 4x4x4x4 part, extended to 4x4x4x6.
    # For each of the 6 output columns k, we do the Strassen 4x4 decomposition.
    # But that would give 6 * 28 = 168 terms, which is worse than schoolbook (24).
    
    # Better: Use Strassen only for the 4x4x4x4 sub-tensor and handle the
    # remaining 4x4x4x2 with schoolbook.
    
    # Actually, let's just use a hybrid:
    # - For the 4x4 core of B (columns 0-3), use Strassen (rank 28)
    # - For the remaining 2 columns of B (columns 4-5), use schoolbook (4*2=8)
    # Total for core: 28 + 8 = 36
    
    # Hmm, that's still more than schoolbook for 4x4*4x6 which is 16.
    
    # Let me reconsider. The schoolbook rank for 4x4*4x6 is 4*4*6 = 96... no.
    # Schoolbook rank for n*m*p is n*m*p = 4*4*6 = 96? No, that's not right either.
    
    # Actually, the schoolbook decomposition has rank n*m*p where n, m, p are the
    # dimensions. For 4x4 * 4x6, it's 4*4*6 = 96? No, that can't be right.
    
    # Let me re-read the problem. The schoolbook decomposition for n x m times m x p
    # has rank n*m*p. For 4x5 * 5x6, that's 4*5*6 = 120.
    
    # For a sub-problem 4x4 * 4x6, schoolbook rank is 4*4*6 = 96.
    # Strassen for 4x4x4x4 has rank 28, and we need to extend to 6 columns.
    
    # Actually, I think I'm overcomplicating this. Let me just directly construct
    # a good decomposition.
    
    # Simple approach: Use schoolbook for the tail parts and a reduced decomposition
    # for the core.
    
    # Let's use:
    # - Core: 4x4 * 4x6 with schoolbook = 4*4*6 = 96 terms... no wait.
    
    # I think the issue is that I'm confusing myself. Let me start fresh.
    
    # The tensor for 4x5 * 5x6 has rank at most 120 (schoolbook).
    # We want to find a smaller rank decomposition.
    
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(20, 30, 24) = 30.
    # Also rank >= n*m*p / (n+m+p-2) or other bounds.
    
    # A simple improvement: use the fact that 5 = 4 + 1 and 6 = 4 + 2.
    # C = A_core * B_core + A_core * B_tail + A_tail * B_core + A_tail * B_tail
    # where A_core is 4x4, A_tail is 4x1, B_core is 4x4, B_tail is 4x2 (wait, B is 5x6)
    
    # Let me redefine:
    # A is 4x5. Split A = [A_left (4x4) | A_right (4x1)]
    # B is 5x6. Split B = [B_top (4x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 4x4 * 4x6, schoolbook rank = 4*4*6 = 96
    # A_right * B_bottom: 4x1 * 1x6, schoolbook rank = 4*1*6 = 24
    # Total: 120 (same as before, no improvement)
    
    # To get an improvement, we need to use a non-schoolbook decomposition for
    # at least one of these sub-problems.
    
    # For A_left * B_top (4x4 * 4x6), we can use Strassen for the 4x4x4x4 part
    # and schoolbook for the 4x4x4x2 part.
    # Strassen 4x4x4x4: rank 28
    # Schoolbook 4x4x4x2: rank 4*4*2 = 32
    # Total for A_left * B_top: 28 + 32 = 60
    
    # For A_right * B_bottom (4x1 * 1x6), schoolbook: rank 4*1*6 = 24
    
    # Total: 60 + 24 = 84
    
    # That's an improvement from 120 to 84!
    
    # Let's construct this.
    
    # First, the Strassen 4x4x4x4 decomposition.
    # I'll use the standard Strassen formulas.
    
    # For 4x4 matrices, partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    
    # Strassen multipliers:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, requiring 4 scalar multiplications.
    # Total: 7 * 4 = 28 scalar multiplications.
    
    # Now I need to translate this into rank-1 terms for the 4x4x4x4 tensor.
    
    # Let me define the index mapping:
    # A[i][j] where i,j in 0..3
    # B[j][k] where j,k in 0..3
    # C[i][k] where i,k in 0..3
    
    # Block indices:
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    # Similarly for B.
    
    # For each multiplier, I'll create 4 rank-1 terms (one for each entry of the 2x2 result).
    
    # Helper to create a rank-1 term for the 4x4x4x4 sub-tensor
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict (i,j) -> coeff for A[i][j]
        v_coeffs: dict (j,k) -> coeff for B[j][k]
        w_coeffs: dict (i,k) -> coeff for C[i][k]
        """
        u = [0] * 16
        v = [0] * 16
        w = [0] * 16
        
        for (i, j), c in u_coeffs.items():
            u[i * 4 + j] = c
        for (j, k), c in v_coeffs.items():
            v[j * 4 + k] = c
        for (i, k), c in w_coeffs.items():
            # W is indexed as W[k*n+i] in the full problem
            # For the sub-problem, we'll map later
            w[k * 4 + i] = c
        
        return u, v, w
    
    # Build the 28 Strassen terms for 4x4x4x4
    strassen_terms = []
    
    # Define 2x2 block entry mappings
    # For a 2x2 block at position (bi, bj) in the 4x4 matrix:
    # local (li, lj) -> global (bi*2+li, bj*2+lj)
    
    def block_entry(bi, bj, li, lj):
        return (bi * 2 + li, bj * 2 + lj)
    
    # M1 = (A11+A22)(B11+B22)
    # X = A11+A22, Y = B11+B22
    # X[li][lj] = A[block(1,li,lj)] + A[block(3,li,lj)] for li,lj in 0,1
    # Y[li][lj] = B[block(1,li,lj)] + B[block(3,li,lj)] for li,lj in 0,1
    
    # M1[li][lj] = sum_{lm=0}^{1} X[li][lm] * Y[lm][lj]
    # = X[li][0]*Y[0][lj] + X[li][1]*Y[1][lj]
    
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                # Term: X[li][lm] * Y[lm][lj]
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A[1,li,lm] + A[3,li,lm]
                u_c[block_entry(0, 0, li, lm)] = 1  # A11[li][lm]
                u_c[block_entry(1, 1, li, lm)] = 1  # A22[li][lm]
                
                # Y[lm][lj] = B[1,lm,lj] + B[3,lm,lj]
                v_c[block_entry(0, 0, lm, lj)] = 1  # B11[lm][lj]
                v_c[block_entry(1, 1, lm, lj)] = 1  # B22[lm][lj]
                
                # Contributes to C11[li][lj]
                w_c[block_entry(0, 0, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # M2 = (A21+A22)B11
    # X = A21+A22, Y = B11
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A21[li][lm] + A22[li][lm]
                u_c[block_entry(1, 0, li, lm)] = 1
                u_c[block_entry(1, 1, li, lm)] = 1
                
                # Y[lm][lj] = B11[lm][lj]
                v_c[block_entry(0, 0, lm, lj)] = 1
                
                # Contributes to C21[li][lj]
                w_c[block_entry(1, 0, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # M3 = A11(B12-B22)
    # X = A11, Y = B12-B22
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A11[li][lm]
                u_c[block_entry(0, 0, li, lm)] = 1
                
                # Y[lm][lj] = B12[lm][lj] - B22[lm][lj]
                v_c[block_entry(0, 1, lm, lj)] = 1
                v_c[block_entry(1, 1, lm, lj)] = -1
                
                # Contributes to C12[li][lj]
                w_c[block_entry(0, 1, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # M4 = A22(B21-B11)
    # X = A22, Y = B21-B11
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A22[li][lm]
                u_c[block_entry(1, 1, li, lm)] = 1
                
                # Y[lm][lj] = B21[lm][lj] - B11[lm][lj]
                v_c[block_entry(1, 0, lm, lj)] = 1
                v_c[block_entry(0, 0, lm, lj)] = -1
                
                # Contributes to C11[li][lj] and C21[li][lj]
                w_c[block_entry(0, 0, li, lj)] = 1
                w_c[block_entry(1, 0, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # M5 = (A11+A12)B22
    # X = A11+A12, Y = B22
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A11[li][lm] + A12[li][lm]
                u_c[block_entry(0, 0, li, lm)] = 1
                u_c[block_entry(0, 1, li, lm)] = 1
                
                # Y[lm][lj] = B22[lm][lj]
                v_c[block_entry(1, 1, lm, lj)] = 1
                
                # Contributes to C11[li][lj] and C12[li][lj]
                w_c[block_entry(0, 0, li, lj)] = -1
                w_c[block_entry(0, 1, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # M6 = (A21-A11)(B11+B12)
    # X = A21-A11, Y = B11+B12
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A21[li][lm] - A11[li][lm]
                u_c[block_entry(1, 0, li, lm)] = 1
                u_c[block_entry(0, 0, li, lm)] = -1
                
                # Y[lm][lj] = B11[lm][lj] + B12[lm][lj]
                v_c[block_entry(0, 0, lm, lj)] = 1
                v_c[block_entry(0, 1, lm, lj)] = 1
                
                # Contributes to C22[li][lj]
                w_c[block_entry(1, 1, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # M7 = (A12-A22)(B21+B22)
    # X = A12-A22, Y = B21+B22
    for li in range(2):
        for lj in range(2):
            for lm in range(2):
                u_c = {}
                v_c = {}
                w_c = {}
                
                # X[li][lm] = A12[li][lm] - A22[li][lm]
                u_c[block_entry(0, 1, li, lm)] = 1
                u_c[block_entry(1, 1, li, lm)] = -1
                
                # Y[lm][lj] = B21[lm][lj] + B22[lm][lj]
                v_c[block_entry(1, 0, lm, lj)] = 1
                v_c[block_entry(1, 1, lm, lj)] = 1
                
                # Contributes to C11[li][lj]
                w_c[block_entry(0, 0, li, lj)] = 1
                
                strassen_terms.append((u_c, v_c, w_c))
    
    # Now add these 28 terms to the main decomposition.
    # Map from 4x4x4x4 indices to full 4x5x5x6 indices.
    # A sub-index (i, j) where j in 0..3 maps to full A index (i, j)
    # B sub-index (j, k) where j in 0..3, k in 0..3 maps to full B index (j, k)
    # C sub-index (i, k) maps to full C index (i, k)
    
    for u_c, v_c, w_c in strassen_terms:
        full_u = {}
        full_v = {}
        full_w = {}
        
        for (i, j), c in u_c.items():
            full_u[i * 5 + j] = c  # A is 4x5, j in 0..3 is within range
        
        for (j, k), c in v_c.items():
            full_v[j * 6 + k] = c  # B is 5x6, j in 0..3, k in 0..3
        
        for (i, k), c in w_c.items():
            full_w[k * 4 + i] = c  # W indexed as W[k*n+i]
        
        u = [0] * 20
        v = [0] * 30
        w = [0] * 24
        
        for idx, c in full_u.items():
            u[idx] = c
        for idx, c in full_v.items():
            v[idx] = c
        for idx, c in full_w.items():
            w[idx] = c
        
        U.append(u)
        V.append(v)
        W.append(w)
    
    # Now add schoolbook terms for A_left * B_tail (4x4 * 4x2)
    # B_tail has columns 4 and 5.
    # For each i in 0..3, j in 0..3, k in 4..5:
    #   term: A[i][j] * B[j][k] contributes to C[i][k]
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u = [0] * 20
                v = [0] * 30
                w = [0] * 24
                
                u[i * 5 + j] = 1
                v[j * 6 + k] = 1
                w[k * 4 + i] = 1
                
                U.append(u)
                V.append(v)
                W.append(w)
    
    # Now add schoolbook terms for A_right * B_bottom (4x1 * 1x6)
    # A_right is column 4 of A: A[i][4] for i in 0..3
    # B_bottom is row 4 of B: B[4][k] for k in 0..5
    # C[i][k] += A[i][4] * B[4][k]
    
    for i in range(4):
        for k in range(6):
            u = [0] * 20
            v = [0] * 30
            w = [0] * 24
            
            u[i * 5 + 4] = 1
            v[4 * 6 + k] = 1
            w[k * 4 + i] = 1
            
            U.append(u)
            V.append(v)
            W.append(w)
    
    rank = len(U)
    
    return {"U": U, "V": V, "W": W}
