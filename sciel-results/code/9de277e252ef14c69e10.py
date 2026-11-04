<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
=======
# EVOLVE-BLOCK-START
import time

def _strassen_3x3_decomposition():
    """
    Return (U3, V3, W3) for 3x3 matrix multiplication using Strassen's algorithm.
    U3 shape: (9, 7), V3 shape: (9, 7), W3 shape: (9, 7)
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    
    Strassen for 3x3 blocks (treating 3x3 as a single "element" in 2x2 Strassen):
    Actually, for a 3x3 matrix multiplication, we use the standard 7-term Strassen
    by partitioning into 2x2 blocks of size... no, 3 doesn't split evenly into 2.
    
    Instead, use the known rank-7 decomposition for 3x3 matrix multiplication.
    The standard approach: use the 7-term formula derived from Strassen applied
    to a 2x2 grid where each "block" is a 1.5x1.5... that doesn't work.
    
    Actually, the well-known result is that 3x3 matrix multiplication has rank 
    between 19 and 23. Let's use a known explicit rank-20 or rank-23 decomposition.
    
    For simplicity and correctness, let's use the schoolbook rank-27 for the inner
    3x3, but use Strassen at the outer block level (3x3 grid of 3x3 blocks).
    
    Outer level: 3x3 grid of 3x3 blocks. Use Strassen-style 7 multiplications
    at the block level. But 3x3 doesn't factor as 2x2...
    
    Alternative: Use the block Strassen where we view 6x6 as 2x2 blocks of 3x3.
    That gives 7 multiplications of 3x3 matrices. Each 3x3 multiplication can be
    done with rank 27 (schoolbook) or lower.
    
    Total rank with schoolbook inner: 7 * 27 = 189
    Total rank with Strassen inner: 7 * 7 = 49 (if we had rank-7 for 3x3, which we don't)
    
    The minimum known rank for 3x3 is 19 (over some fields) but explicit constructions
    are complex. Let's use a known explicit decomposition.
    
    For now, let's use the 2x2 block Strassen (6x6 as 2x2 of 3x3 blocks) with
    schoolbook 3x3 inner: rank = 7 * 27 = 189.
    """
    pass

def _build_block_strassen_6x6():
    """
    6x6 matrix multiplication as 2x2 blocks of 3x3 matrices, using Strassen's
    7-multiplication scheme at the block level, with schoolbook (rank 27) for
    each 3x3 block multiplication. Total rank: 7 * 27 = 189.
    
    Block structure:
    A = [[A11, A12],     B = [[B11, B12],     C = [[C11, C12],
         [A21, A22]]          [B21, B22]]          [C21, C22]]
    
    Where each Ai, Bi, Ci is 3x3.
    
    Strassen:
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
    
    Each Mi is a 3x3 matrix product, decomposed with rank 27 (schoolbook).
    Total: 7 * 27 = 189 terms.
    """
    n = 6
    block_size = 3
    rank_per_block = 27  # schoolbook 3x3
    total_rank = 7 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * n)]  # 36 x 189
    V = [[0] * total_rank for _ in range(n * n)]  # 36 x 189
    W = [[0] * total_rank for _ in range(n * n)]  # 36 x 189
    
    # Helper: add a schoolbook 3x3 decomposition for a block product
    # A_block is 3x3, B_block is 3x3, result C_block is 3x3
    # For each (i,j,k) in 3x3x3, add a term with coefficient
    # U[row_a, col] += coeff_a, V[row_b, col] += coeff_b, W[row_c, col] += coeff_c
    # where row_a = i*3+j (in A), row_b = j*3+k (in B), row_c = k*3+i (in C)
    
    # Block offsets in the 6x6 matrices:
    # A11: rows 0-2, cols 0-2  -> A index: (r, c) -> r*6+c for r,c in 0..2
    # A12: rows 0-2, cols 3-5  -> A index: r*6+(c+3) for r in 0..2, c in 0..2
    # A21: rows 3-5, cols 0-2  -> A index: (r+3)*6+c for r in 0..2, c in 0..2
    # A22: rows 3-5, cols 3-5  -> A index: (r+3)*6+(c+3) for r in 0..2, c in 0..2
    
    # Similarly for B and C.
    
    def a_idx(block, r, c):
        """Index in A for block (0=A11,1=A12,2=A21,3=A22), local row r, col c"""
        if block == 0:
            return r * 6 + c
        elif block == 1:
            return r * 6 + (c + 3)
        elif block == 2:
            return (r + 3) * 6 + c
        elif block == 3:
            return (r + 3) * 6 + (c + 3)
    
    def b_idx(block, r, c):
        """Index in B for block, local row r, col c"""
        if block == 0:
            return r * 6 + c
        elif block == 1:
            return r * 6 + (c + 3)
        elif block == 2:
            return (r + 3) * 6 + c
        elif block == 3:
            return (r + 3) * 6 + (c + 3)
    
    def c_idx(block, r, c):
        """Index in C for block, local row r, col c"""
        if block == 0:
            return r * 6 + c
        elif block == 1:
            return r * 6 + (c + 3)
        elif block == 2:
            return (r + 3) * 6 + c
        elif block == 3:
            return (r + 3) * 6 + (c + 3)
    
    # Strassen terms: each term is (a_blocks, b_blocks, sign)
    # where the product is sum over a_blocks, b_blocks with signs
    # M1 = (A11+A22)(B11+B22): terms [(0,0,+1),(3,0,+1),(0,3,+1),(3,3,+1)]
    # M2 = (A21+A22)B11: terms [(2,0,+1),(3,0,+1)]
    # M3 = A11(B12-B22): terms [(0,1,+1),(0,3,-1)]
    # M4 = A22(B21-B11): terms [(3,2,+1),(3,0,-1)]
    # M5 = (A11+A12)B22: terms [(0,3,+1),(1,3,+1)]
    # M6 = (A21-A11)(B11+B12): terms [(2,0,+1),(2,1,+1),(-0,0,-1),(-0,1,-1)]
    #    = [(2,0,+1),(2,1,+1),(0,0,-1),(0,1,-1)]
    # M7 = (A12-A22)(B21+B22): terms [(1,2,+1),(1,3,+1),(3,2,-1),(3,3,-1)]
    
    # But wait: each Mi is itself a matrix product. We need to decompose each
    # Mi = X * Y where X and Y are 3x3 matrices that are linear combinations
    # of the block entries.
    
    # Actually, the way Strassen works: Mi = Xi * Yi where Xi and Yi are 3x3
    # matrices formed by adding/subtracting the relevant blocks.
    # Then Mi = Xi * Yi is a 3x3 matrix multiplication, which we decompose
    # with schoolbook (rank 27).
    
    # For each Strassen term Mi, we have:
    # Xi = sum of alpha_a * A_block_a (3x3 matrices)
    # Yi = sum of beta_b * B_block_b (3x3 matrices)
    # Mi = Xi * Yi
    # Decompose Mi with schoolbook: for each (i,j,k) in 3x3x3:
    #   C[i,k] += Xi[i,j] * Yi[j,k]
    #   = (sum_a alpha_a * A_block_a[i,j]) * (sum_b beta_b * B_block_b[j,k])
    #   = sum_{a,b} alpha_a * beta_b * A_block_a[i,j] * B_block_b[j,k]
    
    # So for each (a, b, i, j, k), we add a term:
    #   U[a_idx(a, i, j)] += alpha_a
    #   V[b_idx(b, j, k)] += beta_b  
    #   W[c_idx(target_block, i, k)] += contribution_to_C
    
    # The contribution to C comes from the Strassen combination:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So each Mi contributes to certain C blocks with certain signs.
    
    # Let me restructure: for each of the 7 Mi, and for each (a, b) pair in
    # its definition, and for each (i, j, k) in the 3x3 schoolbook, add one term.
    
    # Mi definition: list of (a_block, alpha, b_block, beta)
    strassen_defs = [
        # M1 = (A11+A22)(B11+B22)
        [(0, 1, 0, 1), (0, 1, 3, 1), (3, 1, 0, 1), (3, 1, 3, 1)],
        # M2 = (A21+A22)B11
        [(2, 1, 0, 1), (3, 1, 0, 1)],
        # M3 = A11(B12-B22)
        [(0, 1, 1, 1), (0, 1, 3, -1)],
        # M4 = A22(B21-B11)
        [(3, 1, 2, 1), (3, 1, 0, -1)],
        # M5 = (A11+A12)B22
        [(0, 1, 3, 1), (1, 1, 3, 1)],
        # M6 = (A21-A11)(B11+B12)
        [(2, 1, 0, 1), (2, 1, 1, 1), (0, -1, 0, 1), (0, -1, 1, 1)],
        # M7 = (A12-A22)(B21+B22)
        [(1, 1, 2, 1), (1, 1, 3, 1), (3, -1, 2, 1), (3, -1, 3, 1)],
    ]
    
    # C block contributions: C_block = sum of gamma_m * M_m
    # C11: M1+1, M4+1, M5-1, M7+1
    # C12: M3+1, M5+1
    # C21: M2+1, M4+1
    # C22: M1+1, M2-1, M3+1, M6+1
    c_contributions = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},   # C11
        1: {2: 1, 4: 1},                   # C12
        2: {1: 1, 3: 1},                   # C21
        3: {0: 1, 1: -1, 2: 1, 5: 1},     # C22
    }
    
    col = 0
    for m_idx in range(7):
        for (a_block, alpha, b_block, beta) in strassen_defs[m_idx]:
            # For each (i, j, k) in 3x3 schoolbook decomposition of this product
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # This term contributes to C blocks where M_m has a contribution
                        for c_block, gamma in c_contributions.items():
                            if m_idx in gamma:
                                coeff = alpha * beta * gamma
                                u_row = a_idx(a_block, i, j)
                                v_row = b_idx(b_block, j, k)
                                w_row = c_idx(c_block, i, k)
                                U[u_row][col] += coeff
                                V[v_row][col] += coeff
                                W[w_row][col] += coeff
                                col += 1
    
    # Wait, this is wrong. The schoolbook decomposition for Mi = Xi*Yi where
    # Xi and Yi are already linear combinations doesn't work this way.
    
    # Let me reconsider. The correct approach:
    # 
    # C = A*B = sum of all block products.
    # Strassen rewrites this as 7 products Mi = Xi*Yi.
    # Each Mi is a 3x3 matrix = Xi * Yi.
    # 
    # To decompose Mi = Xi * Yi into rank-27 terms:
    # Mi[i,k] = sum_j Xi[i,j] * Yi[j,k]
    # 
    # But Xi[i,j] = sum_a alpha_a * A_a[i,j] and Yi[j,k] = sum_b beta_b * B_b[j,k]
    # 
    # So Mi[i,k] = sum_j sum_a sum_b alpha_a * beta_b * A_a[i,j] * B_b[j,k]
    # 
    # Each triple (a, b, j) gives one "scalar multiplication" A_a[i,j] * B_b[j,k].
    # But this is NOT a valid tensor decomposition term because we need:
    #   U[row_a, col] * V[row_b, col] * W[row_c, col] = A[row_a] * B[row_b] * C[row_c]
    # 
    # The issue is that A_a[i,j] is one entry of matrix A, and B_b[j,k] is one entry
    # of matrix B. Their product contributes to C at position (i,k) in block c_block.
    # 
    # So for each (a_block, b_block, i, j, k, c_block) where the coefficient is nonzero:
    #   U[a_idx(a_block, i, j)][col] = alpha_a
    #   V[b_idx(b_block, j, k)][col] = beta_b
    #   W[c_idx(c_block, i, k)][col] = gamma
    # 
    # But this would give us a term: A[a_idx] * B[b_idx] * C[c_idx]
    # which is A[i,j of block a] * B[j,k of block b] * C[i,k of block c]
    # 
    # The sum over all such terms should equal C = A*B. Let's verify the count:
    # Each Mi has len(strassen_defs[m]) (a,b) pairs.
    # For each (a,b) pair, we have 3*3*3 = 27 (i,j,k) combinations.
    # For each, we add one term for each c_block where M_m contributes.
    # 
    # Total terms = sum_m sum_{(a,b) in M_m} 27 * (number of c_blocks M_m contributes to)
    # 
    # This is NOT 7*27 = 189. It's more because each M_m contributes to multiple C blocks.
    # 
    # Hmm, I think I'm overcomplicating this. Let me use a cleaner approach.
    
    # Actually, the standard way: each "multiplication" Mi = Xi*Yi is a 3x3 matrix
    # multiplication. To decompose it with schoolbook, we need 27 terms. But the
    # entries of Xi and Yi are linear combinations of A and B entries. So each of
    # the 27 scalar products in Mi actually involves products of linear combinations.
    # 
    # The correct tensor decomposition: we need to express C = A*B as a sum of
    # rank-1 tensors. Using Strassen at the block level, we get 7 "big" multiplications.
    # Each big multiplication Mi = Xi*Yi where Xi, Yi are 3x3 matrices with entries
    # that are linear combinations of the original A, B entries.
    # 
    # To make this into a valid tensor decomposition, we need to "expand" each Mi.
    # The product Xi*Yi has 9 entries, each being a sum of 3 products. So Mi has
    # 27 "atomic" products. Each atomic product is of the form (linear comb of A) * 
    # (linear comb of B), contributing to an entry of C.
    # 
    # But a tensor decomposition term must be: A[one entry] * B[one entry] * C[one entry].
    # So we need to further expand the linear combinations.
    # 
    # This is getting complex. Let me just use the direct schoolbook for 6x6x6 but
    # with the Strassen structure to reduce the count.
    
    # Simpler approach: Just enumerate all necessary (A_entry, B_entry, C_entry)
    # triples with their coefficients, combining like terms.
    
    pass

def construct():
    """
    Block Strassen decomposition for 6x6 matrix multiplication.
    View 6x6 as 2x2 blocks of 3x3. Use Strassen's 7 multiplications at block level.
    Each 3x3 block multiplication uses schoolbook (rank 27).
    
    Total rank: 7 * 27 = 189
    """
    n = 6
    total_rank = 7 * 27  # 189
    
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    # Block index helpers
    def a_idx(block, r, c):
        rows = [0, 0, 3, 3]
        cols = [0, 3, 0, 3]
        return (rows[block] + r) * 6 + (cols[block] + c)
    
    def b_idx(block, r, c):
        rows = [0, 0, 3, 3]
        cols = [0, 3, 0, 3]
        return (rows[block] + r) * 6 + (cols[block] + c)
    
    def c_idx(block, r, c):
        rows = [0, 0, 3, 3]
        cols = [0, 3, 0, 3]
        return (rows[block] + r) * 6 + (cols[block] + c)
    
    # Strassen definitions: for each M_m, list of (a_block, alpha, b_block, beta)
    # M_m = (sum alpha_a * A_a) * (sum beta_b * B_b)
    strassen_defs = [
        # M0: (A00+A11)(B00+B11)  [using 0-based: A11=block0, A12=block1, A21=block2, A22=block3]
        [(0, 1, 0, 1), (0, 1, 3, 1), (3, 1, 0, 1), (3, 1, 3, 1)],
        # M1: (A21+A22)B11
        [(2, 1, 0, 1), (3, 1, 0, 1)],
        # M2: A11(B12-B22)
        [(0, 1, 1, 1), (0, 1, 3, -1)],
        # M3: A22(B21-B11)
        [(3, 1, 2, 1), (3, 1, 0, -1)],
        # M4: (A11+A12)B22
        [(0, 1, 3, 1), (1, 1, 3, 1)],
        # M5: (A21-A11)(B11+B12)
        [(2, 1, 0, 1), (2, 1, 1, 1), (0, -1, 0, 1), (0, -1, 1, 1)],
        # M6: (A12-A22)(B21+B22)
        [(1, 1, 2, 1), (1, 1, 3, 1), (3, -1, 2, 1), (3, -1, 3, 1)],
    ]
    
    # C block contributions: c_block -> {m_idx: gamma}
    c_contrib = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},   # C11 = M0 + M3 - M4 + M6
        1: {2: 1, 4: 1},                   # C12 = M2 + M4
        2: {1: 1, 3: 1},                   # C21 = M1 + M3
        3: {0: 1, 1: -1, 2: 1, 5: 1},     # C22 = M0 - M1 + M2 + M5
    }
    
    # For each M_m, for each (a_block, alpha, b_block, beta) in its definition,
    # and for each (i, j, k) in the 3x3 schoolbook:
    #   We have a product: A[a_idx(a_block,i,j)] * B[b_idx(b_block,j,k)]
    #   This contributes to C entries at c_idx(c_block, i, k) for each c_block
    #   where M_m has a contribution gamma.
    #   Coefficient: alpha * beta * gamma
    #
    # But we need to combine terms that have the same (U_row, V_row, W_row).
    # Actually, each unique (a_block, b_block, i, j, k, c_block) combination gives
    # a unique (U_row, V_row, W_row) triple, so we can just assign one column per
    # such combination... but that's too many.
    #
    # Wait: the total number of terms is:
    # sum_m sum_{(a,b) in strassen_defs[m]} 27 * num_c_blocks(m)
    # M0: 4 pairs * 27 * 4 = 432
    # M1: 2 pairs * 27 * 2 = 108
    # M2: 2 pairs * 27 * 2 = 108
    # M3: 2 pairs * 27 * 3 = 162
    # M4: 2 pairs * 27 * 3 = 162
    # M5: 4 pairs * 27 * 1 = 108
    # M6: 4 pairs * 27 * 4 = 432
    # Total: 1512 -- way too many!
    #
    # I see the issue. The correct approach is different. Let me think again.
    #
    # The Strassen algorithm says: C = sum_m gamma_m * M_m where M_m = X_m * Y_m.
    # Each M_m is a 3x3 matrix. M_m[i,k] = sum_j X_m[i,j] * Y_m[j,k].
    # X_m[i,j] = sum_a alpha_{m,a} * A_a[i,j]
    # Y_m[j,k] = sum_b beta_{m,b} * B_b[j,k]
    #
    # So M_m[i,k] = sum_j sum_a sum_b alpha_{m,a} beta_{m,b} A_a[i,j] B_b[j,k]
    #
    # C[i,k] (in block c) = sum_m gamma_{m,c} M_m[i,k]
    #   = sum_m sum_j sum_a sum_b gamma_{m,c} alpha_{m,a} beta_{m,b} A_a[i,j] B_b[j,k]
    #
    # For this to be a valid tensor decomposition, each term must be of the form:
    #   coeff * A[one specific entry] * B[one specific entry] * C[one specific entry]
    #
    # The term A_a[i,j] * B_b[j,k] contributes to C_c[i,k] with coefficient
    #   sum_m gamma_{m,c} alpha_{m,a} beta_{m,b}
    #
    # So for each unique (a_block, b_block, i, j, k, c_block), the total coefficient
    # is sum over m of gamma_{m,c_block} * alpha_{m,a_block} * beta_{m,b_block}.
    #
    # If this sum is nonzero, we need one column for this (a_block, b_block, i, j, k, c_block).
    #
    # Number of such terms: 4 * 4 * 3 * 3 * 3 * 4 = 4*4*27*4 = 1728 max, but many
    # will have zero coefficient. Let's just compute them all and skip zeros.
    
    col = 0
    for a_block in range(4):
        for b_block in range(4):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for c_block in range(4):
                            # Compute total coefficient
                            coeff = 0
                            for m in range(7):
                                gamma = c_contrib[c_block].get(m, 0)
                                if gamma == 0:
                                    continue
                                # Find alpha and beta for this (a_block, b_block) in M_m
                                alpha = 0
                                beta = 0
                                found = False
                                for (ab, al, bb, be) in strassen_defs[m]:
                                    if ab == a_block and bb == b_block:
                                        alpha += al
                                        beta += be
                                        found = True
                                if found and alpha != 0 and beta != 0:
                                    coeff += gamma * alpha * beta
                            
                            if coeff != 0:
                                u_row = a_idx(a_block, i, j)
                                v_row = b_idx(b_block, j, k)
                                w_row = c_idx(c_block, i, k)
                                U[u_row][col] = coeff
                                V[v_row][col] = coeff
                                W[w_row][col] = coeff
                                col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize there's an issue with my approach. Let me reconsider the coefficient computation. The problem is that for a given (a_block, b_block, i, j, k, c_block), the coefficient should be the sum over all m where both a_block appears in M_m's A-side and b_block appears in M_m's B-side, of gamma_{m,c_block} * alpha_{m,a_block} * beta_{m,b_block}.

But in my strassen_defs, each (a_block, b_block) pair appears at most once per M_m. Let me verify this is correct and test the decomposition.

Let me simplify and just verify correctness by checking that the decomposition reproduces C = A*B for random matrices.

