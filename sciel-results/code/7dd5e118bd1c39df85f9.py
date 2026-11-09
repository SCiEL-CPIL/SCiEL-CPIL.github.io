"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random

def _strassen_2x2x2(U, V, W, offset, n, m, p):
    """
    Add Strassen rank-7 terms for a 2x2x2 block.
    The block covers rows of A (size n x m), cols of B (size m x p), and output (size p x n).
    We need to map indices correctly.
    
    For a 2x2x2 block at position (i0, j0, k0) in the larger problem:
    - A block: rows i0..i0+1, cols j0..j0+1 (size 2x2)
    - B block: rows j0..j0+1, cols k0..k0+1 (size 2x2)
    - C block: rows k0..k0+1, cols i0..i0+1 (size 2x2)
    
    Strassen for 2x2 matrices:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B11-B21)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # The 7 Strassen multiplications
    # Each multiplication corresponds to a term in the tensor decomposition
    # We need to add 7 terms to U, V, W
    
    # For each Strassen term, we add one column to U, V, W
    # The term structure:
    # M1: (A11+A22) * (B11+B22) -> contributes to C11 and C22
    # M2: (A21+A22) * B11 -> contributes to C21 and C22
    # M3: A11 * (B21-B22) -> contributes to C12 and C22
    # M4: A22 * (B11-B21) -> contributes to C11 and C21
    # M5: (A11+A12) * B22 -> contributes to C11 and C12
    # M6: (A21-A11) * (B11+B12) -> contributes to C22
    # M7: (A11+A12) * (B21+B22) -> contributes to C11
    
    # Let's define the 7 terms with their coefficients in U, V, W
    # Each term has:
    #   U coefficients: which A entries (with signs)
    #   V coefficients: which B entries (with signs)
    #   W coefficients: which C entries (with signs)
    
    # A entries: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) in local 2x2
    # B entries: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1) in local 2x2
    # C entries: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1) in local 2x2
    
    # Term definitions: (U_coefs, V_coefs, W_coefs)
    # U_coefs: list of (local_row, local_col, coeff)
    # V_coefs: list of (local_row, local_col, coeff)
    # W_coefs: list of (local_row, local_col, coeff)
    
    terms = [
        # M1 = (A11+A22)(B11+B22)
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),
        # M2 = (A21+A22)B11
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1),(1,1,-1))]),
        # M3 = A11(B21-B22)
        ([((0,0,1))], [((1,0,1),(1,1,-1))], [((0,1,1),(1,1,1))]),
        # M4 = A22(B11-B21)
        ([((1,1,1))], [((0,0,1),(1,0,-1))], [((0,0,1),(1,0,1))]),
        # M5 = (A11+A12)B22
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,0,-1),(0,1,1))]),
        # M6 = (A21-A11)(B11+B12)
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((1,1,1))]),
        # M7 = (A12-A22)(B21+B22)
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,1))]),
    ]
    
    for u_coefs, v_coefs, w_coefs in terms:
        col = offset
        # Add U coefficients
        for (lr, lc, coeff) in u_coefs:
            global_row = (i0 + lr) * m + (j0 + lc)
            U[global_row][col] += coeff
        # Add V coefficients
        for (lr, lc, coeff) in v_coefs:
            global_row = (j0 + lr) * p + (k0 + lc)
            V[global_row][col] += coeff
        # Add W coefficients
        for (lr, lc, coeff) in w_coefs:
            global_row = (k0 + lr) * n + (i0 + lc)
            W[global_row][col] += coeff
        offset += 1
    
    return offset

def _schoolbook_2x1x1(U, V, W, offset, i0, j0, k0, n, m, p):
    """
    Add schoolbook rank-2 terms for a 2x1x1 block.
    A block: rows i0..i0+1, col j0 (size 2x1)
    B block: row j0, col k0 (size 1x1)
    C block: row k0, cols i0..i0+1 (size 1x2)
    
    Two terms:
    Term 1: A[i0,j0] * B[j0,k0] -> C[k0,i0]
    Term 2: A[i0+1,j0] * B[j0,k0] -> C[k0,i0+1]
    """
    # Term 1
    col = offset
    U[i0 * m + j0][col] = 1
    V[j0 * p + k0][col] = 1
    W[k0 * n + i0][col] = 1
    offset += 1
    
    # Term 2
    col = offset
    U[(i0+1) * m + j0][col] = 1
    V[j0 * p + k0][col] = 1
    W[k0 * n + (i0+1)][col] = 1
    offset += 1
    
    return offset

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Partition B (5x5) into blocks:
    # Block 1: rows 0-1, cols 0-1 (2x2)
    # Block 2: rows 2-3, cols 2-3 (2x2)
    # Block 3: row 4, col 4 (1x1)
    # 
    # Partition A (2x5) correspondingly:
    # Block 1: rows 0-1, cols 0-1 (2x2)
    # Block 2: rows 0-1, cols 2-3 (2x2)
    # Block 3: rows 0-1, col 4 (2x1)
    #
    # Output C (5x2) blocks:
    # Block 1: rows 0-1, cols 0-1 (2x2)
    # Block 2: rows 2-3, cols 0-1 (2x2)
    # Block 3: row 4, cols 0-1 (1x2)
    
    # Total rank: 4 blocks of Strassen (7 each) + 1 block of schoolbook (2) = 4*7 + 2 = 30
    # Wait, let me reconsider. The blocks in B don't need to be contiguous in the same way.
    # Actually, for matrix multiplication C = A * B where A is 2x5 and B is 5x5:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can partition the j-index (0..4) into groups. For each group of j's, we compute
    # a partial product. If we group j's into {0,1}, {2,3}, {4}, then:
    # - Group {0,1}: A[:,0:2] (2x2) * B[0:2,:] (2x5) -> contributes to all of C
    # - Group {2,3}: A[:,2:4] (2x2) * B[2:4,:] (2x5) -> contributes to all of C
    # - Group {4}: A[:,4:5] (2x1) * B[4:5,:] (1x5) -> contributes to all of C
    # 
    # But this doesn't decompose nicely into independent sub-problems because each group
    # contributes to the entire C.
    
    # Let me reconsider. The correct approach is to partition both the row and column indices
    # of B, and the corresponding column indices of A.
    
    # Actually, for the tensor decomposition to work, we need to partition the middle index j
    # into disjoint sets, and for each set, the contribution is A[:,j_set] * B[j_set,:].
    # This gives a block-diagonal structure in the j-index.
    
    # So if we partition j = {0,1}, {2,3}, {4}:
    # - Block 1: A[:,0:2] (2x2) * B[0:2,:] (2x5) -> 2x5 matrix
    # - Block 2: A[:,2:4] (2x2) * B[2:4,:] (2x5) -> 2x5 matrix
    # - Block 3: A[:,4:5] (2x1) * B[4:5,:] (1x5) -> 2x5 matrix
    # 
    # Each block's output is a 2x5 matrix that gets added to C.
    
    # For Block 1 (2x2 * 2x5): We can further partition the 5 columns of B[0:2,:] into
    # {0,1}, {2,3}, {4} to get sub-blocks. But this gets complicated.
    
    # Let me just use the straightforward block partitioning where we partition the j-index
    # into groups, and for each group, we use an appropriate rank.
    
    # Simplest approach: partition j into {0,1}, {2,3}, {4}
    # For each group of size s, the sub-problem is (2 x s) * (s x 5) = 2 x 5
    # - Group {0,1}: 2x2 * 2x5, rank = 2*2*5 = 20 (schoolbook) or we can do better
    # - Group {2,3}: 2x2 * 2x5, rank = 20
    # - Group {4}: 2x1 * 1x5, rank = 2*1*5 = 10
    # Total: 50, which is worse than 50.
    
    # Let me try a different partition. What if we partition both i and k indices too?
    # C is 5x2, so k=0..4 and i=0..1.
    
    # Actually, the key insight is that we can partition the problem into independent
    # sub-problems if we partition the indices in a compatible way.
    
    # Let me just implement the block-diagonal approach with Strassen for 2x2x2 blocks.
    # Partition j into {0,1}, {2,3}, {4}.
    # For j in {0,1}: A[:,0:2] is 2x2, B[0:2,:] is 2x5.
    #   We can partition the 5 columns of B[0:2,:] into {0,1}, {2,3}, {4}.
    #   - Sub-block (0,1): A[:,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[0:2,0:2] (2x2), rank 7
    #   - Sub-block (0,2): A[:,0:2] (2x2) * B[0:2,2:4] (2x2) -> C[2:4,0:2] (2x2), rank 7
    #   - Sub-block (0,4): A[:,0:2] (2x2) * B[0:2,4:5] (2x1) -> C[4:5,0:2] (1x2), rank 2
    # For j in {2,3}: A[:,2:4] is 2x2, B[2:4,:] is 2x5.
    #   - Sub-block (2,0): A[:,2:4] (2x2) * B[2:4,0:2] (2x2) -> C[0:2,2:4]... wait, this doesn't work.
    
    # I'm overcomplicating this. Let me just use the schoolbook decomposition with rank 50
    # and see if we can find a better one through optimization.
    
    # For now, let's return the schoolbook decomposition.
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
