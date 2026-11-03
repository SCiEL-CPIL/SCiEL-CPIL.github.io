"""Matrix multiplication 2x4x6 tensor decomposition - rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is valid."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x4x6(n, m, p):
    """
    Strassen-based decomposition for 2x4 * 4x6.
    
    View A as 2x2 blocks of 2x2 matrices, B as 2x2 blocks of 2x2 matrices.
    Use 7 Strassen multiplications, each a 2x2*2x2 = 4 scalar products.
    Total rank: 28.
    
    A (2x4) = [[A11, A12], [A21, A22]] where each block is 1x2
    Wait, that doesn't work for Strassen. Let me reconsider.
    
    Actually for 2x4 * 4x6:
    A is 2 rows, 4 cols. B is 4 rows, 6 cols. C is 2 rows, 6 cols.
    
    Split A into 2 column-blocks of width 2: A = [A_left, A_right] (2x2 each)
    Split B into 2 row-blocks of height 2 and 3 column-blocks of width 2:
    B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    
    C = A_left * B_top + A_right * B_bottom
    C = [A_left, A_right] * [[B11,B12,B13],[B21,B22,B23]]
      = A_left*[B11,B12,B13] + A_right*[B21,B22,B23]
    
    Each A_left*B_top is 2x2 * 2x3 = 2x3, using 6 scalar mults.
    Each A_right*B_bottom is 2x2 * 2x3 = 2x3, using 6 scalar mults.
    Total: 12. But that's just schoolbook split differently.
    
    For Strassen, we need square blocks. Let's use a different approach.
    
    Actually, the known rank for 2x4x6 is 24 (from literature).
    Let me try a different block structure.
    
    View as: A (2x4), B (4x6).
    Pad to 4x4 * 4x4 and use rank-20 for 4x4x4? No, that's worse.
    
    Let me try the direct approach: the tensor T[i,j,k] = delta(i,j,k) for the multiplication.
    The rank is the minimum r such that T = sum_r u_r ⊗ v_r ⊗ w_r.
    
    For 2x4x6, a known upper bound is 24 (from Bini et al.).
    Let me construct a rank-24 decomposition.
    
    Actually, let me just do the straightforward schoolbook (rank 48) and try to improve.
    """
    pass


def _rank24_decomposition(n, m, p):
    """
    Attempt rank-24 decomposition for 2x4x6.
    
    Based on the idea that we can use a structured decomposition.
    A is 2x4, B is 4x6.
    
    Let's use the following approach:
    Split the 4 middle dimension into 2+2.
    C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    
    Each term is (2x2) @ (2x6) = 2x6, requiring 12 scalar mults each.
    Total: 24.
    
    For A[:, 0:2] @ B[0:2, :]:
    A[:, 0:2] has entries A[0,0], A[0,1], A[1,0], A[1,1]
    B[0:2, :] has entries B[0,0], B[0,1], ..., B[0,5], B[1,0], ..., B[1,5]
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    ...
    
    Each scalar product in the 2x2 @ 2x6 gives one term.
    There are 2*2*6 = 24 such terms total (12 per block).
    
    This is just the schoolbook with rank 48, not 24.
    The issue is that each C[i,k] = sum_j A[i,j]*B[j,k], and we can't combine terms.
    
    Let me think about this differently. The rank of the tensor for m x n * n x p
    matrix multiplication is at most m*n*p (schoolbook).
    
    For 2x4x6, schoolbook gives rank 48.
    Strassen gives rank 42 (from the knowledge).
    
    Let me implement the rank-42 Strassen decomposition properly.
    """
    pass


def _strassen_decomposition():
    """
    Rank-42 Strassen decomposition for 2x4 * 4x6.
    
    View the multiplication as 2x2 block multiplication where:
    A is split into 2 blocks of size 2x2 (left and right halves)
    B is split into 2 blocks of size 2x2 (top and bottom halves) for the middle dim
    But B is 4x6, so we split it as:
    B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6
    
    And A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    
    C = A_left @ B_top + A_right @ B_bottom
    
    Each 2x2 @ 2x6 multiplication has rank 12 (schoolbook) or we can use
    a Strassen-like approach.
    
    Actually, 2x2 @ 2x6:
    A is 2x2, B is 2x6. C = A@B is 2x6.
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is 4*6 = 24 scalar products for the full 2x2@2x6.
    But we can view it as 6 separate 2x2@2x1 multiplications, each of rank 4 (or 3 with Strassen-like).
    
    Hmm, this is getting complicated. Let me just implement the known rank-42 result.
    
    The key insight from the knowledge: "Using Strassen's algorithm on 2x2 blocks reduces
    the rank from 48 to 42 by replacing 8 scalar multiplications per 2x2 block with 7."
    
    So we have 6 "2x2 blocks" (since 4*6/4 = 6), each contributing 7 instead of 8.
    Total: 6*7 = 42.
    
    The 6 blocks correspond to the 6 columns of C (since C is 2x6).
    For each column k of C, we compute C[:,k] = A @ B[:,k] where B[:,k] is 4x1.
    Split A into [A_left | A_right] (2x2 each) and B[:,k] into [B_top_k; B_bottom_k] (2x1 each).
    C[:,k] = A_left @ B_top_k + A_right @ B_bottom_k
    
    Each A_left @ B_top_k is 2x2 @ 2x1 = 2x1, which has 4 scalar products.
    Using Strassen on the 2x2 part:
    M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])  -- wait, this doesn't work for 2x2@2x1.
    
    Actually for 2x2 @ 2x1:
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    This has 4 scalar products. Can we do it in 3?
    M1 = (A[0,0]+A[1,1])*(B[0]+B[1])
    M2 = (A[0,1]+A[1,1])*B[0]  
    M3 = A[0,0]*(B[1]-B[0])
    Then: C[0] = ? C[1] = ?
    
    This doesn't decompose cleanly. The 2x2@2x1 tensor has rank 4 (it's just 4 independent products).
    
    Let me reconsider. The rank-42 result must come from a different block structure.
    
    Perhaps: view A as 2x4, split into 2 blocks of 2x2.
    View B as 4x6, split into 2 blocks of 2x6.
    C = A1@B1 + A2@B2 where A1,A2 are 2x2 and B1,B2 are 2x6.
    
    Each A_i @ B_i is 2x2 @ 2x6. The tensor for this has rank 2*2*6 = 24 (schoolbook).
    But we can do better.
    
    Actually, I think the correct interpretation is:
    Split the middle dimension (4) into 2+2.
    C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    A[:,0:2] is 2x2, B[0:2,:] is 2x6.
    A[:,2:4] is 2x2, B[2:4,:] is 2x6.
    
    For each 2x2 @ 2x6:
    View B as 3 blocks of 2x2: B = [B_col1, B_col2, B_col3]
    Then A @ B = [A@B_col1, A@B_col2, A@B_col3]
    Each A @ B_colj is 2x2 @ 2x2, which has rank 7 (Strassen).
    So 2x2 @ 2x6 has rank 3*7 = 21.
    Total: 2*21 = 42.
    
    Yes! That's the rank-42 construction.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))  # (8, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (12, 42)
    
    # Strassen 2x2@2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A01+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In tensor form, each Mx = sum of products of linear forms in A and B.
    # For the tensor decomposition, each "multiplication" Mx contributes rank-1 terms.
    # But Mx itself is a single scalar product of two linear forms, so it's rank 1
    # in the bilinear form sense.
    #
    # Wait, no. In the tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r,
    # each term is a rank-1 tensor. The Strassen algorithm gives 7 bilinear forms,
    # but each bilinear form Mx = f_x(A) * g_x(B) is a single product, so it
    # corresponds to one rank-1 term in the tensor.
    #
    # But we also need to express C in terms of the Mx's, which involves
    # C[i,k] = sum_x coeff_{i,k,x} * Mx
    # This means w_{i,k} = sum_x coeff_{i,k,x} * w_x where w_x is the vector
    # for multiplication x.
    #
    # So for each Strassen multiplication x (x=1..7) and each output position (i,k),
    # the coefficient tells us how Mx contributes to C[i,k].
    #
    # In the tensor: T[i,j,k] = sum over all terms
    # For Strassen: T = sum_{x=1}^{7} sum_{(i,j) in relevant} ... 
    #
    # Actually, let me think of it this way:
    # The full tensor T has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k], else 0. More precisely, T[i,j,k] = delta(j, j') where j' is
    # the column of A / row of B that connects them.
    #
    # For our case: T is a tensor of shape (2*4, 4*6, 6*2) = (8, 24, 12).
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if j == j' and i == i' and k == k', else 0.
    # Wait, that's the schoolbook. The tensor for matrix multiplication is:
    # T[i, j, k] = 1 (for the standard basis), meaning T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where the indices are (row of A, col of A = row of B, col of B).
    #
    # Actually the tensor is: T_{i,j,k} where i indexes rows of A (0..n-1),
    # j indexes cols of A = rows of B (0..m-1), k indexes cols of B (0..p-1).
    # T_{i,j,k} = 1 for all i,j,k (it's the all-ones tensor in this indexing).
    # No wait, that's not right either.
    #
    # The matrix multiplication tensor T is defined by:
    # T(A, B) = A @ B
    # In components: (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T_{i,j,k} = 1 for all valid i,j,k. The tensor has entries T[i,j,k] = 1.
    #
    # In the flattened form with U, V, W:
    # U has shape (n*m, rank): U[i*m+j, r]
    # V has shape (m*p, rank): V[j*p+k, r]  
    # W has shape (p*n, rank): W[k*n+i, r]
    #
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # For Strassen on 2x2 blocks:
    # We have 3 groups of 7 multiplications (one for each column-pair of B).
    # For group g (g=0,1,2), corresponding to columns 2g, 2g+1 of B:
    #   A_block = A[:, 0:2] or A[:, 2:4] depending on which half of the middle dim
    #   B_block = B[0:2, 2g:2g+2] or B[2:4, 2g:2g+2]
    #
    # Wait, I need to be more careful. Let me re-derive.
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    # = (A_left @ B_left)[i,k] + (A_right @ B_right)[i,k]
    #
    # where A_left = A[:,0:2] (2x2), B_left = B[0:2,:] (2x6)
    #       A_right = A[:,2:4] (2x2), B_right = B[2:4,:] (2x6)
    #
    # For A_left @ B_left (2x2 @ 2x6):
    # Split B_left into 3 column-pairs: B_left[:,0:2], B_left[:,2:4], B_left[:,4:6]
    # Each is 2x2.
    # (A_left @ B_left)[:, 2g:2g+2] = A_left @ B_left[:, 2g:2g+2] (2x2 @ 2x2)
    #
    # So we have 6 total 2x2@2x2 multiplications:
    # Group (left, col0): A_left @ B_left[:,0:2]
    # Group (left, col1): A_left @ B_left[:,2:4]
    # Group (left, col2): A_left @ B_left[:,4:6]
    # Group (right, col0): A_right @ B_right[:,0:2]
    # Group (right, col1): A_right @ B_right[:,2:4]
    # Group (right, col2): A_right @ B_right[:,4:6]
    #
    # Each uses 7 Strassen multiplications. Total: 6*7 = 42.
    
    # Strassen 2x2@2x2 coefficients:
    # For A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a01+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx = f_x(A) * g_x(B) where f_x and g_x are linear forms.
    #
    # For the tensor decomposition, each Mx contributes one rank-1 term:
    # u_r = f_x evaluated on the A indices
    # v_r = g_x evaluated on the B indices
    # w_r = coefficient of Mx in C[i,k] for the appropriate (i,k)
    
    # Define the 7 Strassen linear forms:
    # f1(A) = a00 + a11, g1(B) = b00 + b11
    # f2(A) = a01 + a11, g2(B) = b00
    # f3(A) = a00,         g3(B) = b01 - b11
    # f4(A) = a11,         g4(B) = b10 - b00
    # f5(A) = a00 + a01,   g5(B) = b11
    # f6(A) = a10 - a00,   g6(B) = b00 + b01
    # f7(A) = a01 - a11,   g7(B) = b10 + b11
    
    # Coefficients for C:
    # C00: M1:+1, M4:+1, M5:-1, M7:+1
    # C01: M3:+1, M5:+1
    # C10: M2:+1, M4:+1
    # C11: M1:+1, M2:-1, M3:+1, M6:+1
    
    strassen_f = [
        # (row, col, coeff) for A
        [(0, 0, 1), (1, 1, 1)],          # f1
        [(0, 1, 1), (1, 1, 1)],          # f2
        [(0, 0, 1)],                      # f3
        [(1, 1, 1)],                      # f4
        [(0, 0, 1), (0, 1, 1)],          # f5
        [(1, 0, 1), (0, 0, -1)],         # f6
        [(0, 1, 1), (1, 1, -1)],         # f7
    ]
    strassen_g = [
        # (row, col, coeff) for B
        [(0, 0, 1), (1, 1, 1)],          # g1
        [(0, 0, 1)],                      # g2
        [(0, 1, 1), (1, 1, -1)],         # g3
        [(1, 0, 1), (0, 0, -1)],         # g4
        [(1, 1, 1)],                      # g5
        [(0, 0, 1), (0, 1, 1)],          # g6
        [(1, 0, 1), (1, 1, 1)],          # g7
    ]
    # C coefficients: C[ci, cj] for ci,cj in {0,1}
    # ci=0,cj=0: M1:+1, M4:+1, M5:-1, M7:+1
    # ci=0,cj=1: M3:+1, M5:+1
    # ci=1,cj=0: M2:+1, M4:+1
    # ci=1,cj=1: M1:+1, M2:-1, M3:+1, M6:+1
    strassen_c = [
        # (mi, coeff) for each C position
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00 (0-indexed: M0=M1, M3=M4, M4=M5, M6=M7)
        [(2, 1), (4, 1)],                      # C01
        [(1, 1), (3, 1)],                      # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],    # C11
    ]
    
    col = 0
    # 6 groups: (a_half, b_half, col_offset)
    # a_half: 0 = left (cols 0-1 of A), 1 = right (cols 2-3 of A)
    # b_half: 0 = top (rows 0-1 of B), 1 = bottom (rows 2-3 of B)
    # col_offset: which pair of columns of B (0, 1, or 2)
    
    groups = [
        (0, 0, 0),  # A_left @ B_top[:, 0:2]
        (0, 0, 1),  # A_left @ B_top[:, 2:4]
        (0, 0, 2),  # A_left @ B_top[:, 4:6]
        (1, 0, 0),  # A_right @ B_top[:, 0:2]
        (1, 0, 1),  # A_right @ B_top[:, 2:4]
        (1, 0, 2),  # A_right @ B_top[:, 4:6]
    ]
    # Wait, I made an error. Let me redo.
    # A_left = A[:, 0:2], B_left = B[0:2, :]
    # A_right = A[:, 2:4], B_right = B[2:4, :]
    #
    # For A_left @ B_left: A_left is 2x2 (rows 0-1, cols 0-1 of A)
    # B_left is 2x6 (rows 0-1 of B, all 6 cols)
    # Split B_left into 3 blocks of 2x2: cols 0-1, 2-3, 4-5
    #
    # For A_right @ B_right: A_right is 2x2 (rows 0-1, cols 2-3 of A)
    # B_right is 2x6 (rows 2-3 of B, all 6 cols)
    # Split B_right into 3 blocks of 2x2: cols 0-1, 2-3, 4-5
    
    groups = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
        # a_row_offset: row offset in A (always 0 since A is 2 rows)
        # a_col_offset: col offset in A (0 for left, 2 for right)
        # b_row_offset: row offset in B (0 for top, 2 for bottom)
        # b_col_offset: col offset in B (0, 2, or 4)
        (0, 0, 0, 0),   # A[0:2, 0:2] @ B[0:2, 0:2]
        (0, 0, 0, 2),   # A[0:2, 0:2] @ B[0:2, 2:4]
        (0, 0, 0, 4),   # A[0:2, 0:2] @ B[0:2, 4:6]
        (0, 2, 2, 0),   # A[0:2, 2:4] @ B[2:4, 0:2]
        (0, 2, 2, 2),   # A[0:2, 2:4] @ B[2:4, 2:4]
        (0, 2, 2, 4),   # A[0:2, 2:4] @ B[2:4, 4:6]
    ]
    
    for (a_ro, a_co, b_ro, b_co) in groups:
        for mx in range(7):
            # Set U entries for f_mx
            for (ar, ac, coeff) in strassen_f[mx]:
                i = a_ro + ar  # row in A (0 or 1)
                j = a_co + ac  # col in A (0-3)
                U[i * m + j, col] = coeff
            
            # Set V entries for g_mx
            for (br, bc, coeff) in strassen_g[mx]:
                j = b_ro + br  # row in B (0-3)
                k = b_co + bc  # col in B (0-5)
                V[j * p + k, col] = coeff
            
            # Set W entries for C contributions
            for (ci, cj) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                # Find coefficient of M_mx in C[ci, cj]
                coeff_c = 0
                for (mi, c) in strassen_c[ci * 2 + cj]:
                    if mi == mx:
                        coeff_c = c
                        break
                if coeff_c != 0:
                    i = ci  # row in C (0 or 1)
                    k = b_co + cj  # col in C (0-5)
                    W[k * n + i, col] = coeff_c
            
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    # Try rank-42 Strassen decomposition
    U, V, W = _strassen_decomposition()
    
    if _verify(U, V, W, n, m, p, tol=1e-6):
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    if time.time() > deadline - 5:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _schoolbook(n, m, p)
    if _verify(U, V, W, n, m, p, tol=1e-6):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
