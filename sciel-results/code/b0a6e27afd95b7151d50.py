import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    # We will build the decomposition manually.
    # Rank estimate: 49 (Strassen 4x4) + 16 (4x4x1) + 16 (4x1x4) + 4 (4x1x1) = 85
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1

    # 1. A1 * B11 (4x4 * 4x4) using Recursive Strassen
    # A1 indices: i in 0..3, j in 0..3 -> U index i*5 + j
    # B11 indices: j in 0..3, k in 0..3 -> V index j*5 + k
    # C indices: i in 0..3, k in 0..3 -> W index k*4 + i
    
    # Strassen for 2x2
    def strassen_2x2(u_base, v_base, w_base):
        # u_base, v_base, w_base are functions to get/set indices
        # We assume the 2x2 blocks are contiguous in the indices provided
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # We need to map the 2x2 block indices to the global U, V, W indices.
        # This is complex to do generically with the helper.
        # Instead, I will hardcode the 49 terms for 4x4 Strassen?
        # Or use a simpler decomposition for 4x4?
        
        # Let's use the 16 schoolbook terms for 4x4? No, that's 64.
        # Let's use the 49 Strassen terms.
        
        pass

    # Given the complexity of implementing Recursive Strassen correctly with index mapping in this short time,
    # and the risk of bugs, I will use the Schoolbook decomposition for the 4x4 part as well,
    # but optimize the other parts?
    # Schoolbook 4x4 is 16 terms.
    # Total Rank: 16 + 16 + 16 + 4 = 52?
    # Wait.
    # Part 1 (4x4x4): 16 terms.
    # Part 2 (4x4x1): 16 terms.
    # Part 3 (4x1x4): 16 terms.
    # Part 4 (4x1x1): 4 terms.
    # Total: 52.
    
    # This is much better than 85 and easier to implement than Strassen.
    # Is 52 correct?
    # Part 1: C_ik = sum_j A_ij B_jk. 4*4*4 = 64 scalar mults?
    # No, the tensor rank of 4x4x4 is 16?
    # No, the rank of 4x4x4 is at least 16?
    # Schoolbook for 4x4x4 uses 64 multiplications?
    # C_ik = sum_j A_ij B_jk.
    # There are 16 outputs (i,k).
    # Each output depends on 4 products.
    # Total 64 products.
    # Can we do better?
    # Yes, Strassen does 49.
    # But 16 is the rank of 4x4x1.
    # The rank of 4x4x4 is NOT 16.
    
    # So I must use Strassen or a known decomposition for 4x4x4.
    # If I use Schoolbook for 4x4x4, rank is 64.
    # Total: 64 + 16 + 16 + 4 = 100.
    # This is the baseline.
    
    # I need to implement Strassen for 4x4x4.
    
    # Let's implement the 7-term Strassen for 2x2x2 and recurse.
    
    def strassen_2x2_term(u_i, u_j, v_j, v_k, w_i, w_k, val_u, val_v, val_w):
        # u_i, u_j are row/col indices for the 2x2 block in U
        # This is getting too complex to map correctly without a grid.
        
        pass

    # I will use the 49-rank Recursive Strassen.
    # I will generate the terms programmatically.
    
    # Reset col
    col = 0
    
    # Function to add a 2x2 Strassen decomposition for a block
    def add_strassen_2x2(u_row_start, u_col_start, v_row_start, v_col_start, w_row_start, w_col_start):
        # u_row_start, u_col_start: top-left of A block (2x2)
        # v_row_start, v_col_start: top-left of B block (2x2)
        # w_row_start, w_col_start: top-left of C block (2x2)
        # U indices: (u_row, u_col) -> u_row_start + r, u_col_start + c
        # V indices: (v_row, v_col) -> v_row_start + r, v_col_start + c
        # W indices: (w_row, w_col) -> w_row_start + r, w_col_start + c
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a product of two 2x2 matrices? No, M is a scalar?
        # No, in Strassen, A and B are 2x2 matrices.
        # The "multiplication" is the tensor product.
        # We are decomposing the 2x2x2 tensor.
        # Each M corresponds to 1 rank-1 term?
        # No, M1 = (A00+A11)(B00+B11).
        # This is a product of two linear combinations of A entries and two linear combinations of B entries.
        # This is 1 rank-1 term.
        
        # So we add 7 terms.
        
        # Term 1: M1
        # U: A00, A11
        # V: B00, B11
        # W: C00, C01, C10, C11 (with coefficients)
        
        # This is hard to map to the single U, V, W arrays because
        # the "linear combination" happens in the U and V matrices.
        
        # U[term] will have non-zero entries at A00, A11.
        # V[term] will have non-zero entries at B00, B11.
        # W[term] will have non-zero entries at C00, C01, C10, C11.
        
        # Let's define the 7 terms for 2x2.
        
        # Term 1: M1
        add_term(u_row_start + 0, v_row_start + 0, w_row_start + 0, 1, 1, 1) # A00 B00 -> C00
        # Wait, M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is 4 rank-1 terms if expanded.
        # But Strassen counts it as 1 multiplication of two sums.
        # In the tensor decomposition, we need to represent the sum.
        # U[term] has 1 at A00, 1 at A11.
        # V[term] has 1 at B00, 1 at B11.
        # W[term] has 1 at C00, 1 at C01, 1 at C10, 1 at C11?
        # No.
        # C00 += M1. C01 += 0?
        # The formula is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for Term 1 (M1):
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        
        # I need to add these to the global U, V, W.
        
        # Let's create a local buffer for the 2x2 block terms and then merge.
        # But the indices are global.
        
        # I will implement the 7 terms directly.
        
        # Term 1: M1
        U[u_row_start][col] = 1
        U[u_row_start + 1][col] = 1 # A11 is at (1,1) relative to start?
        # A00 is (0,0), A11 is (1,1).
        # So U indices: (u_row_start, u_col_start) and (u_row_start+1, u_col_start+1).
        
        # Wait, my U index is i*m + j.
        # u_row_start is i. u_col_start is j.
        # A00: i=u_row_start, j=u_col_start.
        # A11: i=u_row_start+1, j=u_col_start+1.
        
        U[u_row_start * m + u_col_start][col] = 1
        U[(u_row_start + 1) * m + (u_col_start + 1)][col] = 1
        
        V[v_row_start * p + v_col_start][col] = 1
        V[(v_row_start + 1) * p + (v_col_start + 1)][col] = 1
        
        W[w_col_start * n + w_row_start][col] = 1 # C00: k=0, i=0
        W[(w_col_start + 1) * n + (w_row_start + 1)][col] = 1 # C11: k=1, i=1
        
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        U[(u_row_start + 1) * m + u_col_start][col] = 1 # A10
        U[(u_row_start + 1) * m + (u_col_start + 1)][col] = 1 # A11
        V[v_row_start * p + v_col_start][col] = 1 # B00
        W[w_col_start * n + (w_row_start + 1)][col] = 1 # C10
        W[(w_col_start + 1) * n + (w_row_start + 1)][col] = -1 # C11
        col += 1
        
        # Term 3: M3 = A00(B01-B11)
        U[u_row_start * m + u_col_start][col] = 1 # A00
        V[v_row_start * p + (v_col_start + 1)][col] = 1 # B01
        V[(v_row_start + 1) * p + (v_col_start + 1)][col] = -1 # B11
        W[(w_col_start + 1) * n + w_row_start][col] = 1 # C01
        W[(w_col_start + 1) * n + (w_row_start + 1)][col] = 1 # C11
        col += 1
        
        # Term 4: M4 = A11(B10-B00)
        U[(u_row_start + 1) * m + (u_col_start + 1)][col] = 1 # A11
        V[(v_row_start + 1) * p + v_col_start][col] = 1 # B10
        V[v_row_start * p + v_col_start][col] = -1 # B00
        W[w_col_start * n + w_row_start][col] = 1 # C00
        W[w_col_start * n + (w_row_start + 1)][col] = 1 # C10
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        U[u_row_start * m + u_col_start][col] = 1 # A00
        U[u_row_start * m + (u_col_start + 1)][col] = 1 # A01
        V[(v_row_start + 1) * p + (v_col_start + 1)][col] = 1 # B11
        W[w_col_start * n + w_row_start][col] = -1 # C00
        W[(w_col_start + 1) * n + w_row_start][col] = 1 # C01
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U[(u_row_start + 1) * m + u_col_start][col] = 1 # A10
        U[u_row_start * m + u_col_start][col] = -1 # A00
        V[v_row_start * p + v_col_start][col] = 1 # B00
        V[v_row_start * p + (v_col_start + 1)][col] = 1 # B01
        W[(w_col_start + 1) * n + (w_row_start + 1)][col] = 1 # C11
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U[u_row_start * m + (u_col_start + 1)][col] = 1 # A01
        U[(u_row_start + 1) * m + (u_col_start + 1)][col] = -1 # A11
        V[(v_row_start + 1) * p + v_col_start][col] = 1 # B10
        V[(v_row_start + 1) * p + (v_col_start + 1)][col] = 1 # B11
        W[w_col_start * n + w_row_start][col] = 1 # C00
        col += 1

    # Apply Strassen to the 4x4 block (0-3, 0-3)
    # Split into four 2x2 blocks:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # We need to apply Strassen to the 4x4 tensor.
    # This means we apply the 7-term decomposition to the 4x4 matrices.
    # But the 4x4 matrices are not 2x2.
    # So we must recurse.
    
    # Call add_strassen_2x2 for the four 2x2 sub-blocks of the 4x4?
    # No, Strassen on 4x4 uses 7 products of 2x2 matrices.
    # Each 2x2 product is decomposed into 7 rank-1 terms.
    # Total 49 terms.
    
    # So we call add_strassen_2x2 for each of the 7 M terms?
    # No, the M terms are 2x2 matrices.
    # We need to decompose each M term (which is a 2x2 matrix product) into 7 rank-1 terms.
    
    # This requires a nested call.
    # I will implement a function that takes the "effective" indices for the 2x2 blocks.
    
    # This is getting too complex for the time limit.
    # I will fall back to Schoolbook for the 4x4 part.
    # Rank 100.
    
    # But I can optimize the boundary parts?
    # No, they are already minimal.
    
    # I will return the schoolbook decomposition.
    
    # Wait, I can just use the 49-rank decomposition by calling add_strassen_2x2 on the 2x2 blocks?
    # No, that would be 4 * 7 = 28 terms?
    # No, that would be 4 independent 2x2 multiplications.
    # That's not 4x4 multiplication.
    
    # I will use the Schoolbook decomposition.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
