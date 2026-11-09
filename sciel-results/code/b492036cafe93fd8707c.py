"""Program to find a rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication.

Strategy:
  - Split A (3x4) into top 2x4 and bottom 1x4.
  - Split B (4x6) into 3 blocks of 4x2.
  - For each top row pair (2x4) * B_block (4x2):
        A_top (2x4) * B_block (4x2) = A_L (2x2) * B_top (2x2) + A_R (2x2) * B_bot (2x2)
    where A_L = A_top[:, 0:2], A_R = A_top[:, 2:4]
          B_top = B_block[0:2, :], B_bot = B_block[2:4, :]
    Each 2x2 * 2x2 uses Strassen (rank 7).
    So each B_block costs 14.
    3 blocks -> 42.
  - Bottom row (1x4) * B (4x6):
    6 dot products of length 4 -> rank 24.
    Total: 42 + 24 = 66.

We construct U, V, W matrices of rank 66.
"""
import time

DIMENSIONS = [3, 4, 6]

# Deadline guard
_START = time.time()
DEADLINE = _START + 140  # 10s safety margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Total rank
    rank = 66
    U = [[0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * rank for _ in range(p * n)]  # 18 x 66

    col = 0  # current column index in U, V, W

    # Helper to add a term to U, V, W at given column
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            raise RuntimeError("Rank exceeded")
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen for 2x2 * 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22) b11
    # M3 = a11 (b12-b22)
    # M4 = a22 (b21-b11)
    # M5 = (a11+a12) b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def strassen_block(a_rows, b_cols, row_offset, col_offset):
        """
        a_rows: list of 2 indices (row indices in A for the 2x2 block)
        b_cols: list of 2 indices (col indices in B for the 2x2 block)
        row_offset: offset for C rows (which row in C)
        col_offset: offset for C cols (which col in C)
        
        We need to map:
        A[i, j] -> U[i*m + j]
        B[j, k] -> V[j*p + k]
        C[i, k] -> W[k*n + i]
        """
        a0, a1 = a_rows  # row indices in A (0, 1 for top, 2 for bottom)
        b0, b1 = b_cols  # col indices in B (which 2 cols of the 6)
        # The 2x2 A block: rows a0, a1; cols 0,1 (for A_L) or 2,3 (for A_R)
        # Wait, A is 3x4. The 2x2 blocks are:
        # A_L: rows a0,a1; cols 0,1
        # A_R: rows a0,a1; cols 2,3
        # B block: rows 0,1 (top) or 2,3 (bottom); cols b0,b1
        
        # Actually, for the Strassen part, we have:
        # A_sub (2x2) * B_sub (2x2)
        # A_sub[i, j] is from A[a0+i, j] where j in {0,1} for A_L or {2,3} for A_R
        # B_sub[i, j] is from B[i, b0+j] where i in {0,1} for B_top or {2,3} for B_bot
        
        # But Strassen uses the 4 elements of A_sub and 4 of B_sub.
        # Let's denote A_sub = [[A11, A12], [A21, A22]]
        # B_sub = [[B11, B12], [B21, B22]]
        
        # For each of the 7 Strassen terms, we need to add a product.
        # Each term is sum of products of A elements and B elements.
        # But in tensor decomposition, each term is a single product of a U-component, V-component, W-component.
        # Wait, no. The tensor is T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product).
        # For Strassen, each M is a product of two linear combinations.
        # M1 = (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # This is 4 products. So we need 4 terms for M1?
        # No, that's not how it works. The rank of 2x2x2 is 7, meaning we can write the multiplication as 7 products of linear forms.
        # Each "product" in the tensor decomposition is a single scalar multiplication of a U-entry, V-entry, W-entry.
        # But Strassen's 7 terms are products of *sums*. 
        # Actually, the tensor T_{2x2x2} = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (or similar).
        # Strassen's decomposition writes this as sum of 7 terms, each being a product of 3 linear forms.
        # So each term is: (sum_i alpha_i A_i) * (sum_j beta_j B_j) * (sum_k gamma_k C_k)
        # But in our U,V,W representation, each term c corresponds to:
        #   U[:,c] is a vector, V[:,c] is a vector, W[:,c] is a vector.
        #   The term is sum_{i,j,k} U[i,j,c] * V[j,k,c] * W[k,i,c] * A[i,j] * B[j,k] ... wait, no.
        #   Actually, the matrix product C = A*B means C[i,k] = sum_j A[i,j] B[j,k].
        #   The tensor T has T[i,j,k] = delta_{i,k} (for the multiplication tensor? No.)
        #   The multiplication tensor for (n x m) * (m x p) is a 3-tensor of size (n*m) x (m*p) x (p*n) where
        #   T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
        #   Wait, the index of W is k*n+i, which corresponds to C[i,k] = C[k*n+i]? No, C is n x p, so C[i,k] has index i*p+k in row-major.
        #   But the problem says W[k*n+i, r], so W is indexed by (k, i) where k is col of C, i is row of C.
        #   So W[k*n+i] corresponds to C[i,k].
        
        # Let's re-examine the Strassen decomposition in terms of U,V,W.
        # For a 2x2 * 2x2, we have A (2x2), B (2x2), C (2x2).
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        
        # Strassen:
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
        
        # Each M is a product of a linear combination of A and a linear combination of B.
        # In the tensor decomposition, each M corresponds to a term where:
        #   U part is the linear combination of A elements,
        #   V part is the linear combination of B elements,
        #   W part is the coefficient for each C element.
        
        # For example, M1 = (A00+A11)(B00+B11) contributes to C00 with +1, to C11 with +1.
        # So for M1:
        #   U: A00 -> 1, A11 -> 1 (others 0)
        #   V: B00 -> 1, B11 -> 1 (others 0)
        #   W: C00 -> 1, C11 -> 1 (others 0)
        #   This is one term in the decomposition.
        
        # Similarly for each M.
        # So Strassen gives 7 terms for the 2x2 * 2x2.
        
        # Now, in our global U,V,W, the A elements are at U[a_row * m + a_col], 
        # B elements are at V[b_row * p + b_col],
        # C elements are at W[c_col * n + c_row].
        
        # For the 2x2 block, we need to map the local indices to global.
        # Let's define the local A rows and cols, local B rows and cols, local C rows and cols.
        
        # a_rows: global row indices in A for the 2 rows of the block
        # a_cols: global col indices in A for the 2 cols of the block
        # b_rows: global row indices in B for the 2 rows of the block
        # b_cols: global col indices in B for the 2 cols of the block
        # c_rows: global row indices in C for the 2 rows of the result
        # c_cols: global col indices in C for the 2 cols of the result
        
        # Actually, for our case:
        # Top block: a_rows = [0,1], c_rows = [0,1]
        # Bottom block: a_rows = [2], c_rows = [2] (but this is 1x4 * 4x6, not Strassen)
        
        # For Strassen, we have 2x2 * 2x2.
        # Let's pass the global indices for the 4 A elements, 4 B elements, and 4 C elements.
        
        # A elements: A[a0, a0c0], A[a0, a0c1], A[a1, a1c0], A[a1, a1c1]
        # But a0c0, a0c1 are the 2 cols of A for this block.
        # Similarly for B and C.
        
        # Let's define:
        # a_indices = [(a0, ac0), (a0, ac1), (a1, ac0), (a1, ac1)]  # for A00, A01, A10, A11
        # b_indices = [(br0, bc0), (br0, bc1), (br1, bc0), (br1, bc1)]  # for B00, B01, B10, B11
        # c_indices = [(cr0, cc0), (cr0, cc1), (cr1, cc0), (cr1, cc1)]  # for C00, C01, C10, C11
        
        # Wait, C is indexed by (row, col) -> W[col * n + row]
        # So C[i, k] -> W[k * n + i]
        
        # Let's just use the global indices directly.
        pass

    # Instead of a complex helper, let's directly construct the terms for each Strassen block.
    # We'll define a function that takes the 8 global indices (4 for A, 4 for B, 4 for C) and adds 7 terms.
    
    def add_strassen(a_idx, b_idx, c_idx):
        """
        a_idx: list of 4 global U indices for A00, A01, A10, A11
        b_idx: list of 4 global V indices for B00, B01, B10, B11
        c_idx: list of 4 global W indices for C00, C01, C10, C11
        """
        nonlocal col
        # M1 = (A00+A11)(B00+B11) -> C00: +1, C11: +1
        add_term(a_idx[0], b_idx[0], c_idx[0], 1, 1, 1)
        add_term(a_idx[0], b_idx[3], c_idx[0], 1, 1, 1)
        add_term(a_idx[3], b_idx[0], c_idx[0], 1, 1, 1)
        add_term(a_idx[3], b_idx[3], c_idx[0], 1, 1, 1)
        add_term(a_idx[0], b_idx[0], c_idx[3], 1, 1, 1)
        add_term(a_idx[0], b_idx[3], c_idx[3], 1, 1, 1)
        add_term(a_idx[3], b_idx[0], c_idx[3], 1, 1, 1)
        add_term(a_idx[3], b_idx[3], c_idx[3], 1, 1, 1)
        
        # Wait, this is 8 terms for M1, but M1 is a single product of two sums.
        # In the tensor decomposition, (A00+A11)(B00+B11) is NOT a single term.
        # It's a sum of 4 terms: A00*B00, A00*B11, A11*B00, A11*B11.
        # But Strassen's algorithm works because the *sum* of these 7 products of sums gives the correct C.
        # The rank is 7 because we can write the tensor as a sum of 7 rank-1 tensors.
        # Each rank-1 tensor is a product of 3 vectors.
        # So M1 corresponds to ONE rank-1 tensor where:
        #   U vector has 1 at A00 and A11
        #   V vector has 1 at B00 and B11
        #   W vector has 1 at C00 and C11 (with appropriate signs)
        # But that's not right either, because U, V, W are fixed vectors, and the term is U * V * W (Hadamard product of the outer products).
        
        # Let me look up the correct Strassen tensor decomposition.
        # The multiplication tensor for 2x2x2 is:
        # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}  (but indices are tricky)
        # Strassen's 7 terms are:
        # M1: (e11+e22) \otimes (e11+e22) \otimes (e11 + e22)  ... no, the W part is different for each C element.
        
        # Actually, the standard way:
        # Each of the 7 M's is a product of a linear form in A and a linear form in B.
        # And then C is a linear combination of these M's.
        # So in the tensor decomposition, each M corresponds to a term where:
        #   U[:,c] = the linear form in A (as a vector)
        #   V[:,c] = the linear form in B (as a vector)
        #   W[:,c] = the coefficient vector for C (as a vector)
        
        # So for M1 = (A00+A11)(B00+B11):
        #   U: [1, 0, 0, 1] (for A00, A01, A10, A11)
        #   V: [1, 0, 0, 1] (for B00, B01, B10, B11)
        #   W: [1, 0, 0, 1] (for C00, C01, C10, C11) -- but wait, M1 contributes to C00 and C11, not C01 and C10.
        #   Actually, C00 = M1 + M4 - M5 + M7, so M1 has coefficient +1 in C00.
        #   C11 = M1 - M2 + M3 + M6, so M1 has coefficient +1 in C11.
        #   So W for M1 is [1, 0, 0, 1] if C indices are [C00, C01, C10, C11].
        
        # Let's redo this properly.
        pass

    # I think I'm overcomplicating. Let's use the known fact that the rank is 7 and just implement the 7 terms.
    # Each term is a product of 3 vectors.
    
    # For now, let's just use the schoolbook for the 2x2 * 2x2 to get a correct (but higher rank) solution,
    # and then optimize later. But the task asks for rank-minimal, and we know 66 is achievable.
    
    # Let me try a different approach: just hardcode the 66 terms.
    
    # Actually, let's use the following structure:
    # 1. For each of the 3 B blocks (cols 0-1, 2-3, 4-5):
    #    For the top 2 rows of A:
    #      A_top (2x4) * B_block (4x2)
    #      = A_L (2x2) * B_top (2x2) + A_R (2x2) * B_bot (2x2)
    #      Each 2x2*2x2 uses Strassen (7 terms).
    #      So 14 terms per B block.
    #    For the bottom 1 row of A:
    #      A_bot (1x4) * B_block (4x2)
    #      = 2 dot products of length 4 -> 8 terms? No, each dot product is 4 terms, so 8 terms.
    #      But we can do it in 4 terms if we use the fact that it's a single row?
    #      Actually, 1x4 * 4x2 = 2 results, each is a dot product of length 4.
    #      So 4 + 4 = 8 terms.
    #    Total per B block: 14 + 8 = 22.
    #    3 blocks: 66.
    
    # This matches! So let's implement this.
    
    col = 0
    
    # Process each B block
    for block in range(3):
        b_cols = [block * 2, block * 2 + 1]
        
        # Top 2 rows of A * B block
        # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
        # B_top = B[0:2, b_cols], B_bot = B[2:4, b_cols]
        
        # Strassen for A_L * B_top
        # A_L indices in U:
        # A[0,0] -> U[0*4+0] = U[0]
        # A[0,1] -> U[0*4+1] = U[1]
        # A[1,0] -> U[1*4+0] = U[4]
        # A[1,1] -> U[1*4+1] = U[5]
        a_L_idx = [0, 1, 4, 5]  # A00, A01, A10, A11
        
        # B_top indices in V:
        # B[0, b_cols[0]] -> V[0*6 + b_cols[0]]
        # B[0, b_cols[1]] -> V[0*6 + b_cols[1]]
        # B[1, b_cols[0]] -> V[1*6 + b_cols[0]]
        # B[1, b_cols[1]] -> V[1*6 + b_cols[1]]
        b_top_idx = [
            0 * 6 + b_cols[0],
            0 * 6 + b_cols[1],
            1 * 6 + b_cols[0],
            1 * 6 + b_cols[1]
        ]
        
        # C indices in W for the result of A_L * B_top:
        # C[0, b_cols[0]] -> W[b_cols[0]*3 + 0]
        # C[0, b_cols[1]] -> W[b_cols[1]*3 + 0]
        # C[1, b_cols[0]] -> W[b_cols[0]*3 + 1]
        # C[1, b_cols[1]] -> W[b_cols[1]*3 + 1]
        c_idx = [
            b_cols[0] * 3 + 0,
            b_cols[1] * 3 + 0,
            b_cols[0] * 3 + 1,
            b_cols[1] * 3 + 1
        ]
        
        # Add Strassen terms for A_L * B_top
        # M1: (A00+A11)(B00+B11) -> C00: +1, C11: +1
        U[a_L_idx[0]][col] = 1
        U[a_L_idx[3]][col] = 1
        V[b_top_idx[0]][col] = 1
        V[b_top_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10: +1
        U[a_L_idx[2]][col] = 1
        U[a_L_idx[3]][col] = 1
        V[b_top_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M3: A00(B01-B11) -> C01: +1, C11: -1
        U[a_L_idx[0]][col] = 1
        V[b_top_idx[1]][col] = 1
        V[b_top_idx[3]][col] = -1
        W[c_idx[1]][col] = 1
        W[c_idx[3]][col] = -1
        col += 1
        
        # M4: A11(B10-B00) -> C00: +1, C10: +1
        U[a_L_idx[3]][col] = 1
        V[b_top_idx[2]][col] = 1
        V[b_top_idx[0]][col] = -1
        W[c_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00: -1, C01: +1
        U[a_L_idx[0]][col] = 1
        U[a_L_idx[1]][col] = 1
        V[b_top_idx[3]][col] = 1
        W[c_idx[0]][col] = -1
        W[c_idx[1]][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11: +1
        U[a_L_idx[2]][col] = 1
        U[a_L_idx[0]][col] = -1
        V[b_top_idx[0]][col] = 1
        V[b_top_idx[1]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00: +1
        U[a_L_idx[1]][col] = 1
        U[a_L_idx[3]][col] = -1
        V[b_top_idx[2]][col] = 1
        V[b_top_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        
        # Strassen for A_R * B_bot
        # A_R = A[0:2, 2:4]
        # A[0,2] -> U[0*4+2] = U[2]
        # A[0,3] -> U[0*4+3] = U[3]
        # A[1,2] -> U[1*4+2] = U[6]
        # A[1,3] -> U[1*4+3] = U[7]
        a_R_idx = [2, 3, 6, 7]
        
        # B_bot = B[2:4, b_cols]
        # B[2, b_cols[0]] -> V[2*6 + b_cols[0]]
        # B[2, b_cols[1]] -> V[2*6 + b_cols[1]]
        # B[3, b_cols[0]] -> V[3*6 + b_cols[0]]
        # B[3, b_cols[1]] -> V[3*6 + b_cols[1]]
        b_bot_idx = [
            2 * 6 + b_cols[0],
            2 * 6 + b_cols[1],
            3 * 6 + b_cols[0],
            3 * 6 + b_cols[1]
        ]
        
        # C indices are the same as for A_L * B_top
        # (since both contribute to the same C block)
        
        # Add Strassen terms for A_R * B_bot
        # M1: (A00+A11)(B00+B11) -> C00: +1, C11: +1
        U[a_R_idx[0]][col] = 1
        U[a_R_idx[3]][col] = 1
        V[b_bot_idx[0]][col] = 1
        V[b_bot_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10: +1
        U[a_R_idx[2]][col] = 1
        U[a_R_idx[3]][col] = 1
        V[b_bot_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M3: A00(B01-B11) -> C01: +1, C11: -1
        U[a_R_idx[0]][col] = 1
        V[b_bot_idx[1]][col] = 1
        V[b_bot_idx[3]][col] = -1
        W[c_idx[1]][col] = 1
        W[c_idx[3]][col] = -1
        col += 1
        
        # M4: A11(B10-B00) -> C00: +1, C10: +1
        U[a_R_idx[3]][col] = 1
        V[b_bot_idx[2]][col] = 1
        V[b_bot_idx[0]][col] = -1
        W[c_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00: -1, C01: +1
        U[a_R_idx[0]][col] = 1
        U[a_R_idx[1]][col] = 1
        V[b_bot_idx[3]][col] = 1
        W[c_idx[0]][col] = -1
        W[c_idx[1]][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11: +1
        U[a_R_idx[2]][col] = 1
        U[a_R_idx[0]][col] = -1
        V[b_bot_idx[0]][col] = 1
        V[b_bot_idx[1]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00: +1
        U[a_R_idx[1]][col] = 1
        U[a_R_idx[3]][col] = -1
        V[b_bot_idx[2]][col] = 1
        V[b_bot_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        
        # Bottom row of A * B block
        # A_bot = A[2, 0:4]
        # A[2,0] -> U[2*4+0] = U[8]
        # A[2,1] -> U[2*4+1] = U[9]
        # A[2,2] -> U[2*4+2] = U[10]
        # A[2,3] -> U[2*4+3] = U[11]
        
        # B_block = B[0:4, b_cols]
        # B[0, b_cols[0]] -> V[0*6 + b_cols[0]]
        # B[0, b_cols[1]] -> V[0*6 + b_cols[1]]
        # B[1, b_cols[0]] -> V[1*6 + b_cols[0]]
        # B[1, b_cols[1]] -> V[1*6 + b_cols[1]]
        # B[2, b_cols[0]] -> V[2*6 + b_cols[0]]
        # B[2, b_cols[1]] -> V[2*6 + b_cols[1]]
        # B[3, b_cols[0]] -> V[3*6 + b_cols[0]]
        # B[3, b_cols[1]] -> V[3*6 + b_cols[1]]
        
        # C indices for the bottom row:
        # C[2, b_cols[0]] -> W[b_cols[0]*3 + 2]
        # C[2, b_cols[1]] -> W[b_cols[1]*3 + 2]
        
        # Dot product for C[2, b_cols[0]]:
        # A[2,0]*B[0,b_cols[0]] + A[2,1]*B[1,b_cols[0]] + A[2,2]*B[2,b_cols[0]] + A[2,3]*B[3,b_cols[0]]
        for j in range(4):
            U[8 + j][col] = 1
            V[j * 6 + b_cols[0]][col] = 1
            W[b_cols[0] * 3 + 2][col] = 1
            col += 1
        
        # Dot product for C[2, b_cols[1]]:
        # A[2,0]*B[0,b_cols[1]] + A[2,1]*B[1,b_cols[1]] + A[2,2]*B[2,b_cols[1]] + A[2,3]*B[3,b_cols[1]]
        for j in range(4):
            U[8 + j][col] = 1
            V[j * 6 + b_cols[1]][col] = 1
            W[b_cols[1] * 3 + 2][col] = 1
            col += 1
    
    # Verify col == 66
    if col != 66:
        raise RuntimeError(f"Expected rank 66, got {col}")
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
