import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will construct U, V, W matrices.
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (36, rank)
    # W: (p*n, rank) = (36, rank)
    
    # Strategy:
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A*B1 + A*B2 + A*B3.
    # Rank(A*B1) <= 49 (Strassen 4x4)
    # Rank(A*B2) <= 49 (Strassen 4x4)
    # Rank(A*B3) <= 16 (Schoolbook 4x4x1)
    # Total Rank = 114.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 4x4 decomposition for a specific B slice
    # B_slice is defined by column offset in B.
    # A is always the same 4x4 matrix.
    # We need to map the 49 scalar multiplications to U, V, W.
    
    # Strassen 2x2 definitions:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4, we apply this to 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    
    # The 7 products M1..M7 are 2x2 matrix multiplications.
    # Each M_i = X_i * Y_i where X_i, Y_i are 2x2 matrices formed by sums/diffs of A, B blocks.
    # We need to decompose each 2x2 multiplication into 7 scalar multiplications.
    
    # Let's define the 7 scalar multiplications for a generic 2x2 multiplication C = X*Y.
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]].
    # s1 = (x11+x22)(y11+y22)
    # s2 = (x21+x22)y11
    # s3 = x11(y12-y22)
    # s4 = x22(y21-y11)
    # s5 = (x11+x12)y22
    # s6 = (x21-x11)(y11+y12)
    # s7 = (x12-x22)(y21+y22)
    
    # c11 = s1 + s4 - s5 + s7
    # c12 = s3 + s5
    # c21 = s2 + s4
    # c22 = s1 - s2 + s3 + s6
    
    # We will iterate over the 7 Strassen products for the 4x4 case.
    # For each product, we identify the 2x2 matrices X and Y in terms of A and B blocks.
    # Then we expand each of the 7 scalar products.
    
    # A blocks (indices in 4x4 A):
    # A11: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,4,5 in flat A (row-major, m=4)
    # Wait, U is (n*m, rank) = (16, rank). Index is i*m + j.
    # A is 4x4.
    # A11 (top-left 2x2): rows 0,1; cols 0,1. Indices: 0,1,4,5.
    # A12 (top-right 2x2): rows 0,1; cols 2,3. Indices: 2,3,6,7.
    # A21 (bottom-left 2x2): rows 2,3; cols 0,1. Indices: 8,9,12,13.
    # A22 (bottom-right 2x2): rows 2,3; cols 2,3. Indices: 10,11,14,15.
    
    # B blocks for a 4x4 slice starting at column offset `b_off`:
    # B11: rows 0,1; cols b_off, b_off+1. Indices in V (m*p, rank): row j, col k.
    # V index is j*p + k. m=4, p=9.
    # B11 (top-left of slice): rows 0,1; cols b_off, b_off+1.
    # Indices: 0*9+b_off, 1*9+b_off, 0*9+b_off+1, 1*9+b_off+1.
    # B12 (top-right of slice): rows 0,1; cols b_off+2, b_off+3.
    # B21 (bottom-left of slice): rows 2,3; cols b_off, b_off+1.
    # B22 (bottom-right of slice): rows 2,3; cols b_off+2, b_off+3.
    
    # W indices: k*n + i. p=9, n=4.
    # C is 4x4 slice.
    # C11: rows 0,1; cols 0,1. Indices: 0*4+0, 1*4+0, 0*4+1, 1*4+1.
    # C12: rows 0,1; cols 2,3.
    # C21: rows 2,3; cols 0,1.
    # C22: rows 2,3; cols 2,3.
    
    # We define the 7 Strassen steps for 4x4.
    # Step 1: M1 = (A11+A22)(B11+B22)
    # Step 2: M2 = (A21+A22)B11
    # Step 3: M3 = A11(B12-B22)
    # Step 4: M4 = A22(B21-B11)
    # Step 5: M5 = (A11+A12)B22
    # Step 6: M6 = (A21-A11)(B11+B12)
    # Step 7: M7 = (A12-A22)(B21+B22)
    
    # For each step, we have a 2x2 multiplication X*Y.
    # We need to decompose X*Y into 7 scalar multiplications.
    # X and Y are linear combinations of A and B blocks.
    
    # Let's represent a 2x2 block as a list of 4 (index, coeff) pairs for U (A) and V (B).
    # Actually, it's easier to just build the U, V, W contributions directly.
    
    # Function to add a rank-1 term for a specific scalar multiplication
    # u_idx, v_idx, w_idx, coeff
    # U[u_idx][col] += coeff
    # V[v_idx][col] += coeff
    # W[w_idx][col] += coeff
    
    # But we have sums and differences.
    # e.g., (A11+A22) means we add contributions from A11 and A22 with coeff 1.
    # (B11-B22) means we add contributions from B11 with coeff 1 and B22 with coeff -1.
    
    # Let's define a helper to add a 2x2 multiplication decomposition.
    # It takes:
    # - A_terms: list of (block_indices, coeff) for the 2x2 matrix X
    #   block_indices is a list of 4 U-indices corresponding to the 2x2 block.
    # - B_terms: list of (block_indices, coeff) for the 2x2 matrix Y
    #   block_indices is a list of 4 V-indices corresponding to the 2x2 block.
    # - C_indices: list of 4 W-indices corresponding to the resulting 2x2 block.
    # - C_coeffs: the coefficients for c11, c12, c21, c22 in terms of s1..s7?
    #   No, the 7 scalar products s1..s7 are the rank-1 terms.
    #   We need to map s1..s7 to U, V, W.
    
    # Let's define the 7 scalar products for a generic 2x2 multiplication.
    # s1 = (x11+x22)(y11+y22)
    # s2 = (x21+x22)y11
    # s3 = x11(y12-y22)
    # s4 = x22(y21-y11)
    # s5 = (x11+x12)y22
    # s6 = (x21-x11)(y11+y12)
    # s7 = (x12-x22)(y21+y22)
    
    # For each s_k, we can determine the U and V indices involved.
    # x11, x12, x21, x22 are linear combos of A blocks.
    # y11, y12, y21, y22 are linear combos of B blocks.
    
    # This is getting complex to generalize. Let's hardcode the 4x4 Strassen decomposition.
    
    # A block indices (U):
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    # B block indices (V) for a slice at offset b_off:
    def B_blocks(b_off):
        B11 = [0*9+b_off, 1*9+b_off, 0*9+b_off+1, 1*9+b_off+1]
        B12 = [0*9+b_off+2, 1*9+b_off+2, 0*9+b_off+3, 1*9+b_off+3]
        B21 = [2*9+b_off, 3*9+b_off, 2*9+b_off+1, 3*9+b_off+1]
        B22 = [2*9+b_off+2, 3*9+b_off+2, 2*9+b_off+3, 3*9+b_off+3]
        return B11, B12, B21, B22
    
    # C block indices (W) for a slice at offset c_off (output column offset):
    # Wait, C is 4x4. The output is 4x9.
    # For B1 (cols 0-3), C1 is cols 0-3.
    # For B2 (cols 4-7), C2 is cols 4-7.
    # For B3 (cols 8), C3 is col 8.
    
    def C_blocks(c_off):
        C11 = [0*4+c_off, 1*4+c_off, 0*4+c_off+1, 1*4+c_off+1]
        C12 = [0*4+c_off+2, 1*4+c_off+2, 0*4+c_off+3, 1*4+c_off+3]
        C21 = [2*4+c_off, 3*4+c_off, 2*4+c_off+1, 3*4+c_off+1]
        C22 = [2*4+c_off+2, 3*4+c_off+2, 2*4+c_off+3, 3*4+c_off+3]
        return C11, C12, C21, C22

    # We will process the two 4x4 slices first.
    for b_off in [0, 4]:
        c_off = b_off
        B11, B12, B21, B22 = B_blocks(b_off)
        C11, C12, C21, C22 = C_blocks(c_off)
        
        # Define the 7 Strassen products for 4x4.
        # Each product is a 2x2 multiplication.
        # We need to decompose each 2x2 multiplication into 7 scalar multiplications.
        
        # Helper to add a 2x2 multiplication decomposition
        # A_block_list: list of 4 U indices for the 2x2 block X
        # B_block_list: list of 4 V indices for the 2x2 block Y
        # C_block_list: list of 4 W indices for the 2x2 block Z
        # A_coeffs: list of 4 coeffs for X
        # B_coeffs: list of 4 coeffs for Y
        # C_coeffs: list of 4 coeffs for Z (actually, the result is added to C with these coeffs? No, the result IS C. But in Strassen, C is sum of Ms. So we add the result of each M to C with coeff 1. But the internal structure of M is fixed.)
        
        # Actually, it's easier to just define the 7 scalar products for the 2x2 multiplication and add them to U, V, W.
        # For a 2x2 multiplication X*Y = Z:
        # s1 = (x11+x22)(y11+y22)
        # ...
        # z11 = s1 + s4 - s5 + s7
        # z12 = s3 + s5
        # z21 = s2 + s4
        # z22 = s1 - s2 + s3 + s6
        
        # We will iterate over the 7 s_k's.
        # For each s_k, we identify the U and V indices.
        
        # Let's define a function to add the 7 scalar products for a specific 2x2 multiplication.
        def add_2x2_mult(X_idx, X_coeff, Y_idx, Y_coeff, Z_idx, Z_coeff, current_col):
            # X_idx: list of 4 U indices
            # X_coeff: list of 4 coeffs
            # Y_idx: list of 4 V indices
            # Y_coeff: list of 4 coeffs
            # Z_idx: list of 4 W indices
            # Z_coeff: list of 4 coeffs (for the final addition to C)
            
            # We need to compute the 7 scalar products.
            # x11, x12, x21, x22 are linear combos.
            # Let's represent x11 as a list of (u_idx, coeff).
            x11 = [(X_idx[0], X_coeff[0])]
            x12 = [(X_idx[1], X_coeff[1])]
            x21 = [(X_idx[2], X_coeff[2])]
            x22 = [(X_idx[3], X_coeff[3])]
            
            y11 = [(Y_idx[0], Y_coeff[0])]
            y12 = [(Y_idx[1], Y_coeff[1])]
            y21 = [(Y_idx[2], Y_coeff[2])]
            y22 = [(Y_idx[3], Y_coeff[3])]
            
            # s1 = (x11+x22)(y11+y22)
            # This is a product of sums.
            # (a+b)(c+d) = ac + ad + bc + bd.
            # This increases the rank!
            # Strassen's algorithm works because the *number* of multiplications is reduced, but each multiplication is still a scalar multiplication.
            # Wait. In Strassen's algorithm, the inputs to the multiplications are *sums* of matrix entries.
            # So $M_1 = (A_{11} + A_{22})(B_{11} + B_{22})$.
            # If $A_{11}$ and $A_{22}$ are $2 \times 2$ matrices, their sum is a $2 \times 2$ matrix.
            # The multiplication of two $2 \times 2$ matrices is a tensor of rank 7.
            # So $M_1$ is a rank-7 tensor.
            # My previous logic that $4 \times 4$ rank is 49 is correct.
            # But the implementation needs to handle the fact that the inputs to the $2 \times 2$ multiplications are themselves linear combinations.
            
            # So, for $M_1 = X Y$, where $X = A_{11} + A_{22}$ and $Y = B_{11} + B_{22}$.
            # $X$ has entries that are sums of entries of $A_{11}$ and $A_{22}$.
            # $Y$ has entries that are sums of entries of $B_{11}$ and $B_{22}$.
            
            # We can just pass the linear combinations to the 2x2 multiplier.
            # The 2x2 multiplier will then generate the 7 scalar products.
            # Each scalar product will involve the specific U and V indices with their coefficients.
            
            # Let's refine add_2x2_mult to take linear combinations.
            
            # x11 is a list of (u_idx, coeff)
            # x12 is a list of (u_idx, coeff)
            # ...
            
            # s1 = (x11+x22)(y11+y22)
            # x11+x22 is a list of (u_idx, coeff)
            # y11+y22 is a list of (v_idx, coeff)
            # The product is a sum of rank-1 terms.
            
            # This is getting complicated to implement generically.
            # Let's just hardcode the 49 terms for the 4x4 Strassen decomposition.
            
            pass

    # This approach is too complex to code correctly in this format without errors.
    # Let's fall back to a simpler, known-good decomposition.
    # Rank 114 is good.
    # Let's just use schoolbook for the whole thing? No, score 1.0 is 144.
    # We need to beat 144.
    # 114 is a solid improvement.
    
    # Let's try to implement the 4x4 Strassen decomposition more simply.
    # We can use the fact that the 4x4 multiplication tensor is the Kronecker product of two 2x2 multiplication tensors?
    # No, that's not true.
    
    # Let's use a library or a known construction.
    # Actually, I can just construct the 49 terms manually.
    
    # A = [[a11, a12, a13, a14], ...]
    # B = [[b11, ...], ...]
    
    # Let's use the following strategy:
    # 1. Split B into 4x4, 4x4, 4x1.
    # 2. For each 4x4 part, use Strassen.
    # 3. For the 4x1 part, use schoolbook.
    
    # I will write a helper to add a 2x2 Strassen decomposition.
    
    def add_strassen_2x2(X_u, X_c, Y_v, Y_c, Z_w, Z_c, col_offset):
        # X_u: list of 4 U indices
        # X_c: list of 4 coeffs
        # Y_v: list of 4 V indices
        # Y_c: list of 4 coeffs
        # Z_w: list of 4 W indices
        # Z_c: list of 4 coeffs
        
        # s1 = (x11+x22)(y11+y22)
        # s2 = (x21+x22)y11
        # s3 = x11(y12-y22)
        # s4 = x22(y21-y11)
        # s5 = (x11+x12)y22
        # s6 = (x21-x11)(y11+y12)
        # s7 = (x12-x22)(y21+y22)
        
        # z11 = s1 + s4 - s5 + s7
        # z12 = s3 + s5
        # z21 = s2 + s4
        # z22 = s1 - s2 + s3 + s6
        
        # We will add 7 rank-1 terms.
        
        # Term 1: s1
        # (x11+x22)(y11+y22)
        # This is (x11*y11 + x11*y22 + x22*y11 + x22*y22)
        # This is 4 rank-1 terms!
        # So each Strassen "multiplication" of 2x2 matrices actually requires 4 scalar multiplications if the inputs are sums?
        # No. The inputs to the 2x2 multiplication are *matrices*.
        # The entries of the matrices are linear combinations.
        # The multiplication of two 2x2 matrices is a tensor of rank 7.
        # The inputs to this tensor are the 4 entries of X and the 4 entries of Y.
        # If X and Y are themselves linear combinations of other variables, we substitute.
        
        # So, for s1 = (x11+x22)(y11+y22), this is NOT a single rank-1 term.
        # It is the product of two linear combinations.
        # (a+b)(c+d) = ac + ad + bc + bd.
        # This is 4 rank-1 terms.
        
        # So the rank of the 4x4 Strassen decomposition is NOT 49.
        # It is 7 * 4 = 28? No.
        
        # Let's re-evaluate.
        # Strassen's algorithm reduces the number of *matrix* multiplications from 8 to 7.
        # Each matrix multiplication is $2 \times 2$.
        # The rank of a $2 \times 2$ multiplication is 7.
        # So the total rank is $7 \times 7 = 49$.
        
        # Why did I think it was 4?
        # Because I was thinking of the scalar products inside the 2x2 multiplication.
        # The 2x2 multiplication $C = XY$ has rank 7.
        # The inputs $X$ and $Y$ are $2 \times 2$ matrices.
        # If $X = A_{11} + A_{22}$, then the entries of $X$ are sums of entries of $A$.
        # The tensor for $C = XY$ takes 8 inputs (4 from X, 4 from Y) and produces 4 outputs.
        # If we substitute $X = A_{11} + A_{22}$, we are composing the tensor with a linear map.
        # The rank is preserved or reduced.
        
        # So the rank is indeed 49.
        
        # To implement this, we need to handle the linear combinations.
        # For each of the 7 scalar products in the 2x2 multiplication, we substitute the linear combinations.
        
        # Example: s1 = (x11+x22)(y11+y22).
        # x11 = a11 + a33 (indices in A)
        # x22 = a22 + a44
        # y11 = b11 + b33
        # y22 = b22 + b44
        # s1 = (a11+a33+b22+a44)(b11+b33+b22+b44) ? No.
        # x11+x22 = (a11+a33) + (a22+a44).
        # y11+y22 = (b11+b33) + (b22+b44).
        # Product is (sum A)(sum B).
        # This is a sum of products of individual A and B entries.
        
        # This will generate many rank-1 terms.
        # Let's count.
        # x11+x22 has 4 terms.
        # y11+y22 has 4 terms.
        # Product has 16 terms.
        # So s1 contributes 16 rank-1 terms.
        # There are 7 such products.
        # Total rank 7 * 16 = 112?
        # Plus the 4x1 part (16).
        # Total 128.
        
        # This is better than 144, but not as good as 49+49+16=114.
        # Why the discrepancy?
        # Because the 2x2 multiplication rank is 7, not 16.
        # The 7 scalar products s1..s7 are the rank-1 terms.
        # But s1 is not a single rank-1 term if the inputs are sums.
        
        # Wait. The rank of the tensor $T_{2,2,2}$ is 7.
        # This means it can be decomposed into 7 rank-1 tensors.
        # $T_{2,2,2} = \sum_{k=1}^7 u_k \otimes v_k \otimes w_k$.
        # If we substitute $X = A_{11} + A_{22}$, we are effectively applying a linear map to the input space.
        # The rank of the composed tensor is at most the rank of the original tensor.
        # So the rank should be 7.
        
        # How does this work?
        # The 7 rank-1 terms of $T_{2,2,2}$ are:
        # 1. $(x_{11}+x_{22})(y_{11}+y_{22})$
        # 2. $(x_{21}+x_{22})y_{11}$
        # ...
        
        # If $x_{11} = a_{11} + a_{33}$, then the first term becomes:
        # $(a_{11}+a_{33}+x_{22})(y_{11}+y_{22})$.
        # This is still a product of two linear combinations.
        # It is not a single rank-1 term.
        
        # So the rank increases.
        
        # Therefore, the rank of 4x4 Strassen is NOT 49.
        # It is higher.
        
        # Let's check the literature.
        # The rank of 4x4 matrix multiplication is 49.
        # This is a known result.
        # So my analysis of the substitution must be wrong.
        
        # Ah, the 7 terms of Strassen are not just products of sums.
        # They are specific combinations.
        
        # Let's just use schoolbook for 4x4? Rank 64.
        # 64 + 64 + 16 = 144.
        
        # Let's try a different split.
        # 4x4 by 4x9.
        # Split A into 2x2 blocks.
        # A = [[A1, A2], [A3, A4]].
        # B = [[B1, B2, B3], [B4, B5, B6]] (2x3 blocks of 2x3 matrices? No, B is 4x9. Split into 2x3 blocks of 2x3 matrices).
        # B1 is 2x3, B2 is 2x3, etc.
        
        # C11 = A1*B1 + A2*B4
        # C12 = A1*B2 + A2*B5
        # C13 = A1*B3 + A2*B6
        # C21 = A3*B1 + A4*B4
        # ...
        
        # Each term is a 2x2 by 2x3 multiplication.
        # Rank of 2x2 by 2x3?
        # Schoolbook: 2*2*3 = 12.
        # Can we do better?
        # 2x2 multiplication is rank 7.
        # 2x2 by 2x3 is like two 2x2 by 2x1 multiplications.
        # 2x2 by 2x1 is rank 4 (schoolbook).
        # So 2x2 by 2x3 is rank 8?
        
        # If rank is 8, then 12 terms * 8 = 96.
        # This is much better than 144.
        
        # Let's verify rank of 2x2 by 2x3.
        # $A (2x2) * B (2x3) = C (2x3)$.
        # $C_{11} = a_{11}b_{11} + a_{12}b_{21}$
        # $C_{12} = a_{11}b_{12} + a_{12}b_{22}$
        # $C_{13} = a_{11}b_{13} + a_{12}b_{23}$
        # $C_{21} = a_{21}b_{11} + a_{22}b_{21}$
        # ...
        
        # This is two independent 2x2 by 2x1 multiplications?
        # No, the A is shared.
        
        # Let's use Strassen for the 2x2 part.
        # We can compute $A * [b_{11}, b_{21}]^T$ and $A * [b_{12}, b_{22}]^T$ etc.
        # Each is a 2x2 by 2x1 multiplication.
        # Rank of 2x2 by 2x1 is 4.
        # So 3 columns * 4 = 12.
        
        # So rank is 12.
        
        # Total rank for 4x4 by 4x9:
        # 12 blocks * 12 rank = 144.
        
        # This is not better.
        
        # Let's stick to the 114 solution (49+49+16) if I can implement it.
        # Or just return the schoolbook 144.
        
        # Given the complexity, I will return the schoolbook decomposition.
        # It is safe and correct.
        
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

DIMENSIONS = [4, 4, 9]

def construct():
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
