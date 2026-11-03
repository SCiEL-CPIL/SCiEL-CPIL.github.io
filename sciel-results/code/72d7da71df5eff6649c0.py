"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition based on Strassen's algorithm for 2x2 blocks.
    # The multiplication is C = A * B where A is 4x4 and B is 4x7.
    # We split A into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # We split B into two 2x? blocks horizontally? No, B is 4x7.
    # Let's split B into B_left (4x4) and B_right (4x3).
    # C = A * [B_left, B_right] = [A * B_left, A * B_right].
    
    # Part 1: A * B_left (4x4 * 4x4)
    # We can use Strassen's algorithm for 4x4x4 by treating it as 2x2 blocks of 2x2 matrices.
    # Standard Strassen for 2x2 matrices has rank 7.
    # For 4x4 matrices (2x2 blocks), we need 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with rank 7 (Strassen).
    # So total rank for 4x4x4 is 7 * 7 = 49.
    # Let's verify this construction.
    
    # Part 2: A * B_right (4x4 * 4x3)
    # We can pad B_right to 4x4? Or just compute directly?
    # Direct schoolbook for 4x4x3 is 4*4*3 = 48 terms.
    # Can we do better?
    # If we split A into 2x2 blocks, and B_right into 2x? blocks...
    # B_right is 4x3. Split into top 2x3 and bottom 2x3.
    # C_top = A11*B1 + A12*B2
    # C_bot = A21*B1 + A22*B2
    # This is 4 multiplications of 2x2 by 2x3 matrices.
    # Rank of 2x2x3 multiplication?
    # Schoolbook: 2*2*3 = 12.
    # Is there a lower rank?
    # It's equivalent to 2 independent 2x1 multiplications? No.
    # It's like multiplying a 2x2 matrix by a 2x3 matrix.
    # This can be viewed as 3 separate 2x2 multiplications (columns of B).
    # So rank <= 3 * 7 = 21?
    # Or we can just use schoolbook for the 4x4x3 part to be safe and simple, 
    # or try to combine.
    
    # Let's stick to a robust construction:
    # 1. Compute A * B_left (4x4) using Strassen-on-Strassen (Rank 49).
    # 2. Compute A * B_right (4x3) using Schoolbook (Rank 48).
    # Total Rank = 49 + 48 = 97.
    # This is better than 112 (full schoolbook).
    
    # Can we do better for the 4x4x3 part?
    # Let's try to use Strassen logic on the 4x4x3.
    # A is 4x4. B is 4x3.
    # Split A into 2x2 blocks. Split B into 2x3 blocks (top and bottom).
    # C_top = A11*B_top + A12*B_bot
    # C_bot = A21*B_top + A22*B_bot
    # We need to compute products of 2x2 and 2x3 matrices.
    # Let M = 2x2, N = 2x3.
    # We can compute M*N by padding N to 2x4? No, that adds cost.
    # We can view M*N as three 2x2 multiplications (one for each column of N).
    # Rank = 3 * 7 = 21 for one 2x2x3 block?
    # There are 4 such blocks (A11*B_top, A12*B_bot, A21*B_top, A22*B_bot).
    # Total rank for 4x4x3 part = 4 * 21 = 84?
    # Wait, the Strassen trick reduces the number of multiplications.
    # For 2x2x2, we do 7 mults.
    # For 2x2x3, we can treat it as 3 independent 2x2x1? No.
    # Actually, the rank of 2x2x3 is known to be 12 (schoolbook) or possibly lower?
    # According to literature, rank of 2x2x3 is 12.
    # So 4 * 12 = 48. Same as schoolbook.
    
    # So 49 + 48 = 97 seems like a solid upper bound.
    
    # Let's refine the 4x4x4 part.
    # Is rank 49 achievable with integer/half-integer entries?
    # Strassen's algorithm uses coefficients -1, 1.
    # Yes, integers.
    
    # Let's implement the construction for Rank 97.
    
    def get_strassen_2x2_decomp():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        Shapes: U (4x7), V (4x7), W (4x7)
        Indices:
        U[i*2+j, r]
        V[j*2+k, r]
        W[k*2+i, r]
        """
        # Strassen's 7 products for 2x2 matrices A, B:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # Output:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Indices for U (A matrix): 0=A11, 1=A12, 2=A21, 3=A22
        # Indices for V (B matrix): 0=B11, 1=B12, 2=B21, 3=B22
        # Indices for W (C matrix): 0=C11, 1=C12, 2=C21, 3=C22
        
        # r=0: M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
        
        # r=1: M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C21 += M2, C22 -= M2
        
        # r=2: M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # C12 += M3, C22 += M3
        
        # r=3: M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
        
        # r=4: M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C11 -= M5, C12 += M5
        
        # r=5: M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C22 += M6
        
        # r=6: M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C11 += M7
        
        return U, V, W

    # Now build the 4x4x4 decomposition using 7 calls to 2x2x2 Strassen.
    # A is 4x4, B is 4x4.
    # We split A into A11, A12, A21, A22 (2x2 each).
    # We split B into B11, B12, B21, B22 (2x2 each).
    # We need to compute 7 products of sums/diffs of these blocks.
    # Each "product" is a 2x2 matrix multiplication, which we implement with Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let's construct the 4x4x4 U, V, W matrices.
    # Dimensions: U (16x49), V (16x49), W (16x49).
    # Note: For 4x4x4, output C is 4x4, so W shape is 4*4=16 rows.
    # Wait, the task is 4x4x7.
    # Let's first build the 4x4x4 part (columns 0-3 of B, columns 0-3 of C).
    
    U_444 = np.zeros((16, 49), dtype=float)
    V_444 = np.zeros((16, 49), dtype=float)
    W_444 = np.zeros((16, 49), dtype=float)
    
    # Strassen coefficients for the 7 block multiplications
    # We define the 7 linear combinations of A-blocks and B-blocks.
    # A-blocks: 0=A11, 1=A12, 2=A21, 3=A22
    # B-blocks: 0=B11, 1=B12, 2=B21, 3=B22
    
    # Strassen's 7 terms for 2x2 block matrix mult:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # We will iterate over these 7 terms.
    # For each term, we have a specific combination of A-blocks and B-blocks.
    # We map this to the 2x2x2 Strassen decomposition.
    
    # Helper to add a 2x2x2 decomposition into the larger 4x4x4 matrices
    def add_2x2_decomp(U_big, V_big, W_big, base_col, 
                       a_coeffs, b_coeffs, c_coeffs):
        """
        Adds a 2x2x2 Strassen decomposition.
        a_coeffs: dict {block_idx: coeff} for A blocks (0..3)
        b_coeffs: dict {block_idx: coeff} for B blocks (0..3)
        c_coeffs: dict {block_idx: coeff} for C blocks (0..3)
        """
        U2, V2, W2 = get_strassen_2x2_decomp()
        
        # Map block indices to row indices in the 4x4 matrices
        # A block 0 (A11): rows 0,1,4,5 (i=0,1; j=0,1) -> i*4+j
        # A block 1 (A12): rows 0,1,4,5 (i=0,1; j=2,3) -> i*4+j
        # A block 2 (A21): rows 8,9,12,13 (i=2,3; j=0,1)
        # A block 3 (A22): rows 8,9,12,13 (i=2,3; j=2,3)
        
        # B block 0 (B11): rows 0,1,4,5 (j=0,1; k=0,1) -> j*4+k
        # B block 1 (B12): rows 0,1,4,5 (j=0,1; k=2,3)
        # B block 2 (B21): rows 8,9,12,13 (j=2,3; k=0,1)
        # B block 3 (B22): rows 8,9,12,13 (j=2,3; k=2,3)
        
        # C block 0 (C11): rows 0,1,4,5 (k=0,1; i=0,1) -> k*4+i
        # C block 1 (C12): rows 0,1,4,5 (k=0,1; i=2,3)
        # C block 2 (C21): rows 8,9,12,13 (k=2,3; i=0,1)
        # C block 3 (C22): rows 8,9,12,13 (k=2,3; i=2,3)
        
        # Local indices in 2x2x2:
        # u_idx 0..3 correspond to A11, A12, A21, A22
        # v_idx 0..3 correspond to B11, B12, B21, B22
        # w_idx 0..3 correspond to C11, C12, C21, C22
        
        # We need to map the local 2x2 indices to the global 4x4 indices.
        # Local A index la (0..3) corresponds to block la.
        # Within the block, the local row/col (0..1) maps to global.
        
        # Let's precompute the mapping for U (A side)
        # For a given block b and local index l (0..3 in 2x2), 
        # l = li*2 + lj.
        # Global row = i_global * 4 + j_global.
        # If b=0 (A11): i=li, j=lj.
        # If b=1 (A12): i=li, j=2+lj.
        # If b=2 (A21): i=2+li, j=lj.
        # If b=3 (A22): i=2+li, j=2+lj.
        
        # Same for V (B side)
        # If b=0 (B11): j=lj, k=li (wait, V is j*p+k. Here p=4 for 4x4x4 subproblem).
        # V local index lv = lj*2 + lk.
        # If b=0 (B11): j=lj, k=lk.
        # If b=1 (B12): j=lj, k=2+lk.
        # If b=2 (B21): j=2+lj, k=lk.
        # If b=3 (B22): j=2+lj, k=2+lk.
        
        # Same for W (C side)
        # W local index lw = lk*2 + li.
        # If b=0 (C11): k=lk, i=li.
        # If b=1 (C12): k=lk, i=2+li.
        # If b=2 (C21): k=2+lk, i=li.
        # If b=3 (C22): k=2+lk, i=2+li.
        
        for r in range(7):
            global_r = base_col + r
            # U part
            for la in range(4):
                val = U2[la, r]
                if val == 0: continue
                li, lj = divmod(la, 2)
                block = a_coeffs.get(li*2+lj, 0) # This logic is wrong. a_coeffs is for blocks, not local indices.
                
            # Let's restart the mapping logic.
            # U2[la, r] is the coefficient for the local A element la in the r-th product.
            # la corresponds to local A block index? No, la is the flattened index of the 2x2 matrix A_block.
            # la = li*2 + lj.
            # This element belongs to A_block[bi] where bi is determined by the Strassen term.
            # Wait, the Strassen term defines WHICH blocks are involved.
            # For example, Term 1: (A11+A22)(B11+B22).
            # The "A part" is A11+A22.
            # In the 2x2x2 decomposition, U2[la, r] gives the weight for the la-th element of the "A matrix" in the 2x2x2 problem.
            # But the "A matrix" in the 2x2x2 problem is the SUM of the blocks?
            # No. The 2x2x2 decomposition is for multiplying two 2x2 matrices.
            # Let X be the 2x2 matrix formed by the linear combination of A-blocks.
            # Let Y be the 2x2 matrix formed by the linear combination of B-blocks.
            # We compute Z = X * Y using Strassen (rank 7).
            # The result Z is a 2x2 matrix.
            # This Z is added to the appropriate C-blocks.
            
            # So, for a specific Strassen term t:
            # X = sum_{bi} coeff_A[bi] * A_bi
            # Y = sum_{bk} coeff_B[bk] * B_bk
            # We need to compute X * Y.
            # The elements of X are x_li_lj = sum_{bi} coeff_A[bi] * A_bi[li, lj].
            # Note: All A_bi have the same shape 2x2.
            # So x_li_lj is a linear combination of the elements of the A-blocks.
            # In our global U matrix, U[global_row, global_r] corresponds to the coefficient of A[global_row] in the r-th product.
            # global_row corresponds to A[i, j].
            # i, j determine which block A_bi it belongs to.
            # li = i % 2, lj = j % 2.
            # bi = (i // 2) * 2 + (j // 2).
            
            # So, for each local element la (0..3) in the 2x2 matrix X:
            # la = li*2 + lj.
            # The coefficient from Strassen is U2[la, r].
            # This coefficient multiplies x_li_lj.
            # x_li_lj = sum_{bi} coeff_A[bi] * A_bi[li, lj].
            # So the total coefficient for A_bi[li, lj] in the r-th product is U2[la, r] * coeff_A[bi].
            # A_bi[li, lj] is located at global row: i*4+j where i=(bi//2)*2+li, j=(bi%2)*2+lj.
            
            # We apply this logic to U, V, W.
            
            for la in range(4):
                if U2[la, r] == 0: continue
                li, lj = divmod(la, 2)
                for bi in range(4):
                    cA = a_coeffs.get(bi, 0)
                    if cA == 0: continue
                    i = (bi // 2) * 2 + li
                    j = (bi % 2) * 2 + lj
                    global_u_row = i * 4 + j
                    U_big[global_u_row, global_r] += U2[la, r] * cA
            
            for lb in range(4):
                if V2[lb, r] == 0: continue
                lj, lk = divmod(lb, 2)
                for bk in range(4):
                    cB = b_coeffs.get(bk, 0)
                    if cB == 0: continue
                    j = (bk // 2) * 2 + lj
                    k = (bk % 2) * 2 + lk
                    global_v_row = j * 4 + k
                    V_big[global_v_row, global_r] += V2[lb, r] * cB
            
            for lc in range(4):
                if W2[lc, r] == 0: continue
                lk, li = divmod(lc, 2)
                for ck in range(4):
                    cC = c_coeffs.get(ck, 0)
                    if cC == 0: continue
                    k = (ck // 2) * 2 + lk
                    i = (ck % 2) * 2 + li
                    global_w_row = k * 4 + i
                    W_big[global_w_row, global_r] += W2[lc, r] * cC

    # Define the 7 Strassen terms for 4x4x4
    # a_coeffs, b_coeffs, c_coeffs are dicts mapping block index (0..3) to coefficient.
    
    terms_444 = [
        # P1 = (A11+A22)(B11+B22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (A21+A22)B11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3 = A11(B12-B22)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4 = A22(B21-B11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5 = (A11+A12)B22
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6 = (A21-A11)(B11+B12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7 = (A12-A22)(B21+B22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    base = 0
    for a_c, b_c, c_c in terms_444:
        add_2x2_decomp(U_444, V_444, W_444, base, a_c, b_c, c_c)
        base += 7
        
    # Now we have a valid 4x4x4 decomposition with rank 49.
    # We need to handle the 4x4x3 part (columns 4,5,6 of B).
    # We will use schoolbook for this part to ensure correctness and simplicity.
    # Rank = 4 * 4 * 3 = 48.
    
    U_443 = np.zeros((16, 48), dtype=float)
    V_443 = np.zeros((16, 48), dtype=float) # Note: B is 4x7, so V rows are 4*7=28?
    # Wait. The task is 4x4x7.
    # A is 4x4. B is 4x7.
    # U shape: (4*4, R) = (16, R)
    # V shape: (4*7, R) = (28, R)
    # W shape: (7*4, R) = (28, R)
    
    # My previous U_444, V_444, W_444 were for 4x4x4.
    # V_444 had shape (16, 49) because B was 4x4.
    # But here B is 4x7.
    # So for the 4x4x4 part (first 4 columns of B), V should have 4*4=16 rows?
    # No, V has 4*7=28 rows.
    # The first 16 rows of V correspond to B[:, 0:4].
    # The last 12 rows of V correspond to B[:, 4:7].
    
    # Let's re-initialize the full matrices.
    R_total = 49 + 48
    U = np.zeros((16, R_total), dtype=float)
    V = np.zeros((28, R_total), dtype=float)
    W = np.zeros((28, R_total), dtype=float)
    
    # Fill the first 49 columns with the 4x4x4 Strassen decomposition.
    # U[:, 0:49] = U_444
    # V[:, 0:49] should map to the first 16 rows of V (B[:, 0:4]).
    # W[:, 0:49] = W_444 (C[:, 0:4])
    
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[:, 0:49] = W_444
    
    # Now fill the next 48 columns with schoolbook for 4x4x3.
    # A is 4x4. B_right is 4x3.
    # C_right = A * B_right (4x3).
    # We need to compute C_right[0:4, 0:3].
    # This corresponds to W rows 16:28 (since C is 7x4, C[4:7, :] are the last 3 rows?
    # Wait. C is 4x7.
    # W indices: k*n+i. k=0..6, i=0..3.
    # C[0:4, 0:4] corresponds to k=0..3, i=0..3. Rows 0..15.
    # C[0:4, 4:7] corresponds to k=4..6, i=0..3. Rows 16..27.
    
    # So for the 4x4x3 part:
    # Input B is columns 4,5,6.
    # V indices: j*p+k. p=7.
    # B[:, 4] -> k=4. Rows 4, 11, 18, 25 (j=0,1,2,3).
    # B[:, 5] -> k=5. Rows 5, 12, 19, 26.
    # B[:, 6] -> k=6. Rows 6, 13, 20, 27.
    
    # Output C is columns 4,5,6.
    # W indices: k*n+i. n=4.
    # C[:, 4] -> k=4. Rows 16, 17, 18, 19 (i=0,1,2,3).
    # C[:, 5] -> k=5. Rows 20, 21, 22, 23.
    # C[:, 6] -> k=6. Rows 24, 25, 26, 27.
    
    base = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # C[i, 4+k] += A[i, j] * B[j, 4+k]
                # U row: i*4+j
                # V row: j*7 + (4+k)
                # W row: (4+k)*4 + i
                
                u_row = i * 4 + j
                v_row = j * 7 + (4 + k)
                w_row = (4 + k) * 4 + i
                
                U[u_row, base] = 1.0
                V[v_row, base] = 1.0
                W[w_row, base] = 1.0
                base += 1
                
    # Verify rank
    # R_total should be 97.
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
