"""Matrix multiplication 2x4 x 4x8 tensor decomposition - Strassen-based rank reduction."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Uses a hybrid approach:
    - For the first 2x2 block of A multiplied by the first 2x4 block of B, use Strassen (rank 7 vs 8)
    - For the remaining blocks, use schoolbook (rank 8 per 2x2 block)
    - Total rank: 7 + 8 + 8 + 8 = 31 (wait, that's not right)
    
    Actually, let me think more carefully.
    
    A is 2x4, B is 4x8. We can split B into two 4x4 blocks: B = [B1, B2]
    Then C = A * B = [A*B1, A*B2]
    
    Each A*B is 2x4 * 4x4 = 2x4 result.
    
    For a 2x4 * 4x4 multiplication, we can split A into two 2x2 blocks: A = [A1, A2]
    Then A*B = A1*B + A2*B
    
    Each 2x2 * 4x4 multiplication can be done with schoolbook rank 8.
    So A*B1 = A1*B1 + A2*B1 needs 8 + 8 = 16 multiplications.
    Similarly A*B2 = A1*B2 + A2*B2 needs 16 multiplications.
    Total: 32.
    
    But we can do better. For the 2x2 * 4x4 case, we can use the fact that we're adding two 2x2*4x4 products.
    
    Actually, let me use a different approach. The known result for 2x4 * 4x8:
    
    One can achieve rank 32 by treating it as two independent 2x4 * 4x4 problems.
    Each 2x4 * 4x4 can be done in rank 16 (schoolbook: 2*4*4=32, but we can do better).
    
    Actually for 2x4 * 4x4: 
    - Split into 2x2 * 4x4 + 2x2 * 4x4
    - Each 2x2 * 4x4 has schoolbook rank 8
    - Total schoolbook: 16
    
    Can we do better than 16 for 2x4 * 4x4?
    
    Let me think about it differently. The 2x4 * 4x4 multiplication:
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    etc.
    
    That's 16 outputs, each a sum of 4 products. Schoolbook uses 4*4=16 scalar mults for the 4x4 part... 
    wait no. 2x4 * 4x4: the middle dimension is 4. So schoolbook rank is 2*4*4 = 32? No wait.
    
    Let me reclarify. For n x m * m x p:
    - U has n*m rows
    - V has m*p rows  
    - W has p*n rows
    - Rank is at most n*m*p (schoolbook)
    
    For 2x4 * 4x4: n=2, m=4, p=4. Schoolbook rank = 2*4*4 = 32.
    
    Hmm, that's a lot. Let me reconsider.
    
    Actually, I think the Strassen insight from the inspiration is key. Let me look at this more carefully.
    
    For standard 2x2 * 2x2, Strassen reduces from 8 to 7.
    
    For our 2x4 * 4x8 problem, one approach is to use block Strassen.
    
    Let me try a concrete construction. I'll use the fact that we can group the computations.
    
    Here's my plan: I'll implement a decomposition that achieves rank 32 by splitting the 4x8 into two 4x4 blocks and using optimized 2x4*4x4 decompositions.
    
    Actually, let me just implement a solid decomposition and verify it. I'll start with a known good approach.
    
    For the 2x4 * 4x8 case, here's a construction:
    
    Split B into B_left (4x4) and B_right (4x4).
    C_left = A * B_left (2x4 result)
    C_right = A * B_right (2x4 result)
    
    For each 2x4 * 4x4 multiplication, I can use the following approach:
    Split A into A_top (1x4) and A_bottom (1x4) - wait, A is 2x4, so split into two 1x4 rows.
    
    C_top = A_row0 * B (1x4 result) - this is a 1x4 * 4x4 = 1x4 dot product
    C_bottom = A_row1 * B (1x4 result)
    
    Each 1x4 * 4x4 multiplication: 
    C[k] = sum_{j=0}^{3} A[j] * B[j,k]
    This needs 4 scalar multiplications per output, 4 outputs = 16 mults.
    But we can share: the 4 products A[j]*B[j,k] for fixed j and varying k are 4 products.
    For 4 values of j, that's 16 products total, which is just schoolbook for 1x4*4x4.
    
    So 2x4 * 4x4 = (1x4 * 4x4) + (1x4 * 4x4) = 16 + 16 = 32 schoolbook.
    
    Total for 2x4 * 4x8 = 32 + 32 = 64. That's just the original schoolbook.
    
    To get below 64, I need to use algebraic identities.
    
    Let me try a different splitting. What if I split along the middle dimension differently?
    
    A is 2x4, B is 4x8.
    
    Idea: Use the fact that we can compute linear combinations.
    
    Let me try to achieve rank 56 as mentioned in the inspiration.
    
    Here's one approach: For the 4x8 part, split into four 1x8 strips of B, or two 4x4 blocks.
    
    Let me try this construction:
    
    Split B into B1 (4x4, columns 0-3) and B2 (4x4, columns 4-7).
    
    For C = A*B1 (2x4):
    Use Strassen-like decomposition on the 2x2 blocks.
    
    Split A into A11 (2x2, cols 0-1) and A12 (2x2, cols 2-3).
    Split B1 into B11 (2x2, rows 0-1) and B12 (2x2, rows 2-3).
    
    C_left_top2 = A11 * B11 + A12 * B12  (2x2)
    C_left_bot2 = A11 * B12 + A12 * B11  (2x2)  -- wait, this isn't right
    
    Actually:
    C = [A11 A12] * [B11; B12] = A11*B11 + A12*B12
    
    That's a 2x2 * 2x2 + 2x2 * 2x2.
    
    Each 2x2*2x2 can be done in rank 7 (Strassen) or 8 (schoolbook).
    Using Strassen for both: 7 + 7 = 14 for the 2x4 * 4x2 part.
    Similarly for the other 2 columns of B1: another 14.
    Total for B1: 28.
    
    Similarly for B2: 28.
    Total: 56.
    
    That matches the inspiration! Let me implement this.
    
    Strassen for 2x2 * 2x2:
    Given X (2x2) and Y (2x2), compute Z = X*Y:
    M1 = (X00+X11)*(Y00+Y11)
    M2 = (X10+X11)*Y00
    M3 = X00*(Y01-Y11)
    M4 = X11*(Y10-Y00)
    M5 = (X00+X01)*Y11
    M6 = (X10-X00)*(Y00+Y01)
    M7 = (X01-X11)*(Y10+Y11)
    
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6
    
    Now I need to translate this into the U, V, W factor form.
    
    For each Strassen multiplication, I have 7 scalar products. Each product M_t = (linear_comb_of_X) * (linear_comb_of_Y).
    
    In the tensor decomposition:
    U maps from A entries to the "X" side of each product
    V maps from B entries to the "Y" side of each product  
    W maps from the 7 products to the output C entries
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We'll build U (2*4=8 rows), V (4*8=32 rows), W (8*2=16 rows)
    # Target rank: 56
    
    rank = 56
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    
    def strassen_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        """
        Add Strassen decomposition for a 2x2 * 2x2 multiplication.
        U_block: list of 4 indices (row-major: 00,01,10,11) in A
        V_block: list of 4 indices (row-major: 00,01,10,11) in B
        W_block: list of 4 indices (row-major: 00,01,10,11) in C output
        u_offset, v_offset, w_offset: base offsets into U, V, W
        Returns number of columns used.
        """
        global col
        # Strassen 7 products
        # M1 = (X00+X11)*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = X00*(Y01-Y11)
        # M4 = X11*(Y10-Y00)
        # M5 = (X00+X01)*Y11
        # M6 = (X10-X00)*(Y00+Y01)
        # M7 = (X01-X11)*(Y10+Y11)
        
        # Each product uses one column
        products = [
            # (U coefficients for X, V coefficients for Y, W coefficients for Z)
            # M1
            ([(0,1),(2,1)], [(0,1),(2,1)], [(0,1),(1,-1),(2,-1),(3,1)]),  # Z00+Z01-Z10+Z11... wait
        ]
        
        # Let me be more careful.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # So for each M_t, I need to know which Z entries it contributes to and with what coefficient.
        
        # M1: Z00 += 1, Z11 += 1
        # M2: Z10 += 1, Z11 += -1
        # M3: Z01 += 1, Z11 += 1
        # M4: Z00 += 1, Z10 += 1
        # M5: Z00 += -1, Z01 += 1
        # M6: Z11 += 1
        # M7: Z00 += 1
        
        strassen_data = [
            # (X_indices, X_coeffs, Y_indices, Y_coeffs, Z_indices, Z_coeffs)
            # M1 = (X00+X11)*(Y00+Y11)
            ([0, 2], [1, 1], [0, 2], [1, 1], [0, 3], [1, 1]),
            # M2 = (X10+X11)*Y00
            ([1, 2], [1, 1], [0], [1], [2, 3], [1, -1]),
            # M3 = X00*(Y01-Y11)
            ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
            # M4 = X11*(Y10-Y00)
            ([2], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
            # M5 = (X00+X01)*Y11
            ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
            # M6 = (X10-X00)*(Y00+Y01)
            ([1, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
            # M7 = (X01-X11)*(Y10+Y11)
            ([1, 2], [1, -1], [2, 3], [1, 1], [0], [1]),
        ]
        
        for x_idx, x_coeff, y_idx, y_coeff, z_idx, z_coeff in strassen_data:
            for xi, xc in zip(x_idx, x_coeff):
                U[u_offset + xi][col] = xc
            for yi, yc in zip(y_idx, y_coeff):
                V[v_offset + yi][col] = yc
            for zi, zc in zip(z_idx, z_coeff):
                W[w_offset + zi][col] = zc
            col += 1
        
        return 7
    
    # Process 4 blocks of 2x2 * 2x2
    # A is 2x4: rows 0-1, cols 0-3
    # B is 4x8: rows 0-3, cols 0-7
    # C is 2x8: rows 0-1, cols 0-7
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: row0col0=0, row0col1=1, row1col0=2, row1col1=3
    # B indices: row0col0=0, row0col1=1, row1col0=2, row1col1=3
    # C indices: row0col0=0, row0col1=1, row1col0=2, row1col1=3
    
    # Wait, I need to be careful about indexing.
    # U has 8 rows: A[i*m+j] = A[i*4+j]
    #   row 0: j=0,1,2,3 -> indices 0,1,2,3
    #   row 1: j=0,1,2,3 -> indices 4,5,6,7
    # V has 32 rows: B[j*p+k] = B[j*8+k]
    #   row 0: k=0..7 -> indices 0-7
    #   row 1: k=0..7 -> indices 8-15
    #   row 2: k=0..7 -> indices 16-23
    #   row 3: k=0..7 -> indices 24-31
    # W has 16 rows: C[k*n+i] = C[k*2+i]
    #   col 0: i=0,1 -> indices 0,1
    #   col 1: i=0,1 -> indices 2,3
    #   ...
    #   col 7: i=0,1 -> indices 14,15
    
    # For 2x2 * 2x2 blocks:
    # A block at (i0, j0): 2x2 submatrix
    #   Elements: A[i0,j0]=i0*4+j0, A[i0,j0+1]=i0*4+j0+1, A[i0+1,j0]=(i0+1)*4+j0, A[i0+1,j0+1]=(i0+1)*4+j0+1
    # B block at (j0, k0): 2x2 submatrix
    #   Elements: B[j0,k0]=j0*8+k0, B[j0,k0+1]=j0*8+k0+1, B[j0+1,k0]=(j0+1)*8+k0, B[j0+1,k0+1]=(j0+1)*8+k0+1
    # C block at (k0, i0): 2x2 submatrix
    #   Elements: C[k0,i0]=k0*2+i0, C[k0,i0+1]=k0*2+i0+1, C[k0+1,i0]=(k0+1)*2+i0, C[k0+1,i0+1]=(k0+1)*2+i0+1
    
    # But wait, my Strassen decomposition computes Z = X*Y where X is 2x2 and Y is 2x2.
    # The contribution to C is X*Y added to the appropriate 2x2 block of C.
    # 
    # For A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2], the result adds to C[k0:k0+2, i0:i0+2].
    
    # We have 4 such blocks:
    # (i0=0, j0=0) * B(0:2, 0:2) -> C(0:2, 0:2)
    # (i0=0, j0=2) * B(2:4, 0:2) -> C(0:2, 0:2)  -- same C block!
    # (i0=1, j0=0) * B(0:2, 0:2) -> C(0:2, 1:3)  -- wait, C has only 2 rows
    
    # I'm confusing myself. Let me restart the block decomposition.
    
    # A is 2x4. B is 4x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split j into 0-1 and 2-3.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #         = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    # First term: for fixed i, sum over j=0,1 of A[i,j]*B[j,k]
    # This is row i of A (first 2 elements) dotted with column k of B (first 2 rows).
    
    # Alternatively, think of it as:
    # C = A_left * B_top + A_right * B_bottom
    # where A_left is 2x2 (cols 0-1), A_right is 2x2 (cols 2-3)
    # B_top is 2x8 (rows 0-1), B_bottom is 2x8 (rows 2-3)
    
    # A_left * B_top is 2x2 * 2x8 = 2x8
    # A_right * B_bottom is 2x2 * 2x8 = 2x8
    
    # For 2x2 * 2x8, we can split B into four 2x2 blocks:
    # A_left * [B1 B2 B3 B4] = [A_left*B1, A_left*B2, A_left*B3, A_left*B4]
    # Each A_left*Bt is 2x2 * 2x2, which can be done in rank 7.
    # Total for A_left * B_top: 4 * 7 = 28
    
    # Similarly for A_right * B_bottom: 4 * 7 = 28
    
    # Total: 56. 
    
    # Now let me implement this properly.
    
    # Reset col
    col = 0
    
    # Define a helper to add a Strassen 2x2*2x2 block
    def add_strassen(a_base, b_base, c_base):
        """
        Add Strassen decomposition for A[a_base:a_base+2, :] * B[:, b_base:b_base+2] style.
        
        Actually, let me define it more carefully.
        
        We want to compute: contribution = X * Y where X is 2x2, Y is 2x2,
        and the result adds to a 2x2 block of C.
        
        X comes from a 2x2 submatrix of A.
        Y comes from a 2x2 submatrix of B.
        The result adds to a 2x2 submatrix of C.
        
        a_base: top-left index in A (row*4+col of top-left element)
        b_base: top-left index in B (row*8+col of top-left element)  
        c_base: top-left index in C (col*2+row of top-left element)
        
        The 2x2 submatrix of A starting at (i,j) has elements:
          A[i,j] = i*4+j = a_base
          A[i,j+1] = i*4+j+1 = a_base+1
          A[i+1,j] = (i+1)*4+j = a_base+4
          A[i+1,j+1] = (i+1)*4+j+1 = a_base+5
        
        The 2x2 submatrix of B starting at (j,k) has elements:
          B[j,k] = j*8+k = b_base
          B[j,k+1] = j*8+k+1 = b_base+1
          B[j+1,k] = (j+1)*8+k = b_base+8
          B[j+1,k+1] = (j+1)*8+k+1 = b_base+9
        
        The 2x2 submatrix of C starting at (k,i) has elements:
          C[k,i] = k*2+i = c_base
          C[k,i+1] = k*2+i+1 = c_base+1
          C[k+1,i] = (k+1)*2+i = c_base+2
          C[k+1,i+1] = (k+1)*2+i+1 = c_base+3
        """
        global col
        
        # Strassen products for X*Y -> Z
        # X indices in local 2x2: 0=X00, 1=X01, 2=X10, 3=X11
        # Y indices in local 2x2: 0=Y00, 1=Y01, 2=Y10, 3=Y11
        # Z indices in local 2x2: 0=Z00, 1=Z01, 2=Z10, 3=Z11
        
        # A local indices: 0->a_base, 1->a_base+1, 2->a_base+4, 3->a_base+5
        a_local = [a_base, a_base+1, a_base+4, a_base+5]
        # B local indices: 0->b_base, 1->b_base+1, 2->b_base+8, 3->b_base+9
        b_local = [b_base, b_base+1, b_base+8, b_base+9]
        # C local indices: 0->c_base, 1->c_base+1, 2->c_base+2, 3->c_base+3
        c_local = [c_base, c_base+1, c_base+2, c_base+3]
        
        strassen_products = [
            # (x_local_idx, x_coeff, y_local_idx, y_coeff, z_local_idx, z_coeff)
            # M1 = (X00+X11)*(Y00+Y11) -> Z00+=1, Z11+=1
            (([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1])),
            # M2 = (X10+X11)*Y00 -> Z10+=1, Z11+=-1
            (([2, 3], [1, 1]), ([0], [1]), ([2, 3], [1, -1])),
            # M3 = X00*(Y01-Y11) -> Z01+=1, Z11+=1
            (([0], [1]), ([1, 3], [1, -1]), ([1, 3], [1, 1])),
            # M4 = X11*(Y10-Y00) -> Z00+=1, Z10+=1
            (([3], [1]), ([2, 0], [1, -1]), ([0, 2], [1, 1])),
            # M5 = (X00+X01)*Y11 -> Z00+=-1, Z01+=1
            (([0, 1], [1, 1]), ([3], [1]), ([0, 1], [-1, 1])),
            # M6 = (X10-X00)*(Y00+Y01) -> Z11+=1
            (([2, 0], [1, -1]), ([0, 1], [1, 1]), ([3], [1])),
            # M7 = (X01-X11)*(Y10+Y11) -> Z00+=1
            (([1, 3], [1, -1]), ([2, 3], [1, 1]), ([0], [1])),
        ]
        
        for (x_idx, x_coeff), (y_idx, y_coeff), (z_idx, z_coeff) in strassen_products:
            for xi, xc in zip(x_idx, x_coeff):
                U[a_local[xi]][col] = xc
            for yi, yc in zip(y_idx, y_coeff):
                V[b_local[yi]][col] = yc
            for zi, zc in zip(z_idx, z_coeff):
                W[c_local[zi]][col] = zc
            col += 1
    
    # Now, the decomposition:
    # C = A_left * B_top + A_right * B_bottom
    # A_left = A[0:2, 0:2], A_right = A[0:2, 2:4]
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    
    # A_left * B_top: A_left is 2x2, B_top is 2x8
    # Split B_top into four 2x2 blocks: cols 0-1, 2-3, 4-5, 6-7
    # A_left * B_top[:, 0:2] -> adds to C[0:2, 0:2]
    # A_left * B_top[:, 2:4] -> adds to C[0:2, 2:4]
    # A_left * B_top[:, 4:6] -> adds to C[0:2, 4:6]
    # A_left * B_top[:, 6:8] -> adds to C[0:2, 6:8]
    
    # A_right * B_bottom: A_right is 2x2, B_bottom is 2x8
    # A_right * B_bottom[:, 0:2] -> adds to C[0:2, 0:2]
    # A_right * B_bottom[:, 2:4] -> adds to C[0:2, 2:4]
    # A_right * B_bottom[:, 4:6] -> adds to C[0:2, 4:6]
    # A_right * B_bottom[:, 6:8] -> adds to C[0:2, 6:8]
    
    # Wait, this is wrong. Let me reconsider.
    
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #         = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # First sum: A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   For i=0: A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   For i=1: A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #   This is [A[0,0], A[0,1]] * [B[0,k]; B[1,k]] for i=0
    #   And [A[1,0], A[1,1]] * [B[0,k]; B[1,k]] for i=1
    
    # So the first sum gives a 2x1 vector for each k, or a 2x8 matrix overall.
    # This can be seen as A_top_left (2x2) * B_top (2x8) where:
    #   A_top_left = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   B_top = [[B[0,0], ..., B[0,7]], [B[1,0], ..., B[1,7]]]
    
    # Similarly, second sum: A_bottom_left (2x2) * B_bottom (2x8) where:
    #   A_bottom_left = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    #   B_bottom = [[B[2,0], ..., B[2,7]], [B[3,0], ..., B[3,7]]]
    
    # Now, A_top_left * B_top is 2x2 * 2x8.
    # I can split B_top into four 2x2 blocks by columns.
    
    # For the t-th 2x2 block of B_top (cols 2t to 2t+1):
    #   It's a 2x2 matrix B_t.
    #   A_top_left * B_t is a 2x2 matrix that adds to C[0:2, 2t:2t+2].
    
    # So I need 4 Strassen decompositions for A_top_left * B_top.
    # And 4 Strassen decompositions for A_bottom_left * B_bottom.
    # Total: 8 Strassen decompositions, each of rank 7, total rank 56.
    
    # Let me define the blocks:
    
    # A_top_left: rows 0-1, cols 0-1 of A
    #   A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    #   So a_base for top-left is 0.
    
    # A_bottom_left: rows 0-1, cols 2-3 of A
    #   A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    #   So a_base for top-left is 2.
    
    # For B_top (rows 0-1 of B):
    #   Block 0: cols 0-1, b_base=0
    #   Block 1: cols 2-3, b_base=2
    #   Block 2: cols 4-5, b_base=4
    #   Block 3: cols 6-7, b_base=6
    
    # For B_bottom (rows 2-3 of B):
    #   Block 0: cols 0-1, b_base=16 (row 2, col 0)
    #   Block 1: cols 2-3, b_base=18
    #   Block 2: cols 4-5, b_base=20
    #   Block 3: cols 6-7, b_base=22
    
    # C blocks:
    #   For cols 0-1: c_base=0 (C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3)
    #   For cols 2-3: c_base=4
    #   For cols 4-5: c_base=8
    #   For cols 6-7: c_base=12
    
    # Process A_top_left * B_top blocks:
    for t in range(4):
        a_base = 0  # A_top_left
        b_base = 2 * t  # B_top block t
        c_base = 4 * t  # C block t
        add_strassen(a_base, b_base, c_base)
    
    # Process A_bottom_left * B_bottom blocks:
    for t in range(4):
        a_base = 2  # A_bottom_left
        b_base = 16 + 2 * t  # B_bottom block t
        c_base = 4 * t  # C block t
        add_strassen(a_base, b_base, c_base)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
