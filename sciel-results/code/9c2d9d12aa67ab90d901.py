"""Matrix multiplication 2,4,8 with Strassen-based block decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We'll build the decomposition by treating the 2x4 * 4x8 as 2x2 * 2x8 blocks
    # Actually, let me think about this more carefully.
    # 
    # The naive approach has rank n*m*p = 2*4*8 = 64.
    # 
    # Let's use the fact that we can decompose the 4x4 intermediate into blocks.
    # A is 2x4, B is 4x8.
    # 
    # We can split A into [A1, A2] where A1, A2 are 2x2.
    # We can split B into [[B11, B12], [B21, B22]] where each is 2x4.
    # 
    # Then C = A1*B11 + A2*B21 (for first half of columns)
    #         A1*B12 + A2*B22 (for second half of columns)
    #
    # Each term is a 2x2 * 2x4 multiplication, which has naive rank 2*2*4 = 16.
    # Total naive rank would be 4*16 = 64 (same as before).
    #
    # But we can use Strassen for the 2x2 part. For 2x2 * 2x4:
    # We can view this as multiplying 2x2 by four 2x1 vectors.
    # 
    # Actually, let me think differently. Let's use the known result that
    # the tensor rank of 2x2 * 2x2 matrix multiplication is 7 (Strassen).
    # 
    # For 2x4 * 4x8, we can think of it as 8 independent 2x4 * 4x1 multiplications.
    # Each 2x4 * 4x1 can be decomposed using the linearity of the 4x1 part.
    #
    # Actually, the most straightforward approach: 
    # The multiplication tensor for (n,m,p) has rank at most n*m*p.
    # For (2,4,8), that's 64.
    #
    # Let's try to use Strassen's algorithm structure. 
    # Strassen multiplies 2x2 by 2x2 with 7 multiplications.
    # 
    # For 2x4 * 4x8:
    # Split into 2x2 * 2x8 blocks (two blocks horizontally in A, four in B)
    # 
    # A = [A_left, A_right]  (each 2x2)
    # B = [B_col1, B_col2, B_col3, B_col4] (each 4x2)
    #
    # C_left = A_left * B_col1 + A_right * B_col2  (2x2)
    # C_right = A_left * B_col3 + A_right * B_col4  (2x2)
    #
    # Wait, that's not right. Let me re-index.
    #
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x8: rows 0,1,2,3; cols 0,1,...,7
    # C is 2x8: rows 0,1; cols 0,1,...,7
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split j into two groups: j=0,1 and j=2,3
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # Let A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # Let B1 = B[0:2,:] (2x8), B2 = B[2:4,:] (2x8)
    #
    # C = A1*B1 + A2*B2
    #
    # Each A1*B1 is 2x2 * 2x8. This can be seen as 8 independent 2x2 * 2x1 products.
    # For each column k, we compute A1 * B1[:,k] which is 2x1.
    # 
    # The 2x2 * 2x1 multiplication: 
    # [a b] * [x] = [ax+bx]
    # [c d]   [y]   [cy+dy]
    # 
    # This has rank 4 naively (ax, bx, cy, dy).
    # But we can use Strassen-like tricks? Actually for 2x2*2x1, the rank is 4.
    #
    # Hmm, let me think about this differently.
    #
    # Actually, the key insight: we can use Strassen's algorithm for the 2x2 part.
    # For A1*B1 where A1 is 2x2 and B1 is 2x8:
    # We can write B1 as [B1a, B1b] where B1a, B1b are 2x4.
    # Then A1*B1 = [A1*B1a, A1*B1b]
    # 
    # For A1*B1a (2x2 * 2x4):
    # We can use Strassen on the 2x2 part, treating each column of B1a as a vector.
    # 
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # For 2x2 * 2x4, we can apply Strassen column-wise. But that doesn't save anything
    # because each column is independent.
    #
    # Let me try a different approach. The rank of the (2,4,8) matrix multiplication
    # tensor is known to be at most 64. Let's see if we can do better.
    #
    # Actually, I recall that for (2,2,2) the rank is 7 (Strassen).
    # For (2,4,8), we can think of it as 4 copies of (2,2,2) in the middle dimension
    # and 4 copies in the output dimension... no, that's not quite right.
    #
    # Let me just implement the naive rank-64 solution first, then try to optimize.
    #
    # Actually, let me try a block-based Strassen approach.
    # 
    # C = A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x8.
    # 
    # For A1*B1 (2x2 * 2x8):
    # We can use the fact that 2x2 * 2x8 = 2x2 * (2x4 + 2x4)
    # Let B1 = [B1_left, B1_right] where each is 2x4.
    # A1*B1 = [A1*B1_left, A1*B1_right]
    #
    # For A1*B1_left (2x2 * 2x4):
    # This is equivalent to 4 independent 2x2 * 2x1 multiplications.
    # Each has rank 4, so total rank 16.
    #
    # Total: 2 (for A1*B1 and A2*B2) * 2 (left/right) * 16 = 64. Same as naive.
    #
    # Let me try using Strassen properly.
    #
    # For 2x2 * 2x2, Strassen uses 7 multiplications.
    # For 2x2 * 2x8, we can view B as 4 blocks of 2x2.
    # A*B = [A*B0, A*B1, A*B2, A*B3] where each Bi is 2x2.
    # Each A*Bi uses 7 Strassen multiplications.
    # Total: 4*7 = 28 multiplications for A1*B1.
    # Similarly for A2*B2: 28.
    # Total: 56.
    #
    # That's better than 64! Let me implement this.
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen factors for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Output:
    # C00 = M1 + M4 - M6 + M7
    # C01 = M5 + M6
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # Strassen U factors (for A, 2x2):
    # M1: A00, A11
    # M2: A10, A11
    # M3: A00
    # M4: A11
    # M5: A00, A01
    # M6: A10, A00
    # M7: A01, A11
    
    # Strassen V factors (for B, 2x2):
    # M1: B00, B11
    # M2: B00
    # M3: B10, B11
    # M4: B01, B00
    # M5: B11
    # M6: B00, B10
    # M7: B01, B11
    
    # Strassen W factors (for C, 2x2):
    # M1: C00, C11
    # M2: C11
    # M3: C10, C11
    # M4: C00, C10
    # M5: C01, C11
    # M6: C00, C01
    # M7: C00
    
    # For 2x2 * 2x8, we have 4 blocks of 2x2 in the B dimension.
    # For each block, we use 7 Strassen multiplications.
    # Total per 2x2*2x8: 28.
    # For A1*B1 + A2*B2: 56 total.
    
    # Let's build this up.
    # 
    # A1 = A[:,0:2], A2 = A[:,2:4]
    # B1 = B[0:2,:], B2 = B[2:4,:]
    #
    # For A1*B1:
    # B1 has 8 columns, split into 4 blocks of 2: B1_0, B1_1, B1_2, B1_3
    # Each B1_i is 2x2.
    # For each block, use 7 Strassen mults.
    #
    # Similarly for A2*B2.
    
    col = 0
    for block in range(2):  # 0 for A1*B1, 1 for A2*B2
        A_offset = block * 2  # column offset in A
        B_offset = block * 2  # row offset in B
        
        for b_block in range(4):  # 4 blocks of 2 columns in B
            B_col_offset = b_block * 2  # column offset within B's 8 cols
            
            for s in range(7):  # 7 Strassen multiplications
                # Determine U, V, W factors based on Strassen term s
                # s=0: M1, s=1: M2, s=2: M3, s=3: M4, s=4: M5, s=5: M6, s=6: M7
                
                if s == 0:  # M1 = (A00+A11)(B00+B11)
                    U[A_offset + 0][col] = 1  # A00
                    U[A_offset + 2][col] = 1  # A11
                    V[B_offset + 0][col] = 1  # B00
                    V[B_offset + 2][col] = 1  # B11
                    W[(B_col_offset + 0) * n + 0][col] = 1  # C00
                    W[(B_col_offset + 1) * n + 1][col] = 1  # C11
                elif s == 1:  # M2 = (A10+A11)B00
                    U[A_offset + 1][col] = 1  # A10
                    U[A_offset + 2][col] = 1  # A11
                    V[B_offset + 0][col] = 1  # B00
                    W[(B_col_offset + 1) * n + 1][col] = -1  # C11
                elif s == 2:  # M3 = A00(B10-B11)
                    U[A_offset + 0][col] = 1  # A00
                    V[B_offset + 1][col] = 1  # B10
                    V[B_offset + 2][col] = -1  # B11
                    W[(B_col_offset + 0) * n + 1][col] = 1  # C10
                    W[(B_col_offset + 1) * n + 1][col] = 1  # C11
                elif s == 3:  # M4 = A11(B01-B00)
                    U[A_offset + 2][col] = 1  # A11
                    V[B_offset + 1][col] = 1  # B01
                    V[B_offset + 0][col] = -1  # B00
                    W[(B_col_offset + 0) * n + 0][col] = 1  # C00
                    W[(B_col_offset + 0) * n + 1][col] = 1  # C10
                elif s == 4:  # M5 = (A00+A01)B11
                    U[A_offset + 0][col] = 1  # A00
                    U[A_offset + 1][col] = 1  # A01
                    V[B_offset + 2][col] = 1  # B11
                    W[(B_col_offset + 1) * n + 0][col] = 1  # C01
                    W[(B_col_offset + 1) * n + 1][col] = 1  # C11
                elif s == 5:  # M6 = (A10-A00)(B00+B10)
                    U[A_offset + 1][col] = 1  # A10
                    U[A_offset + 0][col] = -1  # A00
                    V[B_offset + 0][col] = 1  # B00
                    V[B_offset + 1][col] = 1  # B10
                    W[(B_col_offset + 0) * n + 0][col] = -1  # C00
                    W[(B_col_offset + 1) * n + 0][col] = 1  # C01
                elif s == 6:  # M7 = (A01-A11)(B01+B11)
                    U[A_offset + 1][col] = 1  # A01
                    U[A_offset + 2][col] = -1  # A11
                    V[B_offset + 1][col] = 1  # B01
                    V[B_offset + 2][col] = 1  # B11
                    W[(B_col_offset + 0) * n + 0][col] = 1  # C00
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
