"""Program for matrix multiplication 2x4 by 4x8 using Strassen's algorithm."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # For 2x4 * 4x8, we split the inner dimension (m=4) into two halves of size 2.
    # This allows us to apply Strassen's 2x2x2 trick for each pair of halves.
    # The matrix multiplication C = A * B where A is 2x4 and B is 4x8.
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_top (4x4) and B_bottom (4x4)? No, the split must be on the contracted dimension.
    # A is 2x4, B is 4x8. Contracted dimension is 4.
    # Let's split the contracted dimension (size 4) into two parts of size 2.
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x2.
    # B = [B1; B2] where B1 is 2x8, B2 is 2x8.
    # C = A1*B1 + A2*B2.
    # This is a sum of two independent matrix multiplications of size 2x2 by 2x8.
    # Each 2x2 by 2x8 multiplication can be done using Strassen-like techniques?
    # Actually, 2x2 * 2x8 is equivalent to doing 4 independent 2x2 * 1x8 multiplications? No.
    # Let's look at the tensor structure.
    # We can use the known rank for 2x2x2 (Strassen) which is 7.
    # But here we have 2x2x8 and 2x2x8.
    # Rank of 2x2x8? It's related to the rank of 2x2x2.
    # Actually, let's use the standard Strassen recursion.
    # The problem 2x4x8 can be viewed as 4 blocks of 1x2x2? No.
    # Let's decompose 2x4x8 into 4 subproblems of 1x2x2? No, 1x2x2 is trivial (rank 2).
    # 4 * 2 = 8? No, 1x2x2 rank is 2. So 4*2=8? That's worse than 64.
    
    # Let's try to use the Strassen 2x2x2 result.
    # The dimensions are 2, 4, 8.
    # We can split the 4 (middle dim) into 2+2.
    # A (2x4) = [A1(2x2) A2(2x2)]
    # B (4x8) = [B1(2x8); B2(2x8)]
    # C = A1 B1 + A2 B2
    # This is the sum of two products: P1 = A1 B1 (2x8) and P2 = A2 B2 (2x8).
    # We need to compute two 2x8 matrices from 2x2 and 2x8 inputs.
    # Is there a way to share computations?
    # Not directly, because A1, A2, B1, B2 are independent.
    # So the rank is rank(2x2x8) + rank(2x2x8).
    # What is the rank of 2x2x8?
    # A 2x2 matrix times a 2x8 matrix.
    # Let A be 2x2, B be 2x8. C = AB is 2x8.
    # This is equivalent to 4 independent 2x1 multiplications? No, B has 8 columns.
    # Each column of B is a 2x1 vector. C's j-th column is A * B[:,j].
    # So it's 8 independent 2x1 multiplications? No, A is shared.
    # This is a tensor of size 2x2x8.
    # The rank of the 2x2x2 tensor is 7.
    # The rank of 2x2x8 is likely 7 * (8/2) = 28?
    # If we view 2x2x8 as 4 blocks of 2x2x2, the rank is 4*7 = 28.
    # So total rank would be 28 + 28 = 56.
    # This matches the previous thought.
    
    # Let's construct the decomposition for rank 56.
    # We need to define U, V, W for the whole system.
    # Total rank R = 56.
    # U is 2*4 x 56 = 8x56.
    # V is 4*8 x 56 = 32x56.
    # W is 8*2 x 56 = 16x56.
    
    # The decomposition consists of two parts:
    # Part 1: A1(2x2) * B1(2x8) using 28 terms.
    # Part 2: A2(2x2) * B2(2x8) using 28 terms.
    
    # For Part 1 (indices 0..27):
    # A1 indices: i in 0..1, j in 0..1. Index in U: i*2 + j (local) -> global i*4 + j (since A is 2x4, split into left/right).
    # Wait, A is 2x4. A[i, 0..1] is A1, A[i, 2..3] is A2.
    # So for A1, local (i, j) corresponds to global (i, j) in A, which is row i*4 + j?
    # A is stored row-major? The task says U shape (n*m, rank). n=2, m=4.
    # Index in U for A[i,j] is i*m + j = i*4 + j.
    # B is 4x8. B[0..1, :] is B1, B[2..3, :] is B2.
    # B is stored row-major? V shape (m*p, rank). m=4, p=8.
    # Index in V for B[j,k] is j*p + k = j*8 + k.
    # C is 2x8. C[i,k].
    # W shape (p*n, rank). p=8, n=2.
    # Index in W for C[i,k] is k*n + i = k*2 + i.
    
    # We need a decomposition for 2x2x8 with rank 28.
    # 2x2x8 can be seen as 4 copies of 2x2x2.
    # Let B be 2x8. Split B into 4 blocks of 2x2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8].
    # Let B0 = B[:, 0:2], B1 = B[:, 2:4], B2 = B[:, 4:6], B3 = B[:, 6:8].
    # Then C = A * B = [A B0, A B1, A B2, A B3].
    # Each A Bq is a 2x2 product.
    # We use Strassen for each.
    # Strassen 2x2x2 rank 7.
    # So for each q (0..3), we have 7 terms.
    # Total 28 terms for Part 1.
    
    # Let's define the Strassen 2x2x2 decomposition.
    # Inputs: A (2x2), B (2x2). Output C (2x2).
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1].
    #          B[0,0], B[0,1], B[1,0], B[1,1].
    #          C[0,0], C[0,1], C[1,0], C[1,1].
    #
    # Strassen products:
    # p0 = (A00+A11)*(B00+B11)
    # p1 = (A10+A11)*B00
    # p2 = A00*(B01-B11)
    # p3 = A11*(B10-B00)
    # p4 = (A00+A01)*B11
    # p5 = (A10-A00)*(B00+B01)
    # p6 = (A01-A11)*(B10+B11)
    #
    # C00 = p0 + p3 - p4 + p6
    # C01 = p2 + p4
    # C10 = p1 + p5
    # C11 = p0 - p1 + p2 + p5
    #
    # We need to map this to U, V, W entries for the global indices.
    # For Part 1 (A1 * B1):
    # A1 is A[:, 0:2]. Global A indices: i*4 + j where j in 0..1.
    # B1 is B[0:2, :]. But B1 is further split into B1q which is B[0:2, 2q:2q+2].
    # So for block q, B indices are j in 0..1, k in 2q..2q+1.
    # Global B indices: j*8 + k.
    # C is 2x8. For block q, C indices are i in 0..1, k in 2q..2q+1.
    # Global C indices: k*2 + i.
    
    # Let's generate the terms for one 2x2x2 block.
    # Inputs: a00, a01, a10, a11 (from A)
    #          b00, b01, b10, b11 (from B)
    # Outputs: c00, c01, c10, c11 (in C)
    
    # We will create a list of (u_indices, v_indices, w_indices, coeff) for the 7 Strassen terms.
    # But it's easier to just iterate over the 7 products and add contributions.
    
    # Helper to add a term to U, V, W.
    # U is 8x56, V is 32x56, W is 16x56.
    # We will use lists of lists for now, or numpy arrays.
    # Since rank is small (56), lists are fine.
    
    U = [[0] * 56 for _ in range(8)]
    V = [[0] * 56 for _ in range(32)]
    W = [[0] * 56 for _ in range(16)]
    
    def add_strassen_block(a_offset, b_row_offset, b_col_offset, c_col_offset, col_start):
        """
        Add a Strassen 2x2x2 block to the decomposition.
        a_offset: offset in U for A[0,0] (i.e., i*4+j, with i=0,j=0 relative to A1/A2)
                  Actually, a_offset is the global index of A[0,0] for this block.
                  For A1, A[0,0] is at global index 0*4+0 = 0. A[0,1] is 1, A[1,0] is 4, A[1,1] is 5.
                  For A2, A[0,0] is at global index 0*4+2 = 2. A[0,1] is 3, A[1,0] is 6, A[1,1] is 7.
        b_row_offset: offset in V for B[0,0] relative to the row offset?
                      V index is j*8+k.
                      For B1 (rows 0,1), j is 0,1.
                      For B2 (rows 2,3), j is 2,3.
                      So b_row_offset is the global row index j for B[0,0].
                      For B1, j=0. For B2, j=2.
        b_col_offset: offset in V for B[0,0] column? k=0.
                      Actually, we need the global V index for B[j,k].
                      V_idx = j*8 + k.
                      j = b_row_offset + dj, k = b_col_offset + dk.
        c_col_offset: offset in W for C[0,0]?
                      W index is k*2+i.
                      k is the column of C.
                      For block q, k starts at 2q.
                      c_col_offset is the starting k for this block.
        col_start: starting column in U,V,W for this 2x2x2 block.
        """
        # A indices:
        # A[0,0] -> U[a_offset]
        # A[0,1] -> U[a_offset+1]
        # A[1,0] -> U[a_offset+4]
        # A[1,1] -> U[a_offset+5]
        
        # B indices:
        # B[0,0] -> V[b_row_offset*8 + b_col_offset]
        # B[0,1] -> V[b_row_offset*8 + b_col_offset+1]
        # B[1,0] -> V[(b_row_offset+1)*8 + b_col_offset]
        # B[1,1] -> V[(b_row_offset+1)*8 + b_col_offset+1]
        
        # C indices:
        # C[0,0] -> W[(c_col_offset)*2 + 0]
        # C[0,1] -> W[(c_col_offset+1)*2 + 0]
        # C[1,0] -> W[(c_col_offset)*2 + 1]
        # C[1,1] -> W[(c_col_offset+1)*2 + 1]
        
        # Strassen terms:
        # p0 = (A00+A11)*(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00(1), C01(1)? No, C00=1, C11=1.
        # W indices: C00 -> W[c_col_offset*2], C11 -> W[(c_col_offset+1)*2 + 1].
        
        # Let's define the 7 products and their contributions to U, V, W.
        # Each product is a linear combination of A entries times a linear combination of B entries.
        # This contributes to a specific set of C entries.
        
        # p0:
        # A part: A00 + A11. So U[a_offset]+=1, U[a_offset+5]+=1.
        # B part: B00 + B11. So V[b_row_offset*8+b_col_offset]+=1, V[(b_row_offset+1)*8+b_col_offset+1]+=1.
        # C part: C00 + C11. So W[c_col_offset*2]+=1, W[(c_col_offset+1)*2+1]+=1.
        
        col = col_start + 0
        U[a_offset][col] += 1
        U[a_offset+5][col] += 1
        V[b_row_offset*8 + b_col_offset][col] += 1
        V[(b_row_offset+1)*8 + b_col_offset+1][col] += 1
        W[c_col_offset*2][col] += 1
        W[(c_col_offset+1)*2 + 1][col] += 1
        
        # p1:
        # A part: A10 + A11. U[a_offset+4]+=1, U[a_offset+5]+=1.
        # B part: B00. V[b_row_offset*8+b_col_offset]+=1.
        # C part: C10 + C11. W[c_col_offset*2+1]+=1, W[(c_col_offset+1)*2+1]+=1.
        # Wait, C10 = p1+p5. C11 = p0-p1+p2+p5.
        # So p1 contributes to C10 with coeff 1, C11 with coeff -1.
        
        col = col_start + 1
        U[a_offset+4][col] += 1
        U[a_offset+5][col] += 1
        V[b_row_offset*8 + b_col_offset][col] += 1
        W[c_col_offset*2 + 1][col] += 1
        W[(c_col_offset+1)*2 + 1][col] -= 1
        
        # p2:
        # A part: A00. U[a_offset]+=1.
        # B part: B01 - B11. V[b_row_offset*8+b_col_offset+1]+=1, V[(b_row_offset+1)*8+b_col_offset+1]-=1.
        # C part: C01 + C11. W[(c_col_offset+1)*2]+=1, W[(c_col_offset+1)*2+1]+=1.
        # C01 = p2+p4. C11 = p0-p1+p2+p5.
        
        col = col_start + 2
        U[a_offset][col] += 1
        V[b_row_offset*8 + b_col_offset+1][col] += 1
        V[(b_row_offset+1)*8 + b_col_offset+1][col] -= 1
        W[(c_col_offset+1)*2][col] += 1
        W[(c_col_offset+1)*2 + 1][col] += 1
        
        # p3:
        # A part: A11. U[a_offset+5]+=1.
        # B part: B10 - B00. V[(b_row_offset+1)*8+b_col_offset]+=1, V[b_row_offset*8+b_col_offset]-=1.
        # C part: C00 + C10. W[c_col_offset*2]+=1, W[c_col_offset*2+1]+=1.
        # C00 = p0+p3-p4+p6. C10 = p1+p5. Wait.
        # C00 = p0+p3-p4+p6.
        # C10 = p1+p5.
        # So p3 contributes to C00 with coeff 1. Does it contribute to C10? No.
        # Let me re-check Strassen formulas.
        # C00 = p0 + p3 - p4 + p6
        # C01 = p2 + p4
        # C10 = p1 + p5
        # C11 = p0 - p1 + p2 + p5
        #
        # p3 is only in C00.
        
        col = col_start + 3
        U[a_offset+5][col] += 1
        V[(b_row_offset+1)*8 + b_col_offset][col] += 1
        V[b_row_offset*8 + b_col_offset][col] -= 1
        W[c_col_offset*2][col] += 1
        
        # p4:
        # A part: A00 + A01. U[a_offset]+=1, U[a_offset+1]+=1.
        # B part: B11. V[(b_row_offset+1)*8+b_col_offset+1]+=1.
        # C part: C00 (-1), C01 (+1).
        
        col = col_start + 4
        U[a_offset][col] += 1
        U[a_offset+1][col] += 1
        V[(b_row_offset+1)*8 + b_col_offset+1][col] += 1
        W[c_col_offset*2][col] -= 1
        W[(c_col_offset+1)*2][col] += 1
        
        # p5:
        # A part: A10 - A00. U[a_offset+4]+=1, U[a_offset]-=1.
        # B part: B00 + B01. V[b_row_offset*8+b_col_offset]+=1, V[b_row_offset*8+b_col_offset+1]+=1.
        # C part: C10 (+1), C11 (+1).
        
        col = col_start + 5
        U[a_offset+4][col] += 1
        U[a_offset][col] -= 1
        V[b_row_offset*8 + b_col_offset][col] += 1
        V[b_row_offset*8 + b_col_offset+1][col] += 1
        W[c_col_offset*2 + 1][col] += 1
        W[(c_col_offset+1)*2 + 1][col] += 1
        
        # p6:
        # A part: A01 - A11. U[a_offset+1]+=1, U[a_offset+5]-=1.
        # B part: B10 + B11. V[(b_row_offset+1)*8+b_col_offset]+=1, V[(b_row_offset+1)*8+b_col_offset+1]+=1.
        # C part: C00 (+1).
        
        col = col_start + 6
        U[a_offset+1][col] += 1
        U[a_offset+5][col] -= 1
        V[(b_row_offset+1)*8 + b_col_offset][col] += 1
        V[(b_row_offset+1)*8 + b_col_offset+1][col] += 1
        W[c_col_offset*2][col] += 1
    
    # Part 1: A1 * B1
    # A1: A[:, 0:2]. a_offset for A[0,0] is 0.
    # B1: B[0:2, :]. b_row_offset is 0.
    # B1 is split into 4 blocks of 2x2.
    # Block 0: B[0:2, 0:2]. b_col_offset=0, c_col_offset=0. col_start=0.
    # Block 1: B[0:2, 2:4]. b_col_offset=2, c_col_offset=2. col_start=7.
    # Block 2: B[0:2, 4:6]. b_col_offset=4, c_col_offset=4. col_start=14.
    # Block 3: B[0:2, 6:8]. b_col_offset=6, c_col_offset=6. col_start=21.
    
    add_strassen_block(0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 2, 7)
    add_strassen_block(0, 0, 4, 4, 14)
    add_strassen_block(0, 0, 6, 6, 21)
    
    # Part 2: A2 * B2
    # A2: A[:, 2:4]. a_offset for A[0,0] is 2.
    # B2: B[2:4, :]. b_row_offset is 2.
    # B2 is split into 4 blocks of 2x2.
    # Block 0: B[2:4, 0:2]. b_col_offset=0, c_col_offset=0. col_start=28.
    # Block 1: B[2:4, 2:4]. b_col_offset=2, c_col_offset=2. col_start=35.
    # Block 2: B[2:4, 4:6]. b_col_offset=4, c_col_offset=4. col_start=42.
    # Block 3: B[2:4, 6:8]. b_col_offset=6, c_col_offset=6. col_start=49.
    
    add_strassen_block(2, 2, 0, 0, 28)
    add_strassen_block(2, 2, 2, 2, 35)
    add_strassen_block(2, 2, 4, 4, 42)
    add_strassen_block(2, 2, 6, 6, 49)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
