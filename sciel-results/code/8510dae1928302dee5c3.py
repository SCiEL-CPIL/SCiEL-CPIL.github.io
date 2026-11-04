"""Strassen-inspired structured decomposition for 3x4x6 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Construct a tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    Uses a structured ansatz inspired by Strassen's algorithm principles,
    adapted to the 3x4x6 case via block decomposition.
    
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank) = (12, rank)
      V has shape (m*p, rank) = (24, rank)
      W has shape (p*n, rank) = (18, rank)
    with the trace convention W[k*n + i, r].
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll build the decomposition using a hybrid approach:
    # - Use Strassen-like 7-term decompositions where possible for 2x2x2 blocks
    # - Fall back to schoolbook for remaining terms
    # 
    # Strategy: Split the 3x4x6 multiplication into blocks and apply
    # low-rank decompositions to each block pair.
    
    # The multiplication C = A * B where A is 3x4 and B is 4x6.
    # C has shape 3x6.
    #
    # We can view this as:
    #   Row 0 of C: A[0,:] * B  (1x4 times 4x6)
    #   Row 1 of C: A[1,:] * B  (1x4 times 4x6)
    #   Row 2 of C: A[2,:] * B  (1x4 times 4x6)
    #
    # Each row computation is a (1,4,6) tensor. The schoolbook rank is 24 per row,
    # total 72. We need to do better.
    #
    # Alternative block structure:
    # Split A into rows: A = [A0; A1; A2] where each Ai is 1x4
    # Split B into columns: B = [B0 B1 B2 B3 B4 B5] where each Bj is 4x1
    # Then C[i,j] = Ai * Bj
    #
    # A better approach: Use the fact that we can group computations.
    # Let's split the 4 middle dimension into 2+2:
    #   A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    #   B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6
    #   C = A_left * B_top + A_right * B_bottom
    #
    # Now each term is (3,2,6). Schoolbook rank of (3,2,6) is 36, total 72.
    #
    # For (3,2,6): Split further. The 3x2 part can use Strassen if we pad to 2x2,
    # but 3x2 is awkward. Let's try splitting the 6 into 2+2+2:
    #   C = A_left * B_top + A_right * B_bottom
    #   Each (3,2,6) = sum of three (3,2,2) terms
    #   Schoolbook rank of (3,2,2) is 12, so (3,2,6) is 36, total 72. No gain.
    #
    # Let's try a different split. Use the known result that matmul(3,4,6) 
    # has rank less than 72. The best known bounds suggest rank around 60-66.
    #
    # Approach: Use a combination of:
    # 1. For the 3x4 part, split 4 into 2+2, giving two (3,2,6) subproblems.
    # 2. For each (3,2,6), use a structured decomposition.
    #
    # For (3,2,6): Think of it as 3 rows, each doing a (1,2,6) multiplication.
    # (1,2,6): A = [a1, a2], B is 2x6. C = a1*B[0,:] + a2*B[1,:].
    # This is rank 12 (schoolbook) since each of the 6 outputs needs 2 terms.
    #
    # Can we do (1,2,6) in less than 12? The tensor for (1,2,6) is actually
    # just a sum of two rank-1 tensors for each output, so min rank is 12 
    # (each output j needs both a1*b[0,j] and a2*b[1,j]).
    #
    # So the bottleneck is elsewhere. Let's try a global approach.
    #
    # Key insight: The full tensor for (3,4,6) multiplication can potentially
    # be decomposed using shared intermediate computations.
    #
    # Let's use a concrete construction:
    # Split A (3x4) into:
    #   A = [[a00, a01, a02, a03],
    #        [a10, a11, a12, a13],
    #        [a20, a21, a22, a23]]
    # Split B (4x6) into 2x2 blocks along the middle dimension:
    #   B = [[B00, B01, B02, B03, B04, B05],
    #        [B10, B11, B12, B13, B14, B15],
    #        [B20, B21, B22, B23, B24, B25],
    #        [B30, B31, B32, B33, B34, B35]]
    #   B_top = B[0:2, :]  (2x6)
    #   B_bot = B[2:4, :]  (2x6)
    #
    # C = A_left * B_top + A_right * B_bot where A_left = A[:,0:2], A_right = A[:,2:4]
    #
    # For A_left * B_top: (3,2,6)
    # For A_right * B_bot: (3,2,6)
    #
    # Now, for each (3,2,6) problem, we can write it as:
    #   M is 3x2, N is 2x6, P = M*N is 3x6
    #   P[i,j] = M[i,0]*N[0,j] + M[i,1]*N[1,j]
    #
    # This is inherently rank 36 per (3,2,6) if done independently.
    # 
    # Total: 72. We need a smarter approach.
    #
    # Let's try using Strassen on a padded 4x4x4 problem or similar.
    # Actually, let's just build a valid decomposition with a moderate rank
    # and verify it works. I'll use a systematic construction.
    
    # I'll use the following approach:
    # Decompose the (3,4,6) matmul into independent (1,4,6) row computations,
    # but share factors across rows where possible.
    
    # Actually, let me just implement a correct schoolbook-based decomposition
    # but try to find a lower-rank one through a structured search.
    # Given time constraints, I'll implement a rank-66 construction.
    
    # Construction: Split middle dimension 4 into 2+2.
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # 
    # For A[:,0:2] * B[0:2,:]: This is (3,2,6).
    # We can decompose this using 12 rank-1 terms per row of A, total 36.
    # But we can do better by noting the 2x6 structure.
    
    # Let me just use a known valid construction.
    # I'll build the schoolbook rank-72 and then try to reduce.
    
    # For now, let's implement a valid rank-66 decomposition using
    # Strassen-like identities on 2x2 sub-blocks.
    
    # Split A (3x4) as:
    #   A11 = A[0:2, 0:2]  (2x2)
    #   A12 = A[0:2, 2:4]  (2x2)
    #   A21 = A[2:3, 0:2]  (1x2)
    #   A22 = A[2:3, 2:4]  (1x2)
    #
    # Split B (4x6) as:
    #   B11 = B[0:2, 0:2]  (2x2)
    #   B12 = B[0:2, 2:4]  (2x2)
    #   B13 = B[0:2, 4:6]  (2x2)
    #   B21 = B[2:4, 0:2]  (2x2)
    #   B22 = B[2:4, 2:4]  (2x2)
    #   B23 = B[2:4, 4:6]  (2x2)
    
    # C = A * B (3x6)
    # C[0:2, 0:2] = A11*B11 + A12*B21
    # C[0:2, 2:4] = A11*B12 + A12*B22
    # C[0:2, 4:6] = A11*B13 + A12*B23
    # C[2:3, 0:2] = A21*B11 + A22*B21
    # C[2:3, 2:4] = A21*B12 + A22*B22
    # C[2:3, 4:6] = A21*B13 + A22*B23
    
    # For the 2x2x2 multiplications (A11*B11, etc.), use Strassen (rank 7 each).
    # For the 1x2x2 multiplications (A21*B11, etc.), use schoolbook (rank 4 each).
    
    # Number of 2x2x2 terms: 6 (for C[0:2,:])
    # Each uses Strassen: rank 7, so 6*7 = 42
    # Number of 1x2x2 terms: 6 (for C[2:3,:])
    # Each uses schoolbook: rank 4, so 6*4 = 24
    # Total: 42 + 24 = 66
    
    # Now I need to construct U, V, W matrices.
    
    rank = 66
    U = [[0.0] * rank for _ in range(n * m)]  # (12, 66)
    V = [[0.0] * rank for _ in range(m * p)]  # (24, 66)
    W = [[0.0] * rank for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    # Helper: Add a Strassen decomposition for a 2x2x2 block
    def add_strassen(U_mat, V_mat, W_mat, col_offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for C[c_rows:c_rows+2, c_cols:c_cols+2] 
        = A[a_rows:a_rows+2, a_cols:a_cols+2] * B[b_rows:b_rows+2, b_cols:b_cols+2]
        
        a_rows, a_cols: starting row/col of A block (2x2)
        b_rows, b_cols: starting row/col of B block (2x2)  
        c_rows, c_cols: starting row/col of C block (2x2)
        
        Returns next column offset.
        """
        c = col_offset
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1
        U_mat[a_rows + 0][c] += 1   # A11
        U_mat[a_rows + 1][c] += 1   # A22
        V_mat[b_rows + 0][c] += 1   # B11
        V_mat[b_rows + 1][c] += 1   # B22
        W_mat[(0)*n + (c_rows + 0)][c] += 1  # C11 (k=0, i=c_rows+0)
        W_mat[(2)*n + (c_rows + 0)][c] += 1  # C22 (k=2, i=c_rows+0) -- wait, need to be careful
        # W index: W[k*n + i] where k is col of B (=row of C output), i is row of A (=col of C)
        # For C[c_rows+0, c_cols+0]: i = c_rows+0, k = c_cols+0
        # W[k*n + i] = W[(c_cols+0)*n + (c_rows+0)]
        
        # Let me redo this more carefully.
        # The tensor convention: C[i,j] = sum_r U[i*m + j, r] * V[j*p + k, r] * ... no
        # Actually the standard is: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, let me re-read the problem.
        
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # And shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
        
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 if the matmul has that index, i.e., C[i,k] = sum_j A[i,j]*B[j,k]
        
        # So for a rank-1 term with coefficient 1:
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        # contributes A[i,j]*B[j,k] to C[i,k].
        
        # For Strassen, each M term is a product of linear combinations.
        # M1 = (A11+A22)*(B11+B22) contributes to C11 and C22.
        
        # Let me define local indices. For a 2x2x2 block:
        # A local: rows 0,1; cols 0,1
        # B local: rows 0,1; cols 0,1
        # C local: rows 0,1; cols 0,1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # This contributes to C[0,0] and C[1,1].
        # We need to split this into rank-1 terms.
        # (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        # That's 4 rank-1 terms, not 1. Strassen works by having the same intermediate
        # product appear in multiple C entries.
        
        # In the tensor decomposition, M1 appears as a single rank-1 term in the
        # Strassen algorithm's sense, but in our U,V,W formulation, each "Strassen term"
        # actually corresponds to multiple rank-1 terms in the tensor.
        
        # I think I'm overcomplicating this. Let me just use schoolbook for now
        # with rank 72, which is guaranteed to work, and see if I can improve.
        
        return col_offset + 7  # placeholder
    
    # Given the complexity, let me just implement the schoolbook rank-72 solution
    # which is known to be correct, and return that. The score will be 1.0 (same as seed).
    # But the task asks for a stronger algorithm. Let me try harder.
    
    # Alternative: Use a known lower-rank decomposition.
    # The minimal rank of matmul(3,4,6) is not known exactly, but it's less than 72.
    # Let me implement a rank-66 construction using the block approach described above.
    
    # I'll implement it carefully now.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_1x2x2(col_offset, a_row, a_col_start, b_row_start, b_col_start, c_row, c_col_start):
        """
        Add schoolbook decomposition for C[c_row, c_col_start:c_col_start+2]
        = A[a_row, a_col_start:a_col_start+2] * B[b_row_start:b_row_start+2, c_col_start:c_col_start+2]
        
        This is a (1,2,2) matmul, rank 4.
        """
        c = col_offset
        # C[c_row, c_col_start] = A[a_row, a_col_start+0]*B[b_row_start+0, c_col_start+0] + A[a_row, a_col_start+1]*B[b_row_start+1, c_col_start+0]
        U[a_row * m + (a_col_start + 0)][c] += 1
        V[(a_col_start + 0) * p + (c_col_start + 0)][c] += 1
        W[(c_col_start + 0) * n + c_row][c] += 1
        c += 1
        
        U[a_row * m + (a_col_start + 1)][c] += 1
        V[(a_col_start + 1) * p + (c_col_start + 0)][c] += 1
        W[(c_col_start + 0) * n + c_row][c] += 1
        c += 1
        
        # C[c_row, c_col_start+1]
        U[a_row * m + (a_col_start + 0)][c] += 1
        V[(a_col_start + 0) * p + (c_col_start + 1)][c] += 1
        W[(c_col_start + 1) * n + c_row][c] += 1
        c += 1
        
        U[a_row * m + (a_col_start + 1)][c] += 1
        V[(a_col_start + 1) * p + (c_col_start + 1)][c] += 1
        W[(c_col_start + 1) * n + c_row][c] += 1
        c += 1
        
        return c
    
    # Process C[2, :] using 1x2x2 schoolbook terms
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    col = add_schoolbook_1x2x2(col, 2, 0, 0, 0, 2, 0)
    # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    col = add_schoolbook_1x2x2(col, 2, 0, 0, 2, 2, 2)
    # A[2, 0:2] * B[0:2, 4:6] -> C[2, 4:6]
    col = add_schoolbook_1x2x2(col, 2, 0, 0, 4, 2, 4)
    # A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    col = add_schoolbook_1x2x2(col, 2, 2, 2, 0, 2, 0)
    # A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    col = add_schoolbook_1x2x2(col, 2, 2, 2, 2, 2, 2)
    # A[2, 2:4] * B[2:4, 4:6] -> C[2, 4:6]
    col = add_schoolbook_1x2x2(col, 2, 2, 2, 4, 2, 4)
    
    # Now process C[0:2, :] using 2x2x2 Strassen terms
    # For each pair of 2x2 blocks, use Strassen with 7 rank-1 terms.
    
    def add_strassen_2x2x2(col_offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """
        Add Strassen decomposition for C[c_row_start:c_row_start+2, c_col_start:c_col_start+2]
        = A[a_row_start:a_row_start+2, a_col_start:a_col_start+2] * B[b_row_start:b_row_start+2, c_col_start:c_col_start+2]
        
        Uses 7 rank-1 terms (Strassen's 7 products).
        """
        c = col_offset
        # Local indices: A[i,j] for i,j in 0,1; B[i,j] for i,j in 0,1; C[i,j] for i,j in 0,1
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Each M is a product of two linear forms, which in tensor terms
        # corresponds to a single rank-1 term in the Strassen sense, but
        # we need to expand into the U,V,W format.
        
        # M1 = (A00+A11)(B00+B11)
        # This contributes to C00 with +1 and C11 with +1.
        # In U,V,W: we need U entries for the A part, V for B part, W for C part.
        # The A linear form: A00 + A11 -> U[a_row_start*m+a_col_start, c]=1, U[(a_row_start+1)*m+(a_col_start+1), c]=1
        # The B linear form: B00 + B11 -> V[(a_col_start)*p+(c_col_start), c]=1, V[(a_col_start+1)*p+(c_col_start+1), c]=1
        # The C contributions: C00 gets +1, C11 gets +1 -> W[(c_col_start)*n+(c_row_start), c]=1, W[(c_col_start+1)*n+(c_row_start+1), c]=1
        
        U[a_row_start * m + a_col_start][c] += 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][c] += 1
        V[a_col_start * p + c_col_start][c] += 1
        V[(a_col_start + 1) * p + (c_col_start + 1)][c] += 1
        W[c_col_start * n + c_row_start][c] += 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][c] += 1
        c += 1
        
        # M2 = (A10+A11)B00
        # Contributes to C10 with +1.
        U[(a_row_start + 1) * m + a_col_start][c] += 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][c] += 1
        V[a_col_start * p + c_col_start][c] += 1
        W[(c_col_start) * n + (c_row_start + 1)][c] += 1
        c += 1
        
        # M3 = A00(B01-B11)
        # Contributes to C01 with +1.
        U[a_row_start * m + a_col_start][c] += 1
        V[a_col_start * p + (c_col_start + 1)][c] += 1
        V[(a_col_start + 1) * p + (c_col_start + 1)][c] += -1
        W[(c_col_start + 1) * n + c_row_start][c] += 1
        c += 1
        
        # M4 = A11(B10-B00)
        # Contributes to C00 with +1, C10 with +1.
        U[(a_row_start + 1) * m + (a_col_start + 1)][c] += 1
        V[(a_col_start + 1) * p + c_col_start][c] += 1
        V[a_col_start * p + c_col_start][c] += -1
        W[c_col_start * n + c_row_start][c] += 1
        W[c_col_start * n + (c_row_start + 1)][c] += 1
        c += 1
        
        # M5 = (A00+A01)B11
        # Contributes to C00 with -1, C01 with +1.
        U[a_row_start * m + a_col_start][c] += 1
        U[a_row_start * m + (a_col_start + 1)][c] += 1
        V[(a_col_start + 1) * p + (c_col_start + 1)][c] += 1
        W[c_col_start * n + c_row_start][c] += -1
        W[(c_col_start + 1) * n + c_row_start][c] += 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes to C11 with +1.
        U[(a_row_start + 1) * m + a_col_start][c] += 1
        U[a_row_start * m + a_col_start][c] += -1
        V[a_col_start * p + c_col_start][c] += 1
        V[a_col_start * p + (c_col_start + 1)][c] += 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][c] += 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes to C00 with +1.
        U[a_row_start * m + (a_col_start + 1)][c] += 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][c] += -1
        V[(a_col_start + 1) * p + c_col_start][c] += 1
        V[(a_col_start + 1) * p + (c_col_start + 1)][c] += 1
        W[c_col_start * n + c_row_start][c] += 1
        c += 1
        
        return c
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    col = add_strassen_2x2x2(col, 0, 0, 0, 0, 0, 0)
    col = add_strassen_2x2x2(col, 0, 2, 2, 0, 0, 0)
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    col = add_strassen_2x2x2(col, 0, 0, 0, 2, 0, 2)
    col = add_strassen_2x2x2(col, 0, 2, 2, 2, 0, 2)
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    col = add_strassen_2x2x2(col, 0, 0, 0, 4, 0, 4)
    col = add_strassen_2x2x2(col, 0, 2, 2, 4, 0, 4)
    
    # col should be 24 + 6*7 = 24 + 42 = 66
    assert col == rank, f"Expected col={rank}, got {col}"
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to schoolbook if we're over time (shouldn't happen)
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_fb[i * m + j][idx] = 1
                    V_fb[j * p + k][idx] = 1
                    W_fb[k * n + i][idx] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
