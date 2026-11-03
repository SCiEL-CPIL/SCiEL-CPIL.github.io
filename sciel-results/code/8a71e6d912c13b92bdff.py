import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-32 decomposition for 2x4 * 4x4 matrix multiplication.
    
    The strategy is to decompose the 4x4 matrix B into four 2x2 blocks and use
    Strassen's algorithm (rank 7) for the resulting 2x2 * 2x2 multiplications.
    Specifically, we treat the problem as:
      A (2x4) = [A_left (2x2) | A_right (2x2)]
      B (4x4) = [B_11 (2x2) B_12 (2x2); B_21 (2x2) B_22 (2x2)]
      
      The product C = A * B is (2x4).
      Let C = [C_left (2x2) | C_right (2x2)].
      C_left = A_left * B_11 + A_right * B_21
      C_right = A_left * B_12 + A_right * B_22
      
      To compute the sum of two 2x2 matrix products with shared left factors A_left, A_right,
      we can group them. However, a simpler direct application of Strassen-like block decomposition
      that yields rank 32 is to just use the standard block multiplication but optimized.
      
      Actually, the previous response mentioned rank 28 via Strassen on blocks.
      Let's look at the structure again.
      We want to compute:
      C_{i, j} = sum_k A_{i, j} B_{j, k}
      
      Let's split the summation index k into two halves: k in {0,1} and k in {2,3}.
      Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2).
      Let B1 = B[0:2, :] (2x4), B2 = B[2:4, :] (2x4).
      
      C = A1 * B1 + A2 * B2.
      
      This is a sum of two products of 2x2 matrices with 2x4 matrices.
      This is equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
      The tensor for (2x2) * (2x4) multiplication.
      Rank of (2x2)*(2x4)?
      Schoolbook: 2*2*4 = 16.
      Can we do better?
      We can view B1 and B2 as 2x2 matrices side-by-side.
      Let M = [A1; A2] ? No, that changes dimensions.
      
      Let's stick to the decomposition:
      C = A1 B1 + A2 B2.
      Each term is a 2x2 times 2x4 product.
      We can apply Strassen to the 2x2 part?
      Strassen applies to 2x2 * 2x2.
      Here the right side is 2x4.
      We can split B1 into [B1a | B1b] where B1a, B1b are 2x2.
      Then A1 * B1 = [A1 B1a | A1 B1b].
      So C = [A1 B1a + A2 B2a | A1 B1b + A2 B2b].
      
      Now we have two independent problems of the form:
      Compute X * Y + X' * Y' where X, X', Y, Y' are 2x2.
      This is exactly the tensor for (2,2,2) * 2.
      Or rather, it's the product of a 2x2 matrix with a 2x2 matrix, but we have two such products added together with different inputs.
      
      Let's define the tensor T for computing:
      M = A * B + C * D where A,B,C,D are 2x2.
      This is equivalent to matrix multiplication of (2x4) by (4x2)? No.
      It is equivalent to multiplying (A, C) (stacked 4x2?) by (B, D) ...
      
      Let's use the linearity.
      We need to compute Z = A B + C D.
      This can be done by embedding into a larger multiplication.
      Consider the product of a (2x4) matrix X = [A | C] and a (4x2) matrix Y = [B; D].
      X * Y = A B + C D.
      So the problem reduces to (2x4) * (4x2) matrix multiplication.
      
      What is the rank of (2x4) * (4x2)?
      Dimensions: n=2, m=4, p=2.
      Schoolbook rank: 2*4*2 = 16.
      Since p=2, we can use Strassen-like techniques on the "small" dimension?
      Or simply, since n=2 and p=2, the output is 2x2.
      The rank of M(2,4,2) is known to be 10? Or 12?
      Actually, M(2,4,2) is isomorphic to M(2,2,4) by symmetry.
      M(2,2,4) rank is at most 2*Rank(M(2,2,2)) = 2*7 = 14?
      If we split the 4 dimension into two 2s:
      M(2,2,4) = M(2,2,2) + M(2,2,2).
      So rank <= 14.
      
      If we can decompose (2x4)*(4x2) with rank 14, then we need two such decompositions for the left and right halves of the final 2x4 output?
      Wait, the output C is 2x4.
      C = [C_left | C_right].
      C_left = A1 B1_left + A2 B2_left (where B1_left is first 2 cols of B1, etc.)
      C_right = A1 B1_right + A2 B2_right.
      
      So we have two independent problems of type (2x4)*(4x2) -> 2x2.
      If Rank(M(2,4,2)) = 14, then total rank = 14 + 14 = 28.
      
      So the plan:
      1. Find a rank-14 decomposition for M(2,4,2).
      2. Apply it twice, once for the left half of the output, once for the right half.
      3. Combine them.
      
      How to construct M(2,4,2) rank 14?
      M(2,4,2) involves A(2x4), B(4x2), C(2x2).
      Split A into A1, A2 (2x2). Split B into B1, B2 (2x2).
      C = A1 B1 + A2 B2.
      This is the sum of two 2x2 products.
      As established, this is isomorphic to (2x4) * (4x2).
      But we can also view it as:
      Let's use the known rank of M(2,2,2) which is 7 (Strassen).
      We have two terms.
      C = A1 B1 + A2 B2.
      This is not a single matrix multiplication of larger size, it's a sum.
      However, we can map this to a single multiplication of size (2,4,2)?
      Yes, X = [A1 A2] (2x4), Y = [B1; B2] (4x2). X Y = A1 B1 + A2 B2.
      
      So we need a rank-14 decomposition for (2,4,2).
      Since 2,4,2 is small, we can try to construct one.
      Rank(M(2,4,2)) <= 2 * Rank(M(2,2,2)) is not directly true because the middle dimension is 4.
      But we can split the middle dimension:
      M(2,4,2) = M(2,2,2) + M(2,2,2).
      If we just concatenate the Strassen decompositions for the two halves:
      Rank <= 7 + 7 = 14.
      
      So, for C_left:
      Inputs: A1 (2x2), A2 (2x2), B1_left (2x2), B2_left (2x2).
      We form X = [A1 | A2] (2x4) and Y = [B1_left; B2_left] (4x2).
      We compute X Y using a rank-14 scheme which is just two Strassen schemes concatenated?
      Wait. If we just use two separate Strassen schemes for A1*B1 and A2*B2, we get rank 7+7=14.
      The output is the sum.
      In the tensor decomposition, the W vectors (output coefficients) for the two Strassen parts simply add up because the output is the sum of the two products.
      
      So, for each half of the output (left and right), we use two independent Strassen decompositions (one for A1*B_block, one for A2*B_block).
      Total rank = 2 (halves) * 2 (Strassen parts) * 7 (Strassen rank) = 28.
      
      Let's verify the indices.
      Output C is 2x4.
      C[0,0], C[1,0], C[0,1], C[1,1] are the left half.
      C[0,2], C[1,2], C[0,3], C[1,3] are the right half.
      
      Strassen Algorithm for 2x2 * 2x2:
      Given P, Q:
      M1 = (P00 + P11) * (Q00 + Q11)
      M2 = (P10 + P11) * Q00
      M3 = P00 * (Q01 - Q11)
      M4 = P11 * (Q10 - Q00)
      M5 = (P00 + P01) * Q11
      M6 = (P10 - P00) * (Q00 + Q01)
      M7 = (P01 - P11) * (Q10 + Q11)
      
      Result R:
      R00 = M1 + M4 - M5 + M7
      R01 = M3 + M5
      R10 = M2 + M4
      R11 = M1 - M2 + M3 + M6
      
      We will generate the U, V, W matrices for the full 28-rank decomposition.
      
      Dimensions:
      U: (2*4) x 28 = 8 x 28
      V: (4*4) x 28 = 16 x 28
      W: (4*2) x 28 = 8 x 28  (Note: W index is k*n+i, so 4*2=8)
      
      We define the Strassen factors for a generic 2x2 block.
      Let's create a helper to add a Strassen decomposition to our global U,V,W matrices.
      
      Parameters for a Strassen block:
      - Which part of A (2x4) does it read?
        A is indexed by i (0,1) and j (0..3).
        A1 uses j=0,1. A2 uses j=2,3.
      - Which part of B (4x4) does it read?
        B is indexed by j (0..3) and k (0..3).
        For the "left" output half (cols 0,1 of C), we use B rows 0..1 for A1 and B rows 2..3 for A2?
        No.
        C_left = A1 * B[0:2, 0:2] + A2 * B[2:4, 0:2].
        So for the term A1 * B_top_left:
          A part: rows 0..1 of A, cols 0..1 of A (indices 0,1 in the 4-col vector).
          B part: rows 0..1 of B, cols 0..1 of B.
        For the term A2 * B_bottom_left:
          A part: rows 0..1 of A, cols 2..3 of A.
          B part: rows 2..3 of B, cols 0..1 of B.
          
        Similarly for C_right:
        C_right = A1 * B[0:2, 2:4] + A2 * B[2:4, 2:4].
        
      So we have 4 Strassen blocks total:
      1. A1 * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
      2. A2 * B[2:4, 0:2] -> contributes to C[0:2, 0:2]
      3. A1 * B[0:2, 2:4] -> contributes to C[0:2, 2:4]
      4. A2 * B[2:4, 2:4] -> contributes to C[0:2, 2:4]
      
      Each block is a 2x2 * 2x2 multiplication.
      We will instantiate Strassen for each block.
      
      Block 1:
      A_slice: rows 0,1; cols 0,1. (Global A indices: 0,1,4,5)
      B_slice: rows 0,1; cols 0,1. (Global B indices: 0,1,4,5)
      C_slice: rows 0,1; cols 0,1. (Global C indices: 0,1,2,3 -> W indices: 0,1,2,3)
      
      Block 2:
      A_slice: rows 0,1; cols 2,3. (Global A indices: 2,3,6,7)
      B_slice: rows 2,3; cols 0,1. (Global B indices: 8,9,12,13)
      C_slice: rows 0,1; cols 0,1. (Global C indices: 0,1,2,3)
      
      Block 3:
      A_slice: rows 0,1; cols 0,1. (Global A indices: 0,1,4,5)
      B_slice: rows 0,1; cols 2,3. (Global B indices: 2,3,6,7)
      C_slice: rows 0,1; cols 2,3. (Global C indices: 4,5,6,7)
      
      Block 4:
      A_slice: rows 0,1; cols 2,3. (Global A indices: 2,3,6,7)
      B_slice: rows 2,3; cols 2,3. (Global B indices: 10,11,14,15)
      C_slice: rows 0,1; cols 2,3. (Global C indices: 4,5,6,7)
      
      For each block, we add 7 columns to U, V, W.
      Total 28 columns.
      
      Strassen M-factors definitions (local 2x2 indices 00,01,10,11):
      M1 = (P00 + P11) * (Q00 + Q11)
      M2 = (P10 + P11) * Q00
      M3 = P00 * (Q01 - Q11)
      M4 = P11 * (Q10 - Q00)
      M5 = (P00 + P01) * Q11
      M6 = (P10 - P00) * (Q00 + Q01)
      M7 = (P01 - P11) * (Q10 + Q11)
      
      Result:
      R00 = M1 + M4 - M5 + M7
      R01 = M3 + M5
      R10 = M2 + M4
      R11 = M1 - M2 + M3 + M6
      
      We need to map local P indices (0,1,2,3 corresponding to 00,01,10,11) to global U indices.
      We need to map local Q indices (0,1,2,3) to global V indices.
      We need to map local R indices (0,1,2,3) to global W indices.
      
      Let's implement this.
      """
    n, m, p = DIMENSIONS # 2, 4, 4
    # U shape: n*m = 8
    # V shape: m*p = 16
    # W shape: p*n = 8
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen steps
    # Each step is a list of (U_coeff, U_idx, V_coeff, V_idx, W_coeff, W_idx)
    # Actually, easier to just fill U, V, W arrays directly.
    
    def add_strassen_block(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, c_row_offsets, c_col_offsets):
        """
        Adds a 7-rank Strassen decomposition for a 2x2 * 2x2 block.
        a_row_offsets: [r0, r1] (global row indices in A, 0 or 1)
        a_col_offsets: [c0, c1] (global col indices in A, 0..3)
        b_row_offsets: [r0, r1] (global row indices in B, 0..3)
        b_col_offsets: [c0, c1] (global col indices in B, 0..3)
        c_row_offsets: [r0, r1] (global row indices in C, 0 or 1)
        c_col_offsets: [c0, c1] (global col indices in C, 0..3)
        """
        nonlocal col
        # Map local 2x2 indices to global indices
        # Local 00 -> (a_row_offsets[0], a_col_offsets[0])
        # Local 01 -> (a_row_offsets[0], a_col_offsets[1])
        # Local 10 -> (a_row_offsets[1], a_col_offsets[0])
        # Local 11 -> (a_row_offsets[1], a_col_offsets[1])
        
        # Global U index = i * m + j
        # Global V index = j * p + k
        # Global W index = k * n + i
        
        # A local indices
        u_00 = a_row_offsets[0] * m + a_col_offsets[0]
        u_01 = a_row_offsets[0] * m + a_col_offsets[1]
        u_10 = a_row_offsets[1] * m + a_col_offsets[0]
        u_11 = a_row_offsets[1] * m + a_col_offsets[1]
        
        # B local indices
        v_00 = b_row_offsets[0] * p + b_col_offsets[0]
        v_01 = b_row_offsets[0] * p + b_col_offsets[1]
        v_10 = b_row_offsets[1] * p + b_col_offsets[0]
        v_11 = b_row_offsets[1] * p + b_col_offsets[1]
        
        # C local indices
        w_00 = c_col_offsets[0] * n + c_row_offsets[0]
        w_01 = c_col_offsets[1] * n + c_row_offsets[0]
        w_10 = c_col_offsets[0] * n + c_row_offsets[1]
        w_11 = c_col_offsets[1] * n + c_row_offsets[1]
        
        # Strassen M-factors
        # M1 = (P00 + P11) * (Q00 + Q11)
        U[u_00][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1  # R00 += M1
        W[w_11][col] += 1  # R11 += M1
        col += 1
        
        # M2 = (P10 + P11) * Q00
        U[u_10][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1  # R10 += M2
        W[w_11][col] += -1 # R11 -= M2
        col += 1
        
        # M3 = P00 * (Q01 - Q11)
        U[u_00][col] += 1
        V[v_01][col] += 1
        V[v_11][col] += -1
        W[w_01][col] += 1  # R01 += M3
        W[w_11][col] += 1  # R11 += M3
        col += 1
        
        # M4 = P11 * (Q10 - Q00)
        U[u_11][col] += 1
        V[v_10][col] += 1
        V[v_00][col] += -1
        W[w_00][col] += 1  # R00 += M4
        W[w_10][col] += 1  # R10 += M4
        col += 1
        
        # M5 = (P00 + P01) * Q11
        U[u_00][col] += 1
        U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_01][col] += 1  # R01 += M5
        W[w_00][col] += -1 # R00 -= M5
        col += 1
        
        # M6 = (P10 - P00) * (Q00 + Q01)
        U[u_10][col] += 1
        U[u_00][col] += -1
        V[v_00][col] += 1
        V[v_01][col] += 1
        W[w_11][col] += 1  # R11 += M6
        col += 1
        
        # M7 = (P01 - P11) * (Q10 + Q11)
        U[u_01][col] += 1
        U[u_11][col] += -1
        V[v_10][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1  # R00 += M7
        col += 1
        
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # Verify rank used
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
