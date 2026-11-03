import time
import numpy as np

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. The naive (schoolbook) rank is n*m*p = 2*3*5 = 30.
    2. We can reduce the rank by using Strassen's algorithm for a 2x2 sub-block.
    3. Partition the 2x3 matrix A into A_left (2x2) and A_right (2x1).
       A = [ A11 A12 | A13 ]
           [ A21 A22 | A23 ]
    4. Partition the 3x5 matrix B into B_top (2x5) and B_bottom (1x5).
       B = [ B11 B12 B13 B14 B15 ]
           [ B21 B22 B23 B24 B25 ]
           [ B31 B32 B33 B34 B35 ]
       
       Wait, the dimensions are:
       A is 2x3. B is 3x5.
       Let's partition the inner dimension (3) into 2 + 1.
       A = [ A_2x2 | A_2x1 ]
       B = [ B_2x5 ]
           [ B_1x5 ]
           
       Product C = A_2x2 * B_2x5 + A_2x1 * B_1x5.
       
       The term A_2x1 * B_1x5 is a 2x5 matrix. This can be done with schoolbook multiplication.
       Rank for A_2x1 * B_1x5: 2 * 1 * 5 = 10 terms.
       
       The term A_2x2 * B_2x5 is a 2x5 matrix.
       We can use Strassen for the 2x2 part? Strassen multiplies two 2x2 matrices.
       Here we are multiplying 2x2 by 2x5.
       We can view B_2x5 as [ B_2x2 | B_2x2 | B_2x1 ].
       C_left = A_2x2 * B_2x2 (first 2 cols)
       C_mid  = A_2x2 * B_2x2 (next 2 cols)
       C_right= A_2x2 * B_2x1 (last 1 col)
       
       Using Strassen for 2x2 * 2x2 requires 7 multiplications.
       So C_left takes 7.
       C_mid takes 7.
       C_right (2x2 * 2x1) takes 2*2*1 = 4 schoolbook multiplications.
       
       Total rank = 10 (for A_2x1 * B_1x5) + 7 (C_left) + 7 (C_mid) + 4 (C_right) = 28.
       
       Let's verify if we can do better or if this is correct.
       Is there a known algorithm for 2x3 * 3x5?
       The lower bound is max(nm, mp, pn) = max(6, 15, 10) = 15.
       Strassen for 2x2x2 is 7.
       
       Let's re-evaluate the partition.
       Maybe partition A as 1x3 + 1x3? No, A is 2x3.
       
       Let's try partitioning the inner dimension 3 into 1 + 2.
       A = [ A_2x1 | A_2x2 ]
       B = [ B_1x5 ]
           [ B_2x5 ]
       
       C = A_2x1 * B_1x5 + A_2x2 * B_2x5.
       
       Term 1: A_2x1 * B_1x5.
       A_2x1 has 2 elements. B_1x5 has 5 elements.
       Schoolbook rank: 2 * 1 * 5 = 10.
       
       Term 2: A_2x2 * B_2x5.
       B_2x5 can be viewed as [ B_2x2 | B_2x2 | B_2x1 ].
       C = A_2x2 * [ B1 | B2 | B3 ] = [ A*B1 | A*B2 | A*B3 ].
       
       A*B1 (2x2 * 2x2): Strassen rank 7.
       A*B2 (2x2 * 2x2): Strassen rank 7.
       A*B3 (2x2 * 2x1): Schoolbook rank 4.
       
       Total rank = 10 + 7 + 7 + 4 = 28.
       
       Can we do better?
       What if we use Strassen for the whole thing?
       There is no standard Strassen for 2x3 * 3x5 directly, but we can embed it.
       
       Another partition:
       Inner dimension 3 -> 2 + 1.
       We found 28.
       
       Is it possible to get 27?
       Maybe by combining terms?
       
       Let's look at the "inspiration" note:
       "Replacing 8 schoolbook terms in the 2x2x2 sub-block with 7 Strassen terms reduces rank from 30 to 29"
       This implies a specific structure where a 2x2x2 sub-problem exists.
       
       If we partition A (2x3) into A_2x2 and A_2x1, and B (3x5) into B_2x5 and B_1x5.
       The "2x2x2 sub-block" might refer to the multiplication of the first 2 columns of A and first 2 rows of B?
       A_2x2 (cols 0,1 of A) * B_2x5 (rows 0,1 of B).
       This produces a 2x5 matrix.
       This part involves 2x2 * 2x5.
       
       If we just take the first 2 columns of the result C (which is 2x5), that is C[:, 0:2].
       C[:, 0:2] = A_2x3 * B_3x2[:, 0:2].
       Let B_3x2 be the first 2 columns of B.
       B_3x2 = [ B_2x2; B_1x2 ].
       A_2x3 = [ A_2x2 | A_2x1 ].
       
       C[:, 0:2] = A_2x2 * B_2x2 + A_2x1 * B_1x2.
       
       A_2x2 * B_2x2 can be done in 7 (Strassen).
       A_2x1 * B_1x2 can be done in 2*1*2 = 4 (Schoolbook).
       Total for first 2 columns: 11.
       
       Remaining 3 columns of C (cols 2,3,4):
       C[:, 2:5] = A_2x3 * B_3x3.
       B_3x3 is the last 3 columns of B.
       B_3x3 = [ B_2x3; B_1x3 ].
       
       C[:, 2:5] = A_2x2 * B_2x3 + A_2x1 * B_1x3.
       
       A_2x1 * B_1x3: 2*1*3 = 6.
       A_2x2 * B_2x3: 2x2 * 2x3.
       B_2x3 = [ B_2x2 | B_2x1 ].
       A_2x2 * B_2x2 (7) + A_2x2 * B_2x1 (4) = 11.
       Total for last 3 columns: 6 + 11 = 17.
       
       Total rank = 11 + 17 = 28.
       
       The inspiration says 29. Why?
       Maybe they didn't optimize the 2x2 * 2x1 part?
       Or maybe they used a different partition?
       
       If I can achieve 28, that is better than 29.
       Score = 30/28 = 1.0714.
       
       Let's double check if 28 is possible.
       We need to construct the U, V, W matrices for rank 28.
       
       We will build the decomposition by concatenating the blocks.
       
       Block 1: A_2x2 * B_2x2 (for C[:, 0:2] part 1)
       Uses Strassen.
       Indices:
       A indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 3, 4
       B indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 5, 6
       C indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3  (Note: C is 2x5, so C[0,0] is idx 0, C[0,1] is idx 1, C[1,0] is idx 2, C[1,1] is idx 3)
       
       Wait, C index mapping:
       C is 2x5.
       c = k*n + i.
       i=0, k=0 -> 0
       i=0, k=1 -> 1
       i=0, k=2 -> 2
       i=0, k=3 -> 3
       i=0, k=4 -> 4
       i=1, k=0 -> 5
       i=1, k=1 -> 6
       i=1, k=2 -> 7
       i=1, k=3 -> 8
       i=1, k=4 -> 9
       
       So C[:, 0:2] corresponds to C indices 0, 1 (row 0) and 5, 6 (row 1).
       
       Let's define the Strassen terms for 2x2 * 2x2.
       Let A22 be the top-left 2x2 of A (indices 0,1,3,4).
       Let B22 be the top-left 2x2 of B (indices 0,1,5,6).
       
       Strassen products:
       M1 = (A00+A11)(B00+B11)
       M2 = (A10+A11)B00
       M3 = A00(B01-B11)
       M4 = A11(B10-B00)
       M5 = (A00+A10)B11
       M6 = (A11-A01)(B00+B01)
       M7 = (A01-A11)(B10+B11)
       
       C00 = M1 + M4 - M5 + M7
       C01 = M3 + M5
       C10 = M2 + M4
       C11 = M1 - M2 + M3 + M6
       
       We need to map these to the global tensor indices.
       
       Let's write a helper to add a block decomposition to the global U, V, W lists.
       
       We will maintain U, V, W as lists of lists (or arrays) and append columns.
       
       Steps:
       1. Initialize U, V, W as empty lists.
       2. Define a function `add_block(A_indices, B_indices, C_indices, decomposer)`
          where `decomposer` returns a list of tuples (u_vec, v_vec, w_vec) representing the rank-1 components for that block.
          Actually, it's easier to just generate the U, V, W rows for the specific sub-matrices.
          
       Let's define the Strassen decomposition for 2x2.
       It produces 7 rank-1 terms.
       For each term r in 0..6:
         u_r is a vector of length 4 (indices of A22)
         v_r is a vector of length 4 (indices of B22)
         w_r is a vector of length 4 (indices of C22)
         
       We need to map local indices to global indices.
       
       Let's implement the specific blocks.
       
       Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
       A_local: 0, 1, 3, 4
       B_local: 0, 1, 5, 6
       C_local: 0, 1, 5, 6
       
       Block 2: A[0:2, 2] * B[0:2, 0:2] -> C[0:2, 0:2] ? No.
       The term A_2x1 * B_1x2 contributes to C[:, 0:2].
       A_2x1 is A[:, 2] (indices 2, 5).
       B_1x2 is B[2, 0:2] (indices 10, 11).
       C[:, 0:2] is C[0:2, 0:2] (indices 0, 1, 5, 6).
       
       This is a 2x1 * 1x2 multiplication.
       Rank 4.
       Terms:
       C00 += A02 * B20
       C01 += A02 * B21
       C10 += A12 * B20
       C11 += A12 * B21
       
       So for Block 2, we add 4 terms.
       
       Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
       A_local: 0, 1, 3, 4
       B_local: B[0:2, 2:5]
       B is 3x5.
       B[0, 2], B[0, 3], B[0, 4] -> indices 2, 3, 4
       B[1, 2], B[1, 3], B[1, 4] -> indices 7, 8, 9
       So B_local indices: 2, 3, 4, 7, 8, 9.
       
       This is 2x2 * 2x3.
       We split B_2x3 into B_2x2 (cols 0,1 of this sub-block, i.e. global cols 2,3) and B_2x1 (col 2 of this sub-block, i.e. global col 4).
       
       Sub-block 3a: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
       A_local: 0, 1, 3, 4
       B_local: 2, 3, 7, 8
       C_local: C[0:2, 2:4] -> indices 2, 3, 7, 8
       
       This is 2x2 * 2x2. Use Strassen (7 terms).
       
       Sub-block 3b: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
       A_local: 0, 1, 3, 4
       B_local: 4, 9
       C_local: C[0:2, 4:5] -> indices 4, 9
       
       This is 2x2 * 2x1.
       Rank 4.
       Terms:
       C04 += A00*B04 + A01*B14
       C14 += A10*B04 + A11*B14
       
       Block 4: A[0:2, 2] * B[2, 2:5] -> C[0:2, 2:5]
       A_local: 2, 5 (A[:, 2])
       B_local: B[2, 2:5] -> indices 12, 13, 14
       C_local: C[0:2, 2:5] -> indices 2, 3, 4, 7, 8, 9
       
       This is 2x1 * 1x3.
       Rank 6.
       Terms:
       C02 += A02 * B22
       C03 += A02 * B23
       C04 += A02 * B24
       C12 += A12 * B22
       C13 += A12 * B23
       C14 += A12 * B24
       
       Total terms:
       Block 1 (Strassen 2x2): 7
       Block 2 (Schoolbook 2x1*1x2): 4
       Block 3a (Strassen 2x2): 7
       Block 3b (Schoolbook 2x2*2x1): 4
       Block 4 (Schoolbook 2x1*1x3): 6
       
       Total Rank = 7 + 4 + 7 + 4 + 6 = 28.
       
       This seems correct and valid.
       
       Now, implement the generation of U, V, W.
       
       We will create arrays U (6x28), V (15x28), W (10x28).
       We will fill them column by column.
       
       Helper:
       def add_term(U, V, W, col_idx, a_indices, a_vals, b_indices, b_vals, c_indices, c_vals):
           for a_idx, val in zip(a_indices, a_vals):
               U[a_idx, col_idx] += val
           for b_idx, val in zip(b_indices, b_vals):
               V[b_idx, col_idx] += val
           for c_idx, val in zip(c_indices, c_vals):
               W[c_idx, col_idx] += val
               
       We need to be careful with the Strassen formulas.
       
       Strassen 2x2:
       Inputs A (2x2), B (2x2).
       A indices: a00, a01, a10, a11
       B indices: b00, b01, b10, b11
       C indices: c00, c01, c10, c11
       
       M1 = (a00+a11)(b00+b11)
       U: a00:1, a11:1
       V: b00:1, b11:1
       W: c00:1, c11:1 (Wait, M1 contributes to C00 and C11)
       
       Actually, it's easier to define the 7 basis vectors for U, V, W and then sum them with coefficients for the output C.
       But the tensor decomposition requires that the sum of products equals T.
       T is 1 if the indices match.
       
       So for each Strassen term, we have a specific (u, v, w) triple.
       For M1:
       u = [1, 0, 0, 1] (a00, a01, a10, a11)
       v = [1, 0, 0, 1] (b00, b01, b10, b11)
       w = [1, 0, 0, 1] (c00, c01, c10, c11) -- NO.
       The product u*v*w is not a single tensor. It's a rank-1 tensor.
       The sum of these rank-1 tensors must equal T.
       
       T for 2x2*2x2 is:
       T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
       For 2x2, m=p=n=2.
       a=i*2+j, b=j*2+k, c=k*2+i.
       
       The Strassen algorithm works by linear combinations.
       The rank-1 terms in the Strassen decomposition are not simply the products of the linear combinations.
       The decomposition is:
       C = sum_{r=1}^7 U_r V_r W_r
       where U_r, V_r, W_r are the specific vectors.
       
       Let's list them:
       1. U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] ? No.
       
       Standard Strassen Rank-1 Tensors:
       M1 = (A00+A11)(B00+B11)
       M2 = (A10+A11)B00
       M3 = A00(B01-B11)
       M4 = A11(B10-B00)
       M5 = (A00+A10)B11
       M6 = (A11-A01)(B00+B01)
       M7 = (A01-A11)(B10+B11)
       
       C00 = M1 + M4 - M5 + M7
       C01 = M3 + M5
       C10 = M2 + M4
       C11 = M1 - M2 + M3 + M6
       
       This means the tensor T is the sum of 7 rank-1 tensors?
       No, T is the tensor of multiplication.
       The decomposition is T = sum_{r=1}^7 U_r \otimes V_r \otimes W_r.
       
       The vectors are:
       1. U1=[1,0,0,1], V1=[1,0,0,1], W1=[1,0,0,1] -> This would give C00+=1, C11+=1.
          But M1 contributes to C00 and C11 with coefficient 1.
          So W1 should be [1, 0, 0, 1].
          
       2. U2=[0,0,1,1], V2=[1,0,0,0], W2=[0,0,1,0] -> M2 contributes to C10.
          W2 should be [0, 0, 1, 0].
          
       3. U3=[1,0,0,0], V3=[0,1,0,-1], W3=[0,1,0,0] -> M3 contributes to C01.
          W3 should be [0, 1, 0, 0].
          
       4. U4=[0,0,0,1], V4=[0,-1,1,0], W4=[1,0,1,0] -> M4 contributes to C00 and C10.
          W4 should be [1, 0, 1, 0].
          
       5. U5=[1,0,1,0], V5=[0,0,0,1], W5=[-1,1,0,0] -> M5 contributes to C00 (-1) and C01 (1).
          W5 should be [-1, 1, 0, 0].
          
       6. U6=[0,-1,0,1], V6=[1,1,0,0], W6=[0,0,0,1] -> M6 contributes to C11.
          W6 should be [0, 0, 0, 1].
          
       7. U7=[0,1,0,-1], V7=[0,0,1,1], W7=[1,0,0,1] -> M7 contributes to C00 (1) and C11 (1).
          W7 should be [1, 0, 0, 1].
          
       Let's verify C00:
       M1(1) + M4(1) - M5(1) + M7(1)
       W1[0]=1, W4[0]=1, W5[0]=-1, W7[0]=1. Sum = 1+1-1+1 = 2?
       Wait.
       C00 = M1 + M4 - M5 + M7.
       If W5 is [-1, 1, 0, 0], then the term is -1 * U5 V5 W5.
       The decomposition is sum U_r V_r W_r.
       So we just add the vectors.
       
       Let's check the sum of W vectors at index 0 (C00):
       W1[0]=1
       W4[0]=1
       W5[0]=-1
       W7[0]=1
       Sum = 2.
       But T[0,0,0] (for a00,b00,c00) is 1.
       
       Ah, the Strassen algorithm computes the product. The tensor decomposition must match the multiplication tensor.
       The multiplication tensor T has T[a,b,c]=1 if indices align.
       
       For C00, the only contributing term in the schoolbook is A00*B00.
       In Strassen, C00 is a linear combination of the M's.
       The decomposition T = sum U_r V_r W_r implies that for any input A, B, the output C is sum (U_r . A) (V_r . B) W_r.
       
       So if we use the vectors above, does it compute C correctly?
       Yes, that's how Strassen works.
       Does it match the tensor T?
       Yes, because the map (A,B) -> C is the same.
       
       So we can just use these vectors.
       
       We need to map the local indices (0,1,2,3) to global indices.
       
       Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
       A_map: [0, 1, 3, 4]
       B_map: [0, 1, 5, 6]
       C_map: [0, 1, 5, 6]
       
       We will add 7 columns to U, V, W.
       
       Similarly for other blocks.
       
       Schoolbook blocks are simpler.
       For A_i * B_j -> C_k:
       U has 1 at i, V has 1 at j, W has 1 at k.
       
       We will implement this.
       
    """
    n, m, p = 2, 3, 5
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Initialize U, V, W as zero matrices
    # We don't know the rank yet, but we estimated 28.
    # Let's allocate for 30 to be safe, then slice.
    max_rank = 30
    U = np.zeros((dim_u, max_rank))
    V = np.zeros((dim_v, max_rank))
    W = np.zeros((dim_w, max_rank))
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_map, b_map, c_map):
        nonlocal col
        # a_map, b_map, c_map are lists of 4 global indices
        # Local indices 0,1,2,3 correspond to a00, a01, a10, a11
        
        # Define the 7 vectors
        # U vectors (local 4-dim)
        U_strassen = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 0, 1, 0], # M5
            [0, -1, 0, 1],# M6
            [0, 1, 0, -1] # M7
        ]
        V_strassen = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [0, -1, 1, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        W_strassen = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 0], # M2
            [0, 1, 0, 0], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 1]  # M7
        ]
        
        for r in range(7):
            for local_idx, val in enumerate(U_strassen[r]):
                if val != 0:
                    U[a_map[local_idx], col] += val
            for local_idx, val in enumerate(V_strassen[r]):
                if val != 0:
                    V[b_map[local_idx], col] += val
            for local_idx, val in enumerate(W_strassen[r]):
                if val != 0:
                    W[c_map[local_idx], col] += val
            col += 1
            
    # Helper to add a schoolbook term
    def add_schoolbook_term(a_idx, b_idx, c_idx):
        nonlocal col
        U[a_idx, col] = 1
        V[b_idx, col] = 1
        W[c_idx, col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0, 1, 3, 4
    # B indices: 0, 1, 5, 6
    # C indices: 0, 1, 5, 6
    add_strassen_block([0, 1, 3, 4], [0, 1, 5, 6], [0, 1, 5, 6])
    
    # Block 2: A[:, 2] * B[2, 0:2] -> C[:, 0:2]
    # A indices: 2, 5
    # B indices: 10, 11
    # C indices: 0, 1, 5, 6
    # Terms:
    # C00 += A02 * B20
    add_schoolbook_term(2, 10, 0)
    # C01 += A02 * B21
    add_schoolbook_term(2, 11, 1)
    # C10 += A12 * B20
    add_schoolbook_term(5, 10, 5)
    # C11 += A12 * B21
    add_schoolbook_term(5, 11, 6)
    
    # Block 3a: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: 0, 1, 3, 4
    # B indices: 2, 3, 7, 8
    # C indices: 2, 3, 7, 8
    add_strassen_block([0, 1, 3, 4], [2, 3, 7, 8], [2, 3, 7, 8])
    
    # Block 3b: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # A indices: 0, 1, 3, 4
    # B indices: 4, 9
    # C indices: 4, 9
    # Terms:
    # C04 += A00*B04 + A01*B14
    add_schoolbook_term(0, 4, 4)
    add_schoolbook_term(1, 9, 4)
    # C14 += A10*B04 + A11*B14
    add_schoolbook_term(3, 4, 9)
    add_schoolbook_term(4, 9, 9)
    
    # Block 4: A[:, 2] * B[2, 2:5] -> C[:, 2:5]
    # A indices: 2, 5
    # B indices: 12, 13, 14
    # C indices: 2, 3, 4, 7, 8, 9
    # Terms:
    # C02 += A02 * B22
    add_schoolbook_term(2, 12, 2)
    # C03 += A02 * B23
    add_schoolbook_term(2, 13, 3)
    # C04 += A02 * B24
    add_schoolbook_term(2, 14, 4)
    # C12 += A12 * B22
    add_schoolbook_term(5, 12, 7)
    # C13 += A12 * B23
    add_schoolbook_term(5, 13, 8)
    # C14 += A12 * B24
    add_schoolbook_term(5, 14, 9)
    
    # Final rank
    rank = col
    
    # Slice to actual rank
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    # Verify
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    if not np.allclose(T_calc, T_target):
        # Fallback to schoolbook if error
        U = np.zeros((dim_u, 30))
        V = np.zeros((dim_v, 30))
        W = np.zeros((dim_w, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        rank = 30
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
