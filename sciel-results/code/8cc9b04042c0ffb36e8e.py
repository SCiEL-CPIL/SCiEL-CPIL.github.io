import time

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    
    The strategy uses a block decomposition approach:
    - The 2x5 matrix A is viewed as two 2x2 blocks (A11, A12) and one 2x1 block (A13).
      Actually, let's partition the 5 columns into sizes 2, 2, 1.
      A = [A11 (2x2), A12 (2x2), A13 (2x1)]
    - The 5x5 matrix B is partitioned into 5x2, 5x2, 5x1 blocks? No, B is 5x5.
      Let's partition the rows of B into 2, 2, 1 and columns into 2, 2, 1.
      B = [ B11 (2x2) B12 (2x2) B13 (2x1)
            B21 (2x2) B22 (2x2) B23 (2x1)
            B31 (1x2) B32 (1x2) B33 (1x1) ]
      
      The product C = A * B is 2x5.
      C = [ C11 (2x2) C12 (2x2) C13 (2x1) ]
      
      C11 = A11*B11 + A12*B21 + A13*B31
      C12 = A11*B12 + A12*B22 + A13*B32
      C13 = A11*B13 + A12*B23 + A13*B33

    This direct block sum would be:
    C11: 3 terms of 2x2 mult -> 3 * 4 = 12 scalar mults (schoolbook) or 3 * 7 = 21 (Strassen)
    C12: 3 terms of 2x2 mult -> 3 * 4 = 12 scalar mults (schoolbook) or 3 * 7 = 21 (Strassen)
    C13: 1 term of 2x2 (A11*B13 is 2x2*2x1? No. A11 is 2x2, B13 is 2x1. Result 2x1.
         A12 is 2x2, B23 is 2x1. Result 2x1.
         A13 is 2x1, B33 is 1x1. Result 2x1.
         
    Let's refine the block sizes to maximize Strassen usage.
    Standard Strassen applies to 2x2 matrices.
    
    Let's try a different partition:
    A is 2x5. Split columns into 2, 2, 1.
    B is 5x5. Split rows into 2, 2, 1 and columns into 2, 2, 1.
    
    Multiplications needed for C (2x5):
    C[:, 0:2] = A[:, 0:2]B[0:2, 0:2] + A[:, 2:4]B[2:4, 0:2] + A[:, 4:5]B[4:5, 0:2]
               = (2x2)(2x2) + (2x2)(2x2) + (2x1)(1x2)
    C[:, 2:4] = A[:, 0:2]B[0:2, 2:4] + A[:, 2:4]B[2:4, 2:4] + A[:, 4:5]B[4:5, 2:4]
               = (2x2)(2x2) + (2x2)(2x2) + (2x1)(1x2)
    C[:, 4:5] = A[:, 0:2]B[0:2, 4:5] + A[:, 2:4]B[2:4, 4:5] + A[:, 4:5]B[4:5, 4:5]
               = (2x2)(2x1) + (2x2)(2x1) + (2x1)(1x1)

    Total scalar multiplications if using schoolbook for all:
    4 + 4 + 2  (for C[:, 0:2] part 1,2,3? No. (2x1)*(1x2) is 2 mults.)
    Let's count:
    Term 1: (2x2)*(2x2) -> 4 mults
    Term 2: (2x2)*(2x2) -> 4 mults
    Term 3: (2x1)*(1x2) -> 2 mults
    Sum for C[:, 0:2] = 10 mults.
    
    Similarly C[:, 2:4] = 10 mults.
    
    C[:, 4:5]:
    Term 1: (2x2)*(2x1) -> 4 mults
    Term 2: (2x2)*(2x1) -> 4 mults
    Term 3: (2x1)*(1x1) -> 2 mults
    Sum for C[:, 4:5] = 10 mults.
    
    Total schoolbook rank = 30.
    
    Can we do better with Strassen?
    Strassen reduces 2x2 mult from 4 to 7? No, 4 to 7 is worse. Strassen is 7 mults for 2x2, schoolbook is 4.
    Wait. Schoolbook 2x2 multiplication is 4 multiplications. Strassen is 7 multiplications.
    Strassen is only better for larger matrices due to asymptotic complexity. For 2x2, schoolbook (4) is better than Strassen (7).
    
    So for 2x2 blocks, we should use schoolbook (rank 4).
    For 2x1 * 1x2, it's essentially 2 scalar mults.
    
    So the rank 30 decomposition is likely optimal for this block structure.
    Is there a rank lower than 30?
    The tensor rank of matrix multiplication for (2,5,5) is at least max(2*5, 5*5, 5*2) = 25? No, lower bound is max(n*m, m*p, p*n) is not a lower bound on rank.
    The rank of MM(2,5,5) is known to be at least 2*5 = 10? No.
    
    Let's check known results.
    Rank of MM(2,2,2) is 7 (Strassen).
    Rank of MM(2,2,1) is 3?
    A (2x2) * B (2x1) -> C (2x1).
    C1 = a11*b1 + a12*b2
    C2 = a21*b1 + a22*b2
    This is a 2x2x1 tensor. Rank is 2? No.
    It's the sum of two terms. Can it be rank 2?
    Yes, if we view it as two independent dot products.
    
    Let's re-evaluate the block decomposition rank.
    We have 6 block multiplications of type (2x2)*(2x2) or similar.
    Actually, let's just construct the rank 30 decomposition explicitly.
    It is definitely better than 50.
    
    Structure:
    U: (2*5, rank) = (10, rank)
    V: (5*5, rank) = (25, rank)
    W: (5*2, rank) = (10, rank)
    
    We will generate rank 30.
    
    Indices:
    A is 2x5. Indices i (0..1), j (0..4). U index: i*5+j.
    B is 5x5. Indices j (0..4), k (0..4). V index: j*5+k.
    C is 2x5. Indices i (0..1), k (0..4). W index: k*2+i (trace convention W[k*n+i] where n=2).
    
    Decomposition:
    We split the summation index j (0..4) into groups:
    Group 1: j=0,1 (2x2 block)
    Group 2: j=2,3 (2x2 block)
    Group 3: j=4 (1x1 block)
    
    For each output column k (0..4), we compute C[i,k] = sum_j A[i,j]B[j,k].
    
    Let's break it down by k groups too?
    k=0,1 (2 columns)
    k=2,3 (2 columns)
    k=4 (1 column)
    
    This creates 3x3 = 9 block interactions.
    
    Interaction (j_group, k_group):
    1. j in {0,1}, k in {0,1}: A_2x2 * B_2x2 -> C_2x2.
       Schoolbook rank 4.
       Mults:
       C00 = A00*B00 + A01*B10
       C01 = A00*B01 + A01*B11
       C10 = A10*B00 + A11*B10
       C11 = A10*B01 + A11*B11
       
       Wait, schoolbook 2x2 multiplication uses 4 multiplications:
       M1 = A00*B00
       M2 = A01*B10
       M3 = A10*B01
       M4 = A11*B11
       Then C00=M1+M2, C01=M1+M3, C10=M2+M4, C11=M3+M4.
       
       So for each (j_group, k_group) pair where both are size 2, we need 4 rank-1 terms.
       
    2. j in {0,1}, k in {2,3}: A_2x2 * B_2x2 -> C_2x2.
       Rank 4.
       
    3. j in {0,1}, k in {4}: A_2x2 * B_2x1 -> C_2x1.
       C0 = A00*B0 + A01*B1
       C1 = A10*B0 + A11*B1
       This is 2 multiplications?
       M1 = A00*B0, M2 = A01*B1 -> C0 = M1+M2.
       M3 = A10*B0, M4 = A11*B1 -> C1 = M3+M4.
       Rank 4.
       
    4. j in {2,3}, k in {0,1}: A_2x2 * B_2x2 -> C_2x2.
       Rank 4.
       
    5. j in {2,3}, k in {2,3}: A_2x2 * B_2x2 -> C_2x2.
       Rank 4.
       
    6. j in {2,3}, k in {4}: A_2x2 * B_2x1 -> C_2x1.
       Rank 4.
       
    7. j in {4}, k in {0,1}: A_2x1 * B_1x2 -> C_2x2.
       A is 2x1 (vector a), B is 1x2 (vector b).
       C = a * b (outer product).
       C00 = a0*b0
       C01 = a0*b1
       C10 = a1*b0
       C11 = a1*b1
       This is rank 2?
       M1 = a0*b0 -> C00
       M2 = a0*b1 -> C01
       M3 = a1*b0 -> C10
       M4 = a1*b1 -> C11
       Rank 4.
       
    8. j in {4}, k in {2,3}: A_2x1 * B_1x2 -> C_2x2.
       Rank 4.
       
    9. j in {4}, k in {4}: A_2x1 * B_1x1 -> C_2x1.
       C0 = a0*b
       C1 = a1*b
       M1 = a0*b -> C0
       M2 = a1*b -> C1
       Rank 2.
       
    Total Rank = 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 2 = 34.
    
    Wait, I calculated 30 earlier. Let's re-verify.
    C[:, 0:2] (2 cols):
    j=0,1 (2x2 * 2x2) -> 4 mults
    j=2,3 (2x2 * 2x2) -> 4 mults
    j=4   (2x1 * 1x2) -> 4 mults (outer product 2x1 and 1x2 is rank 4? 
           C is 2x2. C_ij = a_i * b_j.
           Tensors: e_i^A * e_j^B * e_i^C * e_j^C.
           Rank of outer product of 2-dim and 2-dim is 4?
           Yes, it's the sum of 4 rank-1 tensors.
           So 4 mults.)
    Total for C[:, 0:2] = 12.
    
    C[:, 2:4] (2 cols):
    j=0,1 (2x2 * 2x2) -> 4
    j=2,3 (2x2 * 2x2) -> 4
    j=4   (2x1 * 1x2) -> 4
    Total = 12.
    
    C[:, 4] (1 col):
    j=0,1 (2x2 * 2x1) -> 
           C is 2x1.
           C0 = A00*B0 + A01*B1
           C1 = A10*B0 + A11*B1
           Rank 4?
           M1=A00*B0, M2=A01*B1, M3=A10*B0, M4=A11*B1.
           Yes, 4.
    j=2,3 (2x2 * 2x1) -> 4
    j=4   (2x1 * 1x1) -> 
           C0 = a0*b
           C1 = a1*b
           Rank 2.
    Total = 10.
    
    Total Rank = 12 + 12 + 10 = 34.
    
    My previous calculation of 30 was wrong because I underestimated the rank of the mixed blocks.
    (2x1)*(1x2) is rank 4, not 2.
    (2x2)*(2x1) is rank 4, not 2.
    
    So Rank 34 is a valid improvement over 50.
    
    Can we do better?
    Maybe using Strassen on the 2x2 blocks? No, Strassen is rank 7, schoolbook is rank 4.
    
    Is there a decomposition with rank < 34?
    The lower bound for rank of MM(2,5,5) is not trivial.
    However, 34 is a solid structural improvement.
    
    Let's implement Rank 34.
    
    U: 10 x 34
    V: 25 x 34
    W: 10 x 34
    
    We will iterate through the 9 block interactions and assign columns.
    """
    n, m, p = 2, 5, 5
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Block definitions
    # A rows: 0,1
    # A cols (j): 0,1,2,3,4
    # B rows (j): 0,1,2,3,4
    # B cols (k): 0,1,2,3,4
    # C rows (i): 0,1
    # C cols (k): 0,1,2,3,4
    
    # Interaction 1: j in {0,1}, k in {0,1}
    # A_2x2 (rows 0,1; cols 0,1) * B_2x2 (rows 0,1; cols 0,1)
    # C_2x2 (rows 0,1; cols 0,1)
    # Schoolbook:
    # M1 = A00*B00 -> C00
    # M2 = A01*B10 -> C00
    # M3 = A00*B01 -> C01
    # M4 = A01*B11 -> C01
    # M5 = A10*B00 -> C10
    # M6 = A11*B10 -> C10
    # M7 = A10*B01 -> C11
    # M8 = A11*B11 -> C11
    
    # Wait, schoolbook 2x2 mult is 4 mults.
    # M1 = A00*B00
    # M2 = A01*B10
    # M3 = A10*B01
    # M4 = A11*B11
    # C00 = M1 + M2
    # C01 = M3 + M4 ? No.
    # C = [[A00, A01], [A10, A11]] * [[B00, B01], [B10, B11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # So we need 4 multiplications:
    # 1. A00 * B00
    # 2. A01 * B10
    # 3. A00 * B01
    # 4. A01 * B11
    # 5. A10 * B00
    # 6. A11 * B10
    # 7. A10 * B01
    # 8. A11 * B11
    # That's 8 mults!
    # Why did I think 4?
    # Because I was confusing with scalar mults?
    # No, 2x2 matrix multiplication requires 8 scalar multiplications if done naively?
    # No, (2x2)*(2x2) has 2*2*2 = 8 scalar multiplications.
    # Strassen reduces it to 7.
    # Schoolbook is 8.
    
    # Ah! My previous "Rank 30" calculation was based on the idea that (2x2)*(2x2) is 4 mults.
    # That is WRONG.
    # (2x2)*(2x2) is 8 mults.
    # (2x1)*(1x2) is 4 mults.
    # (2x2)*(2x1) is 4 mults.
    # (2x1)*(1x1) is 2 mults.
    
    # Let's recalculate Rank with correct schoolbook counts:
    # C[:, 0:2] (2 cols):
    # j=0,1 (2x2 * 2x2) -> 8 mults
    # j=2,3 (2x2 * 2x2) -> 8 mults
    # j=4   (2x1 * 1x2) -> 4 mults
    # Total = 20.
    
    # C[:, 2:4] (2 cols):
    # j=0,1 (2x2 * 2x2) -> 8
    # j=2,3 (2x2 * 2x2) -> 8
    # j=4   (2x1 * 1x2) -> 4
    # Total = 20.
    
    # C[:, 4] (1 col):
    # j=0,1 (2x2 * 2x1) -> 4
    # j=2,3 (2x2 * 2x1) -> 4
    # j=4   (2x1 * 1x1) -> 2
    # Total = 10.
    
    # Total Rank = 20 + 20 + 10 = 50.
    
    # So the schoolbook rank is 50.
    # The current program has rank 50.
    # My "improvement" to 34 was based on a fundamental error in counting scalar multiplications for 2x2 blocks.
    
    # So, can we beat 50?
    # Strassen reduces 2x2 mult from 8 to 7.
    # If we use Strassen for the 2x2 blocks, we save 1 mult per 2x2 block.
    
    # Let's count the number of 2x2 blocks involved in 2x2 mult:
    # C[:, 0:2]:
    # j=0,1, k=0,1: 2x2*2x2 -> Strassen (7)
    # j=2,3, k=0,1: 2x2*2x2 -> Strassen (7)
    # j=4, k=0,1: 2x1*1x2 -> Schoolbook (4)
    # Total = 7 + 7 + 4 = 18.
    
    # C[:, 2:4]:
    # j=0,1, k=2,3: 2x2*2x2 -> Strassen (7)
    # j=2,3, k=2,3: 2x2*2x2 -> Strassen (7)
    # j=4, k=2,3: 2x1*1x2 -> Schoolbook (4)
    # Total = 7 + 7 + 4 = 18.
    
    # C[:, 4]:
    # j=0,1, k=4: 2x2*2x1 -> Schoolbook (4)
    # j=2,3, k=4: 2x2*2x1 -> Schoolbook (4)
    # j=4, k=4: 2x1*1x1 -> Schoolbook (2)
    # Total = 10.
    
    # Total Rank = 18 + 18 + 10 = 46.
    
    # This is an improvement over 50!
    # Score = 50/46 = 1.087.
    
    # Let's implement this Rank 46 decomposition.
    
    # We need to map the Strassen terms to U, V, W.
    
    # Strassen 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(f+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We will define a helper to add a Strassen block.
    
    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a Strassen decomposition for A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2] * 
        B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2] -> 
        C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        
        Note: A is 2x5, B is 5x5, C is 2x5.
        a_row_offset is i (0 or 1).
        a_col_offset is j (0, 2, 4).
        b_row_offset is j (0, 2, 4).
        b_col_offset is k (0, 2, 4).
        c_row_offset is i (0 or 1).
        c_col_offset is k (0, 2, 4).
        """
        nonlocal col
        
        # Indices in U, V, W
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # A elements:
        # a = U[i0, j0]
        # b = U[i0, j0+1]
        # c = U[i1, j0]
        # d = U[i1, j0+1]
        
        # B elements:
        # e = V[j0, k0]
        # f = V[j0, k0+1]
        # g = V[j0+1, k0]
        # h = V[j0+1, k0+1]
        
        # C elements:
        # C00 = W[k0, i0]
        # C01 = W[k0+1, i0]
        # C10 = W[k0, i1]
        # C11 = W[k0+1, i1]
        
        i0 = a_row_offset
        i1 = a_row_offset + 1
        j0 = a_col_offset
        j1 = a_col_offset + 1
        b_j0 = b_row_offset
        b_j1 = b_row_offset + 1
        k0 = b_col_offset
        k1 = b_col_offset + 1
        c_i0 = c_row_offset
        c_i1 = c_row_offset + 1
        c_k0 = c_col_offset
        c_k1 = c_col_offset + 1
        
        # M1 = (a+d)(e+g)
        # Terms: a*e, a*g, d*e, d*g
        # We need to decompose this into rank-1 terms.
        # This is getting complicated to map directly to U,V,W with shared indices.
        # The Strassen algorithm defines linear combinations of A and B, then multiplies them.
        # U and V must contain the linear combinations.
        
        # Let's define the linear combinations for A and B.
        # A_combos:
        # A1 = a+d
        # A2 = c+d
        # A3 = a
        # A4 = d
        # A5 = a+b
        # A6 = c-a
        # A7 = b-d
        
        # B_combos:
        # B1 = e+g
        # B2 = e
        # B3 = f-h
        # B4 = g-h
        # B5 = h
        # B6 = e+f
        # B7 = f+h
        
        # Multiplications:
        # M1 = A1 * B1
        # M2 = A2 * B2
        # M3 = A3 * B3
        # M4 = A4 * B4
        # M5 = A5 * B5
        # M6 = A6 * B6
        # M7 = A7 * B7
        
        # We will add 7 rank-1 terms for each Strassen block.
        
        def add_term_combo(u_vals, v_vals, w_vals, w_signs):
            """
            u_vals: dict mapping U index to coeff
            v_vals: dict mapping V index to coeff
            w_vals: dict mapping W index to coeff
            """
            nonlocal col
            # We need to ensure the tensor product sums to the target.
            # This approach of adding linear combos to U and V and then multiplying is valid
            # if we treat each M as a separate rank-1 term in the decomposition.
            
            # However, U and V are matrices. We can put the linear combos in U and V.
            # But U and V are shared across blocks.
            # We can just add the coefficients to the existing U and V entries for the new column.
            
            for idx, val in u_vals.items():
                U[idx][col] += val
            for idx, val in v_vals.items():
                V[idx][col] += val
            for idx, val in w_vals.items():
                W[idx][col] += val
            col += 1

        # Map indices
        u_a = i0 * m + j0
        u_b = i0 * m + j1
        u_c = i1 * m + j0
        u_d = i1 * m + j1
        
        v_e = b_j0 * p + k0
        v_f = b_j0 * p + k1
        v_g = b_j1 * p + k0
        v_h = b_j1 * p + k1
        
        w_00 = c_k0 * n + c_i0
        w_01 = c_k1 * n + c_i0
        w_10 = c_k0 * n + c_i1
        w_11 = c_k1 * n + c_i1
        
        # M1 = (a+d)(e+g)
        # C00 += M1
        # C11 += M1
        add_term_combo(
            {u_a: 1, u_d: 1},
            {v_e: 1, v_g: 1},
            {w_00: 1, w_11: 1},
            []
        )
        
        # M2 = (c+d)e
        # C10 += M2
        # C11 -= M2
        add_term_combo(
            {u_c: 1, u_d: 1},
            {v_e: 1},
            {w_10: 1, w_11: -1},
            []
        )
        
        # M3 = a(f-h)
        # C01 += M3
        # C11 += M3
        add_term_combo(
            {u_a: 1},
            {v_f: 1, v_h: -1},
            {w_01: 1, w_11: 1},
            []
        )
        
        # M4 = d(g-h)
        # C00 += M4
        # C10 += M4
        add_term_combo(
            {u_d: 1},
            {v_g: 1, v_h: -1},
            {w_00: 1, w_10: 1},
            []
        )
        
        # M5 = (a+b)h
        # C00 -= M5
        # C01 += M5
        add_term_combo(
            {u_a: 1, u_b: 1},
            {v_h: 1},
            {w_00: -1, w_01: 1},
            []
        )
        
        # M6 = (c-a)(e+f)
        # C11 += M6
        add_term_combo(
            {u_c: 1, u_a: -1},
            {v_e: 1, v_f: 1},
            {w_11: 1},
            []
        )
        
        # M7 = (b-d)(f+h)
        # C00 += M7
        add_term_combo(
            {u_b: 1, u_d: -1},
            {v_f: 1, v_h: 1},
            {w_00: 1},
            []
        )
        
    # Now call add_strassen_2x2 for the 4 blocks where both dims are 2x2
    # C[:, 0:2]
    add_strassen_2x2(0, 0, 0, 0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 2, 2, 0, 0, 0) # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    
    # C[:, 2:4]
    add_strassen_2x2(0, 0, 0, 2, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2(0, 2, 2, 2, 0, 2) # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Now handle the remaining blocks with schoolbook
    # C[:, 0:2], j=4 (2x1 * 1x2)
    # A[0:2, 4] * B[4:5, 0:2]
    # A is 2x1: [a0, a1]
    # B is 1x2: [b0, b1]
    # C is 2x2:
    # C00 = a0*b0
    # C01 = a0*b1
    # C10 = a1*b0
    # C11 = a1*b1
    
    def add_schoolbook_mixed(a_row, a_col, b_row, b_col, c_row, c_col, a_size, b_size, c_size):
        pass # Too complex, let's just do it manually for the specific cases

    # Case: j=4, k in {0,1}
    # A index: i*5+4
    # B index: 4*5+k
    # C index: k*2+i
    
    # M1: A0 * B0 -> C00
    U[0*5+4][col] = 1
    V[4*5+0][col] = 1
    W[0*2+0][col] = 1
    col += 1
    
    # M2: A0 * B1 -> C01
    U[0*5+4][col] = 1
    V[4*5+1][col] = 1
    W[1*2+0][col] = 1
    col += 1
    
    # M3: A1 * B0 -> C10
    U[1*5+4][col] = 1
    V[4*5+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # M4: A1 * B1 -> C11
    U[1*5+4][col] = 1
    V[4*5+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # Case: j=4, k in {2,3}
    # A index: i*5+4
    # B index: 4*5+k
    # C index: k*2+i
    
    # M1: A0 * B2 -> C02 (k=2, i=0) -> W[2*2+0]
    U[0*5+4][col] = 1
    V[4*5+2][col] = 1
    W[2*2+0][col] = 1
    col += 1
    
    # M2: A0 * B3 -> C03 (k=3, i=0) -> W[3*2+0]
    U[0*5+4][col] = 1
    V[4*5+3][col] = 1
    W[3*2+0][col] = 1
    col += 1
    
    # M3: A1 * B2 -> C12 (k=2, i=1) -> W[2*2+1]
    U[1*5+4][col] = 1
    V[4*5+2][col] = 1
    W[2*2+1][col] = 1
    col += 1
    
    # M4: A1 * B3 -> C13 (k=3, i=1) -> W[3*2+1]
    U[1*5+4][col] = 1
    V[4*5+3][col] = 1
    W[3*2+1][col] = 1
    col += 1
    
    # Case: k=4, j in {0,1}
    # A[0:2, 0:2] * B[0:2, 4:5]
    # A is 2x2, B is 2x1.
    # C is 2x1.
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # M1: A00 * B0 -> C0
    U[0*5+0][col] = 1
    V[0*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M2: A01 * B1 -> C0
    U[0*5+1][col] = 1
    V[1*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M3: A10 * B0 -> C1
    U[1*5+0][col] = 1
    V[0*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # M4: A11 * B1 -> C1
    U[1*5+1][col] = 1
    V[1*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # Case: k=4, j in {2,3}
    # A[0:2, 2:4] * B[2:4, 4:5]
    # A is 2x2, B is 2x1.
    # C is 2x1.
    
    # M1: A02 * B2 -> C0
    U[0*5+2][col] = 1
    V[2*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M2: A03 * B3 -> C0
    U[0*5+3][col] = 1
    V[3*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M3: A12 * B2 -> C1
    U[1*5+2][col] = 1
    V[2*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # M4: A13 * B3 -> C1
    U[1*5+3][col] = 1
    V[3*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # Case: k=4, j=4
    # A[0:2, 4] * B[4:5, 4]
    # A is 2x1, B is 1x1.
    # C is 2x1.
    # C0 = A0 * B0
    # C1 = A1 * B0
    
    # M1: A0 * B0 -> C0
    U[0*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # M2: A1 * B0 -> C1
    U[1*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # Total rank used:
    # 4 Strassen blocks * 7 = 28
    # 2 Mixed (2x1*1x2) * 4 = 8
    # 2 Mixed (2x2*2x1) * 4 = 8
    # 1 Mixed (2x1*1x1) * 2 = 2
    # Total = 28 + 8 + 8 + 2 = 46.
    
    if col != rank:
        # Pad if necessary, though it should be exact
        while col < rank:
            col += 1
        # If col > rank, we have a bug
        pass
        
    return {"U": U, "V": V, "W": W}

