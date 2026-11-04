"""Matrix multiplication 2x5 * 5x6 decomposition using Strassen on 2x2 column blocks."""

def construct():
    n, m, p = 2, 5, 6

    # We will build U (10 x rank), V (30 x rank), W (12 x rank)
    # Rows of U indexed by i*m + j  (i=0,1; j=0..4)
    # Rows of V indexed by j*p + k  (j=0..4; k=0..5)
    # Rows of W indexed by k*n + i  (k=0..5; i=0,1)

    U_cols = []
    V_cols = []
    W_cols = []

    def add_tensor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Split B's 6 columns into 3 pairs: (0,1), (2,3), (4,5)
    # For each pair, solve a 2x5 * 5x2 subproblem using Strassen-like decomposition
    # Split A's 5 columns into: (0,1), (2,3), (4)
    # For the 2x2 block of A cols x 2x2 block of B cols: use Strassen (7 mults)
    # For the 2x1 block of A cols x 2x2 block of B cols: use schoolbook (4 mults)
    # Total per column block: 7 + 4 = 11 terms
    # Total rank: 3 * 11 = 33

    for block in range(3):
        # B columns: 2*block and 2*block+1
        k1 = 2 * block
        k2 = 2 * block + 1

        # --- Strassen for A cols (0,1) x B cols (k1,k2) ---
        # A sub-block is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # B sub-block is 5x2 but we only use rows corresponding to A cols 0,1
        # Actually, the multiplication is:
        # C_part = sum_{j=0,1} A[:,j] * B[j,:] where B[j,:] is the row of B at index j
        # Wait, no. The full multiplication is C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
        # For the sub-block using A cols 0,1 and B rows 0,1 (which contribute to all B cols):
        # The contribution to C[i,k] from j in {0,1} is:
        # sum_{j=0,1} A[i,j] * B[j,k]
        # This is a 2x2 * 2x2 matrix multiplication where:
        # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (2x2)
        # B_sub = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]  (2x2)
        # C_sub = A_sub * B_sub  (2x2)
        
        # Strassen for 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B00+B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A10)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        # Each M is a scalar multiplication of linear combinations.
        # For the tensor decomposition, each rank-1 term corresponds to one scalar mult.
        
        # M1: (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k1] for C[0,k1], B[0,k2]+B[1,k2] for C[0,k2])
        # Wait, this is getting complex. Let me think in terms of the tensor.
        
        # The tensor T[i,j,k] = delta_{i,j,k} for matrix mult is:
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # For the sub-block, we have a 2x2x2 tensor for the Strassen part.
        
        # Let me just enumerate the 7 Strassen terms for the 2x2x2 case and map them.
        
        # Strassen terms for computing C_sub = A_sub * B_sub:
        # Each term is of the form (linear_comb_A) * (linear_comb_B) contributing to C_sub entries.
        
        # Term 1: M1 = (A00+A11) * (B00+B11)
        #   A part: A[0,0] + A[1,1] -> u_vec has 1 at (0,0) and (1,1)
        #   B part: B[0,k1]+B[1,k1] for k=k1, B[0,k2]+B[1,k2] for k=k2
        #   Wait, B part is a vector over (j,k) indices. For the sub-block, j in {0,1}, k in {k1,k2}
        #   B part: (B[0,k1]+B[1,k1]) contributes to... hmm.
        
        # Actually, let me reconsider. The multiplication C[i,k] = sum_j A[i,j]B[j,k].
        # For the sub-block with A cols {0,1} and B rows {0,1}, the contribution is:
        # C_sub[i,k] = sum_{j in {0,1}} A[i,j] * B[j,k]
        # This is a 2x2 matrix multiplication where the "inner" dimension is j in {0,1}.
        
        # For Strassen, we need to express the 4 output entries as linear combinations of 7 products.
        # Each product is (sum_{i,j} u_{i,j} A[i,j]) * (sum_{j,k} v_{j,k} B[j,k])
        # And the output is C[i,k] = sum_{terms} w_{i,k,term} * product_term
        
        # For the 2x2x2 Strassen:
        # A is 2x2, B is 2x2, C is 2x2.
        # The 7 terms are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B00+B10)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A10)(B10+B11)
        
        # For each term, we create a rank-1 decomposition.
        # The U vector has entries for A indices (i,j) where i in {0,1}, j in {0,1} for this sub-block.
        # The V vector has entries for B indices (j,k) where j in {0,1}, k in {k1,k2}.
        # The W vector has entries for C indices (i,k) where i in {0,1}, k in {k1,k2}.
        
        # Let's define the 7 Strassen terms for 2x2x2:
        strassen_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: (A00, A01, A10, A11) in row-major for 2x2
            # v: (B00, B01, B10, B11) in row-major for 2x2 (but here B is 2x2 with cols k1,k2)
            # w: (C00, C01, C10, C11) in row-major for 2x2
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): 1}),
            # M3
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4
            ({(1,1): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,1): 1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): -1}),
        ]
        
        # Now map these to the global indices.
        # For A: sub-block cols are 0,1. Global A index: i*5 + j
        # For B: sub-block rows are 0,1. Global B index: j*6 + k where k is k1 or k2
        # For C: sub-block cols are k1,k2. Global C index: k*2 + i
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            u_vec = [0] * 10
            v_vec = [0] * 30
            w_vec = [0] * 12
            
            for (ai, aj), coeff in u_coeffs.items():
                u_vec[ai * 5 + aj] = coeff
            
            for (bj, bk), coeff in v_coeffs.items():
                # bk is 0 or 1, mapping to k1 or k2
                global_k = k1 + bk
                v_vec[bj * 6 + global_k] = coeff
            
            for (ci, ck), coeff in w_coeffs.items():
                # ck is 0 or 1, mapping to k1 or k2
                global_k = k1 + ck
                w_vec[global_k * 2 + ci] = coeff
            
            add_tensor(u_vec, v_vec, w_vec)
        
        # --- Schoolbook for A col (4) x B cols (k1,k2) ---
        # This is a 2x1 * 1x2 multiplication.
        # C_part[i,k] = A[i,4] * B[4,k]
        # 4 terms:
        # C[0,k1] = A[0,4] * B[4,k1]
        # C[0,k2] = A[0,4] * B[4,k2]
        # C[1,k1] = A[1,4] * B[4,k1]
        # C[1,k2] = A[1,4] * B[4,k2]
        
        for i in range(2):
            for k_offset in range(2):
                global_k = k1 + k_offset
                u_vec = [0] * 10
                v_vec = [0] * 30
                w_vec = [0] * 12
                
                u_vec[i * 5 + 4] = 1
                v_vec[4 * 6 + global_k] = 1
                w_vec[global_k * 2 + i] = 1
                
                add_tensor(u_vec, v_vec, w_vec)

    # Convert lists to the required format
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}
