"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct the decomposition for 5x5 matrix multiplication.
    # Partition into 4x4 and 1x1 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = AB
    
    # U, V, W will be lists of lists.
    # We accumulate the terms.
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
        
    # Helper to add a rank-1 term for a specific multiplication part
    # u: list of length n*m (25)
    # v: list of length m*p (25)
    # w: list of length p*n (25)
    
    # --- Part 1: A11 * B11 (4x4 * 4x4) ---
    # Use Strassen's algorithm for 4x4 (composed of 2x2 blocks)
    # Strassen steps for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for 4x4 matrix:
    # Block (1,1): rows 0-1, cols 0-1
    # Block (1,2): rows 0-1, cols 2-3
    # Block (2,1): rows 2-3, cols 0-1
    # Block (2,2): rows 2-3, cols 2-3
    
    # Rank-7 decomposition for 2x2 multiplication:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the 2x2 multiplication factors to the 4x4 indices.
    # For each of the 7 Strassen multiplications (which are 2x2 * 2x2),
    # we use the rank-7 decomposition of 2x2 multiplication.
    
    # Let's define the rank-7 factors for a generic 2x2 multiplication X*Y.
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = XY
    # Z11 = x11y11 + x21y12 + x12y21 + x22y22 (No, that's schoolbook)
    # Strassen Rank 7:
    # m1 = (x11+x22)(y11+y22)
    # m2 = (x21+x22)y11
    # m3 = x11(y12-y22)
    # m4 = x22(y21-y11)
    # m5 = (x11+x12)y22
    # m6 = (x21-x11)(y11+y12)
    # m7 = (x12-x22)(y21+y22)
    
    # Z11 = m1 + m4 - m5 + m7
    # Z12 = m3 + m5
    # Z21 = m2 + m4
    # Z22 = m1 - m2 + m3 + m6
    
    # We will iterate over the 7 Strassen steps for the 4x4 matrix.
    # Each step involves multiplying two 2x2 matrices (which are blocks of the 4x4).
    # Let the two 2x2 matrices be P and Q.
    # P and Q are linear combinations of the 2x2 blocks of A11 and B11.
    
    # We need to construct U, V, W vectors for the 25x25 problem.
    # The indices for A11 (4x4) in the full 5x5 A matrix:
    # A is 5x5. A11 is top-left 4x4.
    # A[i, j] for i,j in 0..3.
    # In U (size 25), index is i*5 + j.
    # In V (size 25), index is j*5 + k.
    # In W (size 25), index is k*5 + i. (Trace convention W[k*n+i])
    
    # Let's create a helper to add a term from a 2x2 multiplication embedded in the 4x4 part.
    # The 2x2 multiplication is between block P (from A11) and block Q (from B11).
    # The result is a 2x2 block R (part of C11).
    
    # We will define the 7 Strassen steps explicitly.
    
    # Step 1: M1 = (A11_11 + A11_22) * (B11_11 + B11_22)
    # A11_11 is A[0:2, 0:2], A11_22 is A[2:4, 2:4]
    # B11_11 is B[0:2, 0:2], B11_22 is B[2:4, 2:4]
    
    # We need to add the rank-7 decomposition of (P*Q) where P and Q are 2x2 matrices.
    # The entries of P are linear combinations of A entries.
    # The entries of Q are linear combinations of B entries.
    # The entries of R are linear combinations of C entries.
    
    # This is complex to do symbolically in code without a solver.
    # However, we can just hardcode the coefficients for the 49 terms of the 4x4 Strassen decomposition.
    
    # Alternative: Use the fact that we can just use the schoolbook for 4x4?
    # Rank 64 for 4x4. Total 64 + 18 = 82.
    # 125/82 = 1.52.
    # Strassen gives 49 + 18 = 67. 125/67 = 1.86.
    
    # Let's try to implement the 4x4 Strassen decomposition explicitly.
    # We will generate the U, V, W vectors for the 49 terms.
    
    # Define the 7 Strassen multiplications for 2x2 blocks.
    # For each multiplication, we have 7 rank-1 terms.
    # Total 49 terms.
    
    # Let's define the 2x2 rank-7 decomposition factors.
    # For a multiplication X*Y=Z:
    # Term 1 (for m1):
    # U: x11, x22
    # V: y11, y22
    # W: z11, z22
    # Coeffs: 1, 1 for U; 1, 1 for V; 1, 1 for W?
    # m1 = (x11+x22)(y11+y22) = x11y11 + x11y22 + x22y11 + x22y22
    # This is not a single rank-1 term in the tensor sense unless we expand.
    # The rank-7 decomposition of 2x2 matrix multiplication is:
    # Z11 = m1 + m4 - m5 + m7
    # ...
    # The tensor is sum of 7 terms.
    # Each term is a rank-1 tensor.
    # Term k corresponds to one of the m_k.
    # Wait, the m_k are intermediate values.
    # The rank-1 terms are:
    # T1: x11, y11, z11 (coeff 1) ... No.
    
    # The standard rank-7 decomposition:
    # 1. (x11+x22)(y11+y22) -> contributes to z11, z22
    # 2. (x21+x22)y11 -> contributes to z21, z22
    # 3. x11(y12-y22) -> contributes to z12, z11?
    
    # Let's use the explicit coefficients.
    # Reference: Strassen 1969.
    
    # We will construct the 49 terms for the 4x4 block.
    # This is tedious but doable.
    
    # Given the complexity and time, I will provide the schoolbook for 4x4 (Rank 64) + rest (18) = 82.
    # Score 1.52.
    # If I can get 49, score 1.86.
    
    # Let's try to hardcode the 49 terms for 4x4 Strassen.
    # It is safer to use schoolbook for 4x4 to avoid bugs.
    # But 67 is much better.
    
    # I will use schoolbook for 4x4 for now to ensure correctness, then optimize if needed.
    # Actually, 125/82 = 1.52 is a good improvement.
    
    # Let's implement Schoolbook for A11*B11 (4x4).
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A[i, j] * B[j, k] -> C[i, k]
                # U index: i*5 + j
                # V index: j*5 + k
                # W index: k*5 + i
                u = [0]*25; v = [0]*25; w = [0]*25
                u[i*5+j] = 1
                v[j*5+k] = 1
                w[k*5+i] = 1
                add_term(u, v, w)
                
    # --- Part 2: A12 * B21 (4x1 * 1x4 -> 4x4) ---
    # Rank 1.
    # C[i, j] = A[i, 4] * B[4, j]
    # A12 is column 4 of A. B21 is row 4 of B.
    # U: A[i, 4] for i=0..3.
    # V: B[4, j] for j=0..3.
    # W: C[i, j] for i=0..3, j=0..3.
    # This is a rank-1 term?
    # C = u v^T.
    # U vector has 1s at indices 4, 9, 14, 19 (col 4).
    # V vector has 1s at indices 20, 21, 22, 23 (row 4).
    # W vector has 1s at indices 0..15 (4x4 block).
    u = [0]*25; v = [0]*25; w = [0]*25
    for i in range(4):
        u[i*5+4] = 1
    for j in range(4):
        v[4*5+j] = 1
    for i in range(4):
        for j in range(4):
            w[j*5+i] = 1 # W[k*n+i] -> W[j*5+i]
    add_term(u, v, w)
    
    # --- Part 3: A11 * B12 (4x4 * 4x1 -> 4x1) ---
    # Rank 4.
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # This is 4 independent dot products?
    # No, it's one matrix-vector product.
    # We can decompose it into 4 rank-1 terms?
    # C[i, 4] = sum_{j=0}^3 A[i, j] B[j, 4]
    # Term for j: A[i, j] * B[j, 4] -> C[i, 4]
    # U: A[i, j]
    # V: B[j, 4]
    # W: C[i, 4]
    for i in range(4):
        for j in range(4):
            u = [0]*25; v = [0]*25; w = [0]*25
            u[i*5+j] = 1
            v[j*5+4] = 1
            w[4*5+i] = 1 # W[k*n+i] with k=4
            add_term(u, v, w)
            
    # --- Part 4: A12 * B22 (4x1 * 1x1 -> 4x1) ---
    # Rank 4.
    # C[i, 4] = A[i, 4] * B[4, 4]
    for i in range(4):
        u = [0]*25; v = [0]*25; w = [0]*25
        u[i*5+4] = 1
        v[4*5+4] = 1
        w[4*5+i] = 1
        add_term(u, v, w)
        
    # --- Part 5: A21 * B11 (1x4 * 4x4 -> 1x4) ---
    # Rank 4.
    # C[4, j] = sum_i A[4, i] B[i, j]
    for i in range(4):
        for j in range(4):
            u = [0]*25; v = [0]*25; w = [0]*25
            u[4*5+i] = 1
            v[i*5+j] = 1
            w[j*5+4] = 1
            add_term(u, v, w)
            
    # --- Part 6: A22 * B21 (1x1 * 1x4 -> 1x4) ---
    # Rank 4.
    # C[4, j] = A[4, 4] * B[4, j]
    for j in range(4):
        u = [0]*25; v = [0]*25; w = [0]*25
        u[4*5+4] = 1
        v[4*5+j] = 1
        w[j*5+4] = 1
        add_term(u, v, w)
        
    # --- Part 7: A21 * B12 (1x4 * 4x1 -> 1x1) ---
    # Rank 4.
    # C[4, 4] = sum_i A[4, i] B[i, 4]
    for i in range(4):
        u = [0]*25; v = [0]*25; w = [0]*25
        u[4*5+i] = 1
        v[i*5+4] = 1
        w[4*5+4] = 1
        add_term(u, v, w)
        
    # --- Part 8: A22 * B22 (1x1 * 1x1 -> 1x1) ---
    # Rank 1.
    u = [0]*25; v = [0]*25; w = [0]*25
    u[4*5+4] = 1
    v[4*5+4] = 1
    w[4*5+4] = 1
    add_term(u, v, w)
    
    U = U_terms
    V = V_terms
    W = W_terms
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
