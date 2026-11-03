"""
Kronecker-product-based tensor decomposition for 2x5 * 5x5 matrix multiplication.

Approach:
  Split the middle dimension m=5 into m1=4 and m2=1.
  Split the output dimension p=5 into p1=4 and p2=1.
  
  The multiplication C = A*B (A is 2x5, B is 5x5, C is 2x5) can be written as:
    C[:, :4] = A[:, :4] * B[:4, :4] + A[:, 4:5] * B[4:5, :4]
    C[:, 4:5] = A[:, :4] * B[:4, 4:5] + A[:, 4:5] * B[4:5, 4:5]
  
  This gives four block multiplications:
    T1: 2x4 * 4x4  -> rank 32 (schoolbook)
    T2: 2x4 * 4x1  -> rank 8
    T3: 2x1 * 1x4  -> rank 8
    T4: 2x1 * 1x1  -> rank 2
  Total schoolbook: 50.
  
  But we can do better on T1 (2x4 * 4x4) by using Kronecker structure:
    2x4 * 4x4 = (2x2 * 2x2) ⊗ (1x2 * 2x2)
    Actually, let's think differently.
    
  For 2x4 * 4x4:
    Let A1 be 2x4, B1 be 4x4.
    We can write A1 = [A1_left, A1_right] where each is 2x2.
    B1 = [[B11, B12], [B13, B14]] where each Bi is 2x2.
    
    C1 = A1 * B1 = A1_left * B11 + A1_right * B13  (for left half of C1)
    C1 = A1_left * B12 + A1_right * B14  (for right half of C1)
    
    So C1_left = A1_left * B11 + A1_right * B13
    C1_right = A1_left * B12 + A1_right * B14
    
    Each of these is a sum of two 2x2 * 2x2 multiplications.
    Using Strassen for 2x2 * 2x2 (rank 7), each 2x2*2x2 has rank 7.
    So C1_left needs 2*7 = 14 terms, C1_right needs 2*7 = 14 terms.
    Total for T1: 28.
    
  For T2: 2x4 * 4x1 -> rank 8 (schoolbook, no improvement possible since one dim is 1)
  For T3: 2x1 * 1x4 -> rank 8
  For T4: 2x1 * 1x1 -> rank 2
  
  Total: 28 + 8 + 8 + 2 = 46.
"""

def construct():
    n, m, p = 2, 5, 5
    
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        """Add a single rank-1 term."""
        U.append([u_val] * 1)  # placeholder, we'll build full matrices later
        V.append([v_val] * 1)
        W.append([w_val] * 1)
    
    # We'll build the matrices as lists of rows, each row being a list of values.
    # First, let's collect all terms as (u_row, v_row, w_row, u_val, v_val, w_val)
    terms = []
    
    def add_term_full(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        terms.append((u_row, v_row, w_row, u_val, v_val, w_val))
    
    # Strassen 2x2 * 2x2 decomposition
    # For matrices A (2x2), B (2x2), C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition terms, each Mi corresponds to a set of rank-1 terms.
    # For the tensor decomposition of 2x2*2x2, we need to express each element of
    # A, B, and C as linear combinations of the 7 Strassen products.
    #
    # Let's define the 7 "basis" tensors for Strassen:
    # Each basis tensor has entries for A (4 elements), B (4 elements), C (4 elements).
    # We use indices: A: (0,1,2,3) for (a11,a12,a21,a22)
    #                  B: (0,1,2,3) for (b11,b12,b21,b22)
    #                  C: (0,1,2,3) for (c11,c12,c21,c22)
    
    # Strassen basis tensors (each is a dict mapping (matrix, index) -> coefficient)
    strassen_basis = [
        # M1 = (a11 + a22)(b11 + b22)
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},  # c11, c22
        # M2 = (a21 + a22)b11
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},  # c21, c22
        # M3 = a11(b12 - b22)
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},  # c12, c22
        # M4 = a22(b21 - b11)
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},  # c11, c21
        # M5 = (a11 + a12)b22
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},  # c11, c12
        # M6 = (a21 - a11)(b11 + b12)
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},  # c22
        # M7 = (a12 - a22)(b21 + b22)
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}},  # c11
    ]
    
    # For T1: 2x4 * 4x4
    # A1 is 2x4, indexed by (i, j) where i in {0,1}, j in {0,1,2,3}
    # B1 is 4x4, indexed by (j, k) where j in {0,1,2,3}, k in {0,1,2,3}
    # C1 is 2x4, indexed by (i, k) where i in {0,1}, k in {0,1,2,3}
    #
    # Split A1 into left (j=0,1) and right (j=2,3):
    #   A1_left: 2x2, A1_right: 2x2
    # Split B1 into quadrants:
    #   B11: rows j=0,1, cols k=0,1
    #   B12: rows j=0,1, cols k=2,3
    #   B13: rows j=2,3, cols k=0,1
    #   B14: rows j=2,3, cols k=2,3
    #
    # C1_left (k=0,1) = A1_left * B11 + A1_right * B13
    # C1_right (k=2,3) = A1_left * B12 + A1_right * B14
    #
    # For each 2x2*2x2 multiplication, we use Strassen (rank 7).
    # So C1_left uses 2*7=14 terms, C1_right uses 2*7=14 terms.
    
    # Helper to add Strassen terms for a 2x2*2x2 block
    def add_strassen_terms(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen terms for a 2x2 * 2x2 block.
        A block is at rows [a_row_offset, a_row_offset+1], cols [a_col_offset, a_col_offset+1]
        B block is at rows [b_row_offset, b_row_offset+1], cols [b_col_offset, b_col_offset+1]
        C block is at rows [c_row_offset, c_row_offset+1], cols [c_col_offset, c_col_offset+1]
        
        Note: In our global indexing:
          U row = i * m + j = i * 5 + j
          V row = j * p + k = j * 5 + k
          W row = k * n + i = k * 2 + i
        
        For the 2x4*4x4 block (T1):
          i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
          U row = i * 5 + j
          V row = j * 5 + k
          W row = k * 2 + i
        """
        for basis in strassen_basis:
            for (a_idx, a_coeff) in basis['A'].items():
                i_local = a_idx // 2  # 0 or 1
                j_local = a_idx % 2   # 0 or 1
                i = a_row_offset + i_local
                j = a_col_offset + j_local
                u_row = i * 5 + j
                u_val = a_coeff
                # We need to pair this with the B and C coefficients
                # Actually, each Strassen basis tensor defines a complete rank-1 term
                # where U, V, W all have specific values.
                pass
        
        # Let me restructure: for each Strassen basis, we add ONE term
        # where U has values for the 4 A elements, V has values for the 4 B elements,
        # and W has values for the 4 C elements.
        # But in our tensor decomposition, each term is a single rank-1 tensor,
        # so U, V, W each have ONE non-zero entry per term? No, that's not right.
        #
        # Actually, in the tensor decomposition of matrix multiplication,
        # each rank-1 term corresponds to a triple (u, v, w) where u is a vector
        # in R^(n*m), v is a vector in R^(m*p), and w is a vector in R^(p*n).
        # The tensor is sum_r u_r ⊗ v_r ⊗ w_r.
        #
        # For Strassen, each of the 7 products M1..M7 corresponds to a set of
        # rank-1 terms. But actually, each Mi is itself a product of linear
        # forms, so it can be expressed as a sum of rank-1 tensors.
        #
        # Wait, I think I'm overcomplicating this. Let me think again.
        #
        # The tensor for 2x2*2x2 has rank 7 (Strassen). This means we can write
        # the tensor as a sum of 7 rank-1 tensors. Each rank-1 tensor is
        # u ⊗ v ⊗ w where u ∈ R^4, v ∈ R^4, w ∈ R^4.
        #
        # For the standard schoolbook, we have 8 rank-1 terms:
        # For each (i,j,k): u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i).
        #
        # For Strassen, we need to find 7 vectors (u_r, v_r, w_r) such that
        # sum_r u_r ⊗ v_r ⊗ w_r = the 2x2*2x2 tensor.
        #
        # The Strassen decomposition can be written as:
        # M1: u has 1 at (a11,a22), v has 1 at (b11,b22), w has 1 at (c11,c22)
        # But this is NOT a rank-1 tensor because u has two non-zero entries.
        #
        # Actually, each Mi is a product of two linear forms in A and B,
        # and it contributes to linear combinations in C. So each Mi corresponds
        # to a rank-1 tensor where:
        #   u = (linear form in A) as a vector in R^4
        #   v = (linear form in B) as a vector in R^4  
        #   w = (linear form in C) as a vector in R^4
        #
        # For example, M1 = (a11 + a22)(b11 + b22) contributes to c11 and c22.
        # So: u = (1, 0, 0, 1) for (a11, a12, a21, a22)
        #     v = (1, 0, 0, 1) for (b11, b12, b21, b22)
        #     w = (1, 0, 0, 1) for (c11, c12, c21, c22)
        #
        # This IS a valid rank-1 tensor! The vectors can have multiple non-zero entries.
        #
        # So let's define the 7 Strassen rank-1 tensors:
        
        strassen_tensors = [
            # M1 = (a11+a22)(b11+b22) -> c11, c22
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            # M2 = (a21+a22)b11 -> c21, -c22
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            # M3 = a11(b12-b22) -> c12, c22
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            # M4 = a22(b21-b11) -> c11, c21
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            # M5 = (a11+a12)b22 -> -c11, c12
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            # M6 = (a21-a11)(b11+b12) -> c22
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            # M7 = (a12-a22)(b21+b22) -> c11
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        ]
        
        for st in strassen_tensors:
            # Map local indices to global indices
            u_vec = {}
            for (local_idx, coeff) in st['u'].items():
                i_local = local_idx // 2
                j_local = local_idx % 2
                i = a_row_offset + i_local
                j = a_col_offset + j_local
                u_row = i * 5 + j
                u_vec[u_row] = u_vec.get(u_row, 0) + coeff
            
            v_vec = {}
            for (local_idx, coeff) in st['v'].items():
                j_local = local_idx // 2
                k_local = local_idx % 2
                j = b_row_offset + j_local
                k = b_col_offset + k_local
                v_row = j * 5 + k
                v_vec[v_row] = v_vec.get(v_row, 0) + coeff
            
            w_vec = {}
            for (local_idx, coeff) in st['w'].items():
                k_local = local_idx // 2
                i_local = local_idx % 2
                k = c_col_offset + k_local
                i = c_row_offset + i_local
                w_row = k * 2 + i
                w_vec[w_row] = w_vec.get(w_row, 0) + coeff
            
            # Add this as a term
            add_term_full(u_vec, v_vec, w_vec)
    
    # T1: 2x4 * 4x4
    # C1_left = A1_left * B11 + A1_right * B13
    #   A1_left: rows i=0,1, cols j=0,1
    #   B11: rows j=0,1, cols k=0,1
    #   C1_left: rows i=0,1, cols k=0,1
    add_strassen_terms(0, 0, 0, 0, 0, 0)  # A1_left * B11 -> C1_left
    add_strassen_terms(0, 2, 2, 0, 0, 0)  # A1_right * B13 -> C1_left
    
    # C1_right = A1_left * B12 + A1_right * B14
    #   B12: rows j=0,1, cols k=2,3
    #   C1_right: rows i=0,1, cols k=2,3
    add_strassen_terms(0, 0, 0, 2, 0, 2)  # A1_left * B12 -> C1_right
    add_strassen_terms(0, 2, 2, 2, 0, 2)  # A1_right * B14 -> C1_right
    
    # T2: 2x4 * 4x1 (A[:, :4] * B[:4, 4:5])
    # A is 2x4 (i=0,1; j=0,1,2,3), B is 4x1 (j=0,1,2,3; k=4)
    # C is 2x1 (i=0,1; k=4)
    # Schoolbook: rank 8
    for i in range(2):
        for j in range(4):
            u_row = i * 5 + j
            v_row = j * 5 + 4
            w_row = 4 * 2 + i
            add_term_full({u_row: 1}, {v_row: 1}, {w_row: 1})
    
    # T3: 2x1 * 1x4 (A[:, 4:5] * B[4:5, :4])
    # A is 2x1 (i=0,1; j=4), B is 1x4 (j=4; k=0,1,2,3)
    # C is 2x4 (i=0,1; k=0,1,2,3)
    # Schoolbook: rank 8
    for i in range(2):
        for k in range(4):
            u_row = i * 5 + 4
            v_row = 4 * 5 + k
            w_row = k * 2 + i
            add_term_full({u_row: 1}, {v_row: 1}, {w_row: 1})
    
    # T4: 2x1 * 1x1 (A[:, 4:5] * B[4:5, 4:5])
    # A is 2x1 (i=0,1; j=4), B is 1x1 (j=4; k=4)
    # C is 2x1 (i=0,1; k=4)
    # Schoolbook: rank 2
    for i in range(2):
        u_row = i * 5 + 4
        v_row = 4 * 5 + 4
        w_row = 4 * 2 + i
        add_term_full({u_row: 1}, {v_row: 1}, {w_row: 1})
    
    # Now build the U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        for row, val in u_vec.items():
            U[row][r] = val
        for row, val in v_vec.items():
            V[row][r] = val
        for row, val in w_vec.items():
            W[row][r] = val
    
    return {"U": U, "V": V, "W": W}
