import time

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    # We'll build the decomposition for 4x4x6 by splitting into 4x4x4 and 4x4x2
    # 4x4x4: Strassen (rank 49)
    # 4x4x2: Split into 4 products of 2x2x2, each Strassen (rank 7), total 28
    # Total rank: 77
    
    # Helper: Create a single tensor term (rank-1)
    # U entry at (i,j) for matrix A position
    # V entry at (j,k) for matrix B position  
    # W entry at (k*n+i) for output C position [following W[k*n+i, r] convention]
    
    # Let's build U, V, W as lists of columns (each column is one rank-1 term)
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    n, m, p = 4, 4, 6
    u_rows = n * m  # 16
    v_rows = m * p  # 24
    w_rows = p * n  # 24
    
    U_terms = []  # List of (row, value) pairs for U
    V_terms = []  # List of (row, value) pairs for V
    W_terms = []  # List of (row, value) pairs for W
    
    # Function to add a rank-1 term with coefficients in U, V, W
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict mapping (i,j) -> coeff for U[i*m+j]
        v_coeffs: dict mapping (j,k) -> coeff for V[j*p+k]
        w_coeffs: dict mapping (i,k) -> coeff for W[k*n+i]
        """
        U_terms.append(u_coeffs)
        V_terms.append(v_coeffs)
        W_terms.append(w_coeffs)
    
    # Strassen 2x2x2 base case
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2(a_base, b_base, c_base, offset_u, offset_v, offset_w):
        """
        Add Strassen terms for a 2x2 block multiplication.
        a_base: (i_offset, j_offset) - top-left position in A
        b_base: (j_offset, k_offset) - top-left position in B
        c_base: (i_offset, k_offset) - top-left position in C
        offset_u, offset_v, offset_w: row offsets for U, V, W
        """
        ai, aj = a_base
        bi, bj = b_base
        ci, ck = c_base
        
        # A indices: (ai+di, aj+dj) for di,dj in {0,1}
        # B indices: (bi+dj, bj+dk) for dj,dk in {0,1}
        # C indices: (ci+di, ck+dk) for di,dk in {0,1}
        
        # M1 = (A11+A22)(B11+B22)
        add_term(
            {(ai,aj): 1, (ai+1,aj+1): 1},
            {(bi,bj): 1, (bi+1,bj+1): 1},
            {(ci,ck): 1, (ci+1,ck+1): 1}  # Wait, this isn't right
        )
        
        # Hmm, I need to be more careful. Each M term is a product of linear combinations.
        # M1 = (A[ai,aj] + A[ai+1,aj+1]) * (B[bi,bj] + B[bi+1,bj+1])
        # This contributes to C[ai,ck], C[ai,ck+1], C[ai+1,ck], C[ai+1,ck+1]
        # specifically: C11 gets +M1, C12 gets 0, C21 gets 0, C22 gets 0... 
        # Wait no, M1 contributes to C11 and C22 with +1, and to C12 and C21 with... 
        # Let me re-derive.
        
        # Actually in Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes +1 to C11 and +1 to C22.
        # But M1 = (A11+A22)(B11+B22) is a single product term.
        # In the tensor decomposition, each M is one rank-1 term.
        
        # So for M1: U has A11 and A22 with coeff 1, V has B11 and B22 with coeff 1
        # W has C11 with coeff +1 and C22 with coeff +1
