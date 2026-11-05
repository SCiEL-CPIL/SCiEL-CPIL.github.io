"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition sizes
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    
    # U, V, W will be lists of lists
    # We don't know the rank yet, so we'll use a list of tuples (index, coeff) for each matrix
    # Then convert to dense matrix at the end.
    U_terms = [dict() for _ in range(n * m)]
    V_terms = [dict() for _ in range(m * p)]
    W_terms = [dict() for _ in range(p * n)]
    
    rank = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal rank
        if coeff == 0:
            return
        # We use a unique ID for each rank-1 tensor
        # But since we are adding to existing columns, we need to ensure consistency.
        # Actually, each "term" in the decomposition is a rank-1 tensor.
        # We can just append to a list of (u, v, w, coeff) and then build the matrices.
        # But U, V, W must have the same number of columns.
        # So we create a new column for each term.
        
    # This approach is inefficient if we don't know the rank.
    # Let's estimate the rank first or just build it dynamically.
    # We'll build a list of tuples: (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    def add_rank1(u_idx, v_idx, w_idx, coeff):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))
            
    # Helper to add schoolbook decomposition for blocks of size ra, ca, rb, cb, rc, cc
    # Here A is ra x ca, B is rb x cb, C is rc x cc
    # Must have ca == rb and ra == rc, ca == cc? No.
    # A (ra, ca) * B (ca, cb) = C (ra, cb)
    # So ra == rc, ca == rb, cb == cc.
    
    def add_schoolbook(ra, ca, rb, cb, rc, cc, start_i, start_j, start_k, offset):
        # start_i is global row start for A/C
        # start_j is global col start for A / row start for B
        # start_k is global col start for B / col start for C
        for i in range(ra):
            for j in range(ca):
                for k in range(cb):
                    # A[i, j] B[j, k] C[i, k]
                    u_idx = (start_i + i) * m + (start_j + j)
                    v_idx = (start_j + j) * p + (start_k + k)
                    w_idx = (start_k + k) * n + (start_i + i)
                    add_rank1(u_idx, v_idx, w_idx, 1)

    def add_strassen(ra, ca, rb, cb, rc, cc, start_i, start_j, start_k, offset):
        # Only for 2x2x2
        if not (ra == 2 and ca == 2 and rb == 2 and cb == 2 and rc == 2 and cc == 2):
            add_schoolbook(ra, ca, rb, cb, rc, cc, start_i, start_j, start_k, offset)
            return
            
        # Strassen's 7 products for 2x2 matrices
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Local indices 0,1
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        # C00, C01, C10, C11
        
        # We need to express M1..M7 as linear combinations of A and B entries,
        # and then C entries as linear combinations of M's.
        # But the tensor decomposition requires rank-1 tensors.
        # Each M is a product of a linear form in A and a linear form in B.
        # So each M corresponds to a rank-1 tensor (or sum of rank-1 if not single term).
        # Wait, M1 = (A00+A11)(B00+B11).
        # This expands to 4 terms: A00B00 + A00B11 + A11B00 + A11B11.
        # Each of these is a rank-1 tensor?
        # No, the tensor is sum of A_ij B_jk C_ik.
        # If we use M1, we are effectively doing:
        # C_ik += M1 * coeff.
        # But M1 is not a single A_ij B_jk product.
        # So we must expand M1 into its constituent A_ij B_jk products.
        # This means Strassen's algorithm doesn't directly give a rank-7 tensor decomposition
        # in the sense of 7 rank-1 tensors T = sum u v w.
        # It gives 7 scalar multiplications, each of which is a bilinear form.
        # The rank of the tensor is the number of scalar multiplications.
        # So we need to represent each scalar multiplication as a rank-1 tensor?
        # No, a scalar multiplication $M = L_A \cdot L_B$ where $L_A$ is a linear form in A and $L_B$ is a linear form in B.
        # The contribution to C is $M \cdot L_C$ where $L_C$ is a linear form in C.
        # This is equivalent to $\sum_{i,j} \alpha_{ij} A_{ij} \sum_{j,k} \beta_{jk} B_{jk} \sum_{i,k} \gamma_{ik} C_{ik}$.
        # This is a rank-1 tensor in the space of A, B, C?
        # Yes, $u \otimes v \otimes w$ where $u$ is the vector of coefficients for A, etc.
        
        # So for each Strassen product M, we define u, v, w vectors.
        
        def get_u_coeff(i, j):
            # Returns coefficient for A[i,j] in the linear form
            pass
        
        # M1 = (A00+A11)(B00+B11)
        # u1 = [1, 0, 0, 1] (A00, A01, A10, A11)
        # v1 = [1, 0, 0, 1] (B00, B01, B10, B11)
        # C contribution: M1 is added to C00 and C11, subtracted from C01 and C10?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1, the w vector is [1, -1, -1, 1] (C00, C01, C10, C11)
        # We add this as a single rank-1 tensor?
        # No, the tensor is $T = \sum M_k u_k \otimes v_k \otimes w_k$.
        # This is a valid decomposition.
        
        # We need to map local indices to global indices for u, v, w.
        
        def add_strassen_term(u_local, v_local, w_local):
            # u_local is dict or list of 4 coeffs for A
            # v_local is dict or list of 4 coeffs for B
            # w_local is dict or list of 4 coeffs for C
            # We create a single rank-1 tensor for this term?
            # No, a rank-1 tensor is $u \otimes v \otimes w$.
            # This corresponds to one column in U, V, W.
            # U column will have entries for all A indices involved.
            
            for i in range(2):
                for j in range(2):
                    coeff = u_local[i][j]
                    if coeff != 0:
                        u_idx = (start_i + i) * m + (start_j + j)
                        # We need to add this to the current column
                        pass
        
        # This is getting complicated to manage dynamically.
        # Let's just expand Strassen into its 7 rank-1 tensors explicitly.
        
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}
        
        # M2
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1, (1,1): -1}
        
        # M3
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}
        
        # M4
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        
        # M5
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}
        
        # M6
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        
        # M7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        
        terms_list = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u_l, v_l, w_l in terms_list:
            # Create a new column for U, V, W
            col_u = {}
            col_v = {}
            col_w = {}
            for (i,j), c in u_l.items():
                idx = (start_i + i) * m + (start_j + j)
                col_u[idx] = c
            for (i,j), c in v_l.items():
                idx = (start_j + i) * p + (start_k + j) # Note: B row is i, col is j
                col_v[idx] = c
            for (i,j), c in w_l.items():
                idx = (start_k + j) * n + (start_i + i) # Note: W index is k*n+i. Here k is col of C, i is row of C.
                col_w[idx] = c
            
            # Add this column to the global terms
            # We need to store these columns
            terms.append((col_u, col_v, col_w))

    # Iterate over blocks
    for a in range(3):
        for b in range(3):
            for c in range(3):
                ra = sizes[a]
                ca = sizes[c]
                rb = sizes[c]
                cb = sizes[b]
                rc = sizes[a]
                cc = sizes[b]
                
                start_i = starts[a]
                start_j = starts[c]
                start_k = starts[b]
                
                if ra == 2 and ca == 2 and rb == 2 and cb == 2:
                    add_strassen(ra, ca, rb, cb, rc, cc, start_i, start_j, start_k, 0)
                else:
                    add_schoolbook(ra, ca, rb, cb, rc, cc, start_i, start_j, start_k, 0)
                    
    # Now build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        col_u, col_v, col_w = terms[r]
        for idx, val in col_u.items():
            U[idx][r] = val
        for idx, val in col_v.items():
            V[idx][r] = val
        for idx, val in col_w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
