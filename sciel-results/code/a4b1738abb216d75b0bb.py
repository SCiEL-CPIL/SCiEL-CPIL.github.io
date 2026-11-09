"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5,5,5
    
    # Partition indices into blocks of size 2, 2, 1
    # Indices: 0,1 (Block 0), 2,3 (Block 1), 4 (Block 2)
    # Sizes: 2, 2, 1
    
    # We will build the U, V, W matrices by appending terms for each block.
    # Each term is a rank-1 component.
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_schoolbook_block(start_i, size_i, start_j, size_j, start_k, size_k):
        """
        Adds schoolbook terms for a block of size size_i x size_j x size_k.
        start_i, start_j, start_k are the global starting indices.
        """
        for i in range(size_i):
            for j in range(size_j):
                for k in range(size_k):
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    # Global indices
                    gi = start_i + i
                    gj = start_j + j
                    gk = start_k + k
                    
                    # U index: i*m + j
                    u_vec[gi * m + gj] = 1
                    # V index: j*p + k
                    v_vec[gj * p + gk] = 1
                    # W index: k*n + i
                    w_vec[gk * n + gi] = 1
                    
                    terms_U.append(u_vec)
                    terms_V.append(v_vec)
                    terms_W.append(w_vec)
                    
    def add_strassen_block(start_i, start_j, start_k):
        """
        Adds Strassen terms for a 2x2x2 block.
        start_i, start_j, start_k are the global starting indices.
        Local indices 0,1 map to start+0, start+1.
        """
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2, C00 -= M2
        # M3 = A00(B01-B11) -> C01 += M3, C11 -= M3
        # M4 = A11(B10-B00) -> C10 += M4, C00 += M4
        # M5 = (A00+A10)B11 -> C01 += M5, C11 += M5
        # M6 = (A11-A10)(B00+B10) -> C00 += M6, C10 -= M6
        # M7 = (A01-A11)(B11+B01) -> C11 -= M7
        
        # We need to generate U, V, W vectors for each of the 7 products.
        # Each product is a bilinear form.
        # For M1: U has 1s at (0,0) and (1,1). V has 1s at (0,0) and (1,1).
        # W has 1s at (0,0) and (1,1) corresponding to C00 and C11.
        # Wait, the W vector for a rank-1 term T = U ⊗ V ⊗ W is such that
        # T[i,j,k] = U[i,j] V[j,k] W[k,i].
        # In Strassen, the "product" M is computed as dot(U_part, V_part).
        # Then M is added to specific C entries.
        # This corresponds to W having non-zero entries at those C positions.
        
        def make_vecs(u_coeffs, v_coeffs, w_coeffs):
            """
            u_coeffs: dict mapping (i,j) to coeff
            v_coeffs: dict mapping (j,k) to coeff
            w_coeffs: dict mapping (k,i) to coeff
            """
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for (li, lj), c in u_coeffs.items():
                gi = start_i + li
                gj = start_j + lj
                u_vec[gi * m + gj] = c
                
            for (lj, lk), c in v_coeffs.items():
                gj = start_j + lj
                gk = start_k + lk
                v_vec[gj * p + gk] = c
                
            for (lk, li), c in w_coeffs.items():
                gk = start_k + lk
                gi = start_i + li
                w_vec[gk * n + gi] = c
                
            terms_U.append(u_vec)
            terms_V.append(v_vec)
            terms_W.append(w_vec)

        # M1: (A00+A11)(B00+B11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: C00 (k=0,i=0)=1, C11 (k=1,i=1)=1
        make_vecs(
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )
        
        # M2: (A10+A11)B00
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: C10 (k=0,i=1)=1, C00 (k=0,i=0)=-1
        make_vecs(
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (0,0): -1}
        )
        
        # M3: A00(B01-B11)
        # U: (0,0)=1
        # V: (0,1)=1, (1,1)=-1
        # W: C01 (k=1,i=0)=1, C11 (k=1,i=1)=-1
        make_vecs(
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): -1}
        )
        
        # M4: A11(B10-B00)
        # U: (1,1)=1
        # V: (1,0)=1, (0,0)=-1
        # W: C10 (k=0,i=1)=1, C00 (k=0,i=0)=1
        make_vecs(
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (0,0): 1}
        )
        
        # M5: (A00+A10)B11
        # U: (0,0)=1, (1,0)=1
        # V: (1,1)=1
        # W: C01 (k=1,i=0)=1, C11 (k=1,i=1)=1
        make_vecs(
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(1,0): 1, (1,1): 1}
        )
        
        # M6: (A11-A10)(B00+B10)
        # U: (1,1)=1, (1,0)=-1
        # V: (0,0)=1, (1,0)=1
        # W: C00 (k=0,i=0)=1, C10 (k=0,i=1)=-1
        make_vecs(
            {(1,1): 1, (1,0): -1},
            {(0,0): 1, (1,0): 1},
            {(0,0): 1, (0,1): -1}
        )
        
        # M7: (A01-A11)(B11+B01)
        # U: (0,1)=1, (1,1)=-1
        # V: (1,1)=1, (0,1)=1
        # W: C11 (k=1,i=1)=-1
        make_vecs(
            {(0,1): 1, (1,1): -1},
            {(1,1): 1, (0,1): 1},
            {(1,1): -1}
        )

    # Define blocks
    # Indices: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    blocks = [
        (0, 2),
        (2, 2),
        (4, 1)
    ]
    
    for a, (start_i, size_i) in enumerate(blocks):
        for b, (start_j, size_j) in enumerate(blocks):
            for c, (start_k, size_k) in enumerate(blocks):
                if size_i == 2 and size_j == 2 and size_k == 2:
                    add_strassen_block(start_i, start_j, start_k)
                else:
                    add_schoolbook_block(start_i, size_i, start_j, size_j, start_k, size_k)
    
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = terms_U[r][i]
        for i in range(m * p):
            V[i][r] = terms_V[r][i]
        for i in range(p * n):
            W[i][r] = terms_W[r][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
