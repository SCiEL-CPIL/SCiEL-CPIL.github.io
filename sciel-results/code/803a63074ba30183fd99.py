import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # We will build the decomposition term by term.
    # Each term is a triple (u_vec, v_vec, w_vec) where u_vec is size n*m, v_vec size m*p, w_vec size p*n.
    # We store them in lists and then assemble matrices.
    
    terms = []
    
    def add_term(u_indices, v_indices, w_indices):
        """
        u_indices: dict {global_idx: coeff}
        v_indices: dict {global_idx: coeff}
        w_indices: dict {global_idx: coeff}
        """
        terms.append((u_indices, v_indices, w_indices))
        
    def add_strassen_block(i_offset, j_offset, k_offset):
        """
        Adds a 2x2x2 Strassen block.
        i_offset: offset for i in A (0 or 1)
        j_offset: offset for j in A/B (0..3)
        k_offset: offset for k in B (0..4)
        
        A rows: i_offset, i_offset+1
        A cols: j_offset, j_offset+1
        B rows: j_offset, j_offset+1
        B cols: k_offset, k_offset+1
        C rows: i_offset, i_offset+1
        C cols: k_offset, k_offset+1
        """
        # Local indices
        # A: a00, a01, a10, a11
        # Global A idx: (i, j) -> i*m + j
        a00 = i_offset * m + j_offset
        a01 = i_offset * m + (j_offset + 1)
        a10 = (i_offset + 1) * m + j_offset
        a11 = (i_offset + 1) * m + (j_offset + 1)
        
        # B: b00, b01, b10, b11
        # Global B idx: (j, k) -> j*p + k
        b00 = j_offset * p + k_offset
        b01 = j_offset * p + (k_offset + 1)
        b10 = (j_offset + 1) * p + k_offset
        b11 = (j_offset + 1) * p + (k_offset + 1)
        
        # C: c00, c01, c10, c11
        # Global C idx: (k, i) -> k*n + i
        c00 = k_offset * n + i_offset
        c01 = k_offset * n + (i_offset + 1)
        c10 = (k_offset + 1) * n + i_offset
        c11 = (k_offset + 1) * n + (i_offset + 1)
        
        # Strassen Terms
        # M1 = (a00 + a11)(b00 + b11)
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1} # Wait, C contributions are sums. 
                             # We need to add to W matrix.
                             # My add_term adds a full rank-1 tensor.
                             # So for M1, it contributes to c00, c11?
                             # No, M1 is a scalar.
                             # The tensor term is u \otimes v \otimes w.
                             # w is the vector of C coefficients for this term.
                             # For M1, w has 1 at c00 and 1 at c11?
                             # Let's check C formula.
                             # C00 = M1 + M4 - M5 + M7
                             # C01 = M3 + M5
                             # C10 = M2 + M4
                             # C11 = M1 - M2 + M3 + M6
                             
                             # So M1 contributes +1 to C00 and +1 to C11.
                             # So w = {c00: 1, c11: 1}.
                             # Yes.
        )
        
        # M2 = (a10 + a11) b00
        add_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1}
        )
        
        # M3 = a00 (b01 - b11)
        add_term(
            {a00: 1},
            {b01: 1, b11: -1},
            {c01: 1, c11: 1}
        )
        
        # M4 = a11 (b10 - b00)
        add_term(
            {a11: 1},
            {b10: 1, b00: -1},
            {c00: 1, c10: 1}
        )
        
        # M5 = (a00 + a01) b11
        add_term(
            {a00: 1, a01: 1},
            {b11: 1},
            {c00: -1, c01: 1}
        )
        
        # M6 = (a10 - a00) (b00 + b01)
        add_term(
            {a10: 1, a00: -1},
            {b00: 1, b01: 1},
            {c11: 1}
        )
        
        # M7 = (a01 - a11) (b10 + b11)
        add_term(
            {a01: 1, a11: -1},
            {b10: 1, b11: 1},
            {c00: 1, c11: -1}
        )
        
    def add_trivial_block(i_offset, j_indices, k_indices):
        """
        Adds trivial terms for A[i, j] * B[j, k] -> C[i, k]
        """
        for i in range(i_offset, i_offset + 2): # n=2
            for j in j_indices:
                for k in k_indices:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    add_term(
                        {a_idx: 1},
                        {b_idx: 1},
                        {c_idx: 1}
                    )

    # Block 1: j in {0, 1}
    # Split k into {0,1}, {2,3}, {4,5}
    add_strassen_block(0, 0, 0) # i=0, j=0, k=0
    add_strassen_block(0, 0, 2) # i=0, j=0, k=2
    add_strassen_block(0, 0, 4) # i=0, j=0, k=4
    
    # Block 2: j in {2, 3, 4}
    # Split j into {2,3} and {4}
    
    # Sub-block 2.1: j in {2, 3}
    # Split k into {0,1}, {2,3}, {4,5}
    add_strassen_block(0, 2, 0) # i=0, j=2, k=0
    add_strassen_block(0, 2, 2) # i=0, j=2, k=2
    add_strassen_block(0, 2, 4) # i=0, j=2, k=4
    
    # Sub-block 2.2: j in {4}
    # Trivial for all k
    add_trivial_block(0, [4], [0, 1, 2, 3, 4, 5])

    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
