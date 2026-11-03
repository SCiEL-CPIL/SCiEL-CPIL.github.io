def construct():
    n, m, p = 2, 3, 8
    # Rank calculation:
    # Two 2x3x4 blocks.
    # Each 2x3x4 block:
    #   2x2x2 (Strassen, rank 7) for A_L * B_T1
    #   2x2x2 (Strassen, rank 7) for A_L * B_T2
    #   2x1x4 (Schoolbook, rank 8) for A_R * B_B
    #   Total per block: 22
    # Total rank: 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal U, V, W
        # Find next available rank? No, we manage rank manually.
        pass # Placeholder

    # We need to manage the rank index.
    # Let's use a list of tuples (u, v, w, val) and then populate.
    # Or just populate directly with a counter.
    
    current_rank = 0
    
    def add_bilinear(u_idx, v_idx, w_idx, val=1):
        nonlocal current_rank
        if current_rank >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][current_rank] += val
        V[v_idx][current_rank] += val
        W[w_idx][current_rank] += val
        current_rank += 1

    def add_strassen(a_idx, b_idx, c_idx):
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # m1 = (a00 + a11) * (b00 + b11)
        # We need to expand this into bilinear terms?
        # No, Strassen terms are bilinear in A and B.
        # m1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # This is 4 terms.
        # Wait, the "rank" of the tensor decomposition counts the number of scalar multiplications.
        # In Strassen, we compute m1, m2, etc. as intermediate values.
        # But the tensor decomposition is a sum of rank-1 tensors.
        # $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
        # If we use Strassen, we are effectively saying that the tensor can be written as a sum of 7 rank-1 tensors?
        # No, Strassen reduces the number of *multiplications* in the algorithm.
        # Does it reduce the *tensor rank*?
        # Yes, the tensor rank of $2 \times 2 \times 2$ multiplication is 7.
        # The decomposition is:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        # Where mr are bilinear forms.
        # $m_1 = (a+d)(e+h) = ae + ah + de + dh$.
        # This $m_1$ is NOT a rank-1 tensor. It is a sum of 4 rank-1 tensors.
        # So if we just sum the $m_r$, we are not using the rank-1 property of the decomposition.
        
        # The rank-1 decomposition of the 2x2x2 multiplication tensor is:
        # We need to find 7 vectors $u_r, v_r, w_r$ such that
        # $C_{ij} = \sum_{r=1}^7 u_{r,i} v_{r,j} w_{r,ij}$? No.
        # $C_{ij} = \sum_{r=1}^7 U_{i,r} V_{j,r} W_{i,j,r}$?
        # The standard form is $C_{i,k} = \sum_r U_{i,r} V_{k,r} W_{i,k,r}$?
        # No, $W$ is the output coefficient.
        # $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$? No.
        # The tensor is $T_{i,j,k}$. $C_{i,k} = \sum_j T_{i,j,k} A_{i,j} B_{j,k}$.
        # Decomposition: $T = \sum_r u_r \otimes v_r \otimes w_r$.
        # $u_r$ is vector in A-space, $v_r$ in B-space, $w_r$ in C-space.
        # $C_{i,k} = \sum_r (\sum_j u_{r,i,j} A_{i,j}) (\sum_k v_{r,j,k} B_{j,k}) w_{r,i,k}$?
        # No.
        # $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$ is not right because indices don't match.
        
        # Correct interface:
        # $U$ shape (n*m, rank). $U[i*m+j, r]$.
        # $V$ shape (m*p, rank). $V[j*p+k, r]$.
        # $W$ shape (p*n, rank). $W[k*n+i, r]$.
        # $C_{i,k} = \sum_r (\sum_j U[i*m+j, r] A_{i,j}) (\sum_l V[j*p+l, r] B_{j,l}) W[k*n+i, r]$?
        # No, the summation over j and l must be consistent for the rank-1 term.
        # A rank-1 term corresponds to a single product of a linear form in A and a linear form in B.
        # $L_A = \sum_{i,j} u_{i,j} A_{i,j}$.
        # $L_B = \sum_{j,k} v_{j,k} B_{j,k}$.
        # Product $P = L_A L_B$.
        # This contributes to $C$ as $\sum_{i,k} w_{i,k} P$.
        # So $C_{i,k} += w_{i,k} P$.
        
        # So for each rank-1 component $r$:
        # We define a vector $u_r$ in $\mathbb{R}^{n \times m}$.
        # We define a vector $v_r$ in $\mathbb{R}^{m \times p}$.
        # We define a vector $w_r$ in $\mathbb{R}^{p \times n}$.
        # The contribution is $w_r \cdot (u_r \cdot A) (v_r \cdot B)$.
        
        # For Strassen, the 7 terms are:
        # $m_1 = (A_{00} + A_{11}) (B_{00} + B_{11})$
        # Here $u_1$ has 1 at (0,0) and (1,1). $v_1$ has 1 at (0,0) and (1,1).
        # $w_1$ has coefficients for $C_{00}, C_{01}, C_{10}, C_{11}$.
        # From $C_{00} = m_1 + \dots$, coeff is 1.
        # From $C_{11} = m_1 + \dots$, coeff is 1.
        # Others 0.
        
        # So yes, we can add these terms.
        
        # Term 1: (a00 + a11) * (b00 + b11)
        # u: a00=1, a11=1
        # v: b00=1, b11=1
        # w: c00=1, c11=1
        
        # Term 2: (a10 + a11) * b00
        # u: a10=1, a11=1
        # v: b00=1
        # w: c10=1, c11=-1
        
        # Term 3: a00 * (b01 - b11)
        # u: a00=1
        # v: b01=1, b11=-1
        # w: c01=1, c11=1
        
        # Term 4: a11 * (b10 - b00)
        # u: a11=1
        # v: b10=1, b00=-1
        # w: c00=1, c10=1
        
        # Term 5: (a00 + a01) * b11
        # u: a00=1, a01=1
        # v: b11=1
        # w: c00=-1, c01=1
        
        # Term 6: (a10 - a00) * (b00 + b01)
        # u: a10=1, a00=-1
        # v: b00=1, b01=1
        # w: c11=1
        
        # Term 7: (a01 - a11) * (b10 + b11)
        # u: a01=1, a11=-1
        # v: b10=1, b11=1
        # w: c00=1, c01? No.
        # Let's re-verify Strassen coefficients.
        
        # Standard Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # C11 (top-left) = m1 + m4 - m5 + m7
        # C12 (top-right) = m3 + m5
        # C21 (bottom-left) = m2 + m4
        # C22 (bottom-right) = m1 - m2 + m3 + m6
        
        # Let's map these to add_bilinear calls.
        
        # m1: u={00:1, 11:1}, v={00:1, 11:1}, w={00:1, 11:1}
        add_strassen_term(a_idx, b_idx, c_idx, 
                          {a_idx[0]: 1, a_idx[2]: 1}, 
                          {b_idx[0]: 1, b_idx[3]: 1}, 
                          {c_idx[0]: 1, c_idx[3]: 1})
        
        # m2: u={10:1, 11:1}, v={00:1}, w={10:1, 11:-1}
        add_strassen_term(a_idx, b_idx, c_idx,
                          {a_idx[2]: 1, a_idx[3]: 1},
                          {b_idx[0]: 1},
                          {c_idx[2]: 1, c_idx[3]: -1})
                          
        # m3: u={00:1}, v={01:1, 11:-1}, w={01:1, 11:1}
        add_strassen_term(a_idx, b_idx, c_idx,
                          {a_idx[0]: 1},
                          {b_idx[1]: 1, b_idx[3]: -1},
                          {c_idx[1]: 1, c_idx[3]: 1})
                          
        # m4: u={11:1}, v={10:1, 00:-1}, w={00:1, 10:1}
        add_strassen_term(a_idx, b_idx, c_idx,
                          {a_idx[3]: 1},
                          {b_idx[2]: 1, b_idx[0]: -1},
                          {c_idx[0]: 1, c_idx[2]: 1})
                          
        # m5: u={00:1, 01:1}, v={11:1}, w={00:-1, 01:1}
        add_strassen_term(a_idx, b_idx, c_idx,
                          {a_idx[0]: 1, a_idx[1]: 1},
                          {b_idx[3]: 1},
                          {c_idx[0]: -1, c_idx[1]: 1})
                          
        # m6: u={10:1, 00:-1}, v={00:1, 01:1}, w={11:1}
        add_strassen_term(a_idx, b_idx, c_idx,
                          {a_idx[2]: 1, a_idx[0]: -1},
                          {b_idx[0]: 1, b_idx[1]: 1},
                          {c_idx[3]: 1})
                          
        # m7: u={01:1, 11:-1}, v={10:1, 11:1}, w={00:1}
        add_strassen_term(a_idx, b_idx, c_idx,
                          {a_idx[1]: 1, a_idx[3]: -1},
                          {b_idx[2]: 1, b_idx[3]: 1},
                          {c_idx[0]: 1})

    # Define add_strassen_term
    def add_strassen_term(a_idx, b_idx, c_idx, u_dict, v_dict, w_dict):
        nonlocal current_rank
        if current_rank >= rank:
            raise ValueError("Rank exceeded")
        
        # u_dict: {global_a_idx: val}
        # v_dict: {global_b_idx: val}
        # w_dict: {global_c_idx: val}
        
        # We need to add this rank-1 component.
        # U[row, rank] = u_val
        # V[col, rank] = v_val
        # W[out, rank] = w_val
        
        for idx, val in u_dict.items():
            U[idx][current_rank] = val
        for idx, val in v_dict.items():
            V[idx][current_rank] = val
        for idx, val in w_dict.items():
            W[idx][current_rank] = val
            
        current_rank += 1

    # Now, build the full decomposition.
    
    # Block 1: B cols 0-3
    # A_L indices: A00, A01, A10, A11 -> 0, 1, 3, 4
    a_L_1 = [0, 1, 3, 4]
    # B_T1 (cols 0-1): B00, B01, B10, B11 -> 0, 1, 8, 9
    b_T1_1 = [0, 1, 8, 9]
    # C (cols 0-1): C00, C01, C10, C11 -> 0, 2, 1, 3
    # W index: k*n + i.
    # C00 (i=0,k=0) -> 0
    # C01 (i=0,k=1) -> 2
    # C10 (i=1,k=0) -> 1
    # C11 (i=1,k=1) -> 3
    c_1_1 = [0, 2, 1, 3]
    
    add_strassen(a_L_1, b_T1_1, c_1_1)
    
    # B_T2 (cols 2-3): B02, B03, B12, B13 -> 2, 3, 10, 11
    b_T2_1 = [2, 3, 10, 11]
    # C (cols 2-3): C02, C03, C12, C13 -> 4, 6, 5, 7
    # C02 (i=0,k=2) -> 4
    # C03 (i=0,k=3) -> 6
    # C12 (i=1,k=2) -> 5
    # C13 (i=1,k=3) -> 7
    c_1_2 = [4, 6, 5, 7]
    
    add_strassen(a_L_1, b_T2_1, c_1_2)
    
    # A_R indices: A02, A12 -> 2, 5
    # B_B (cols 0-3): B20, B21, B22, B23 -> 16, 17, 18, 19
    # Outer product
    for i in range(2):
        for k in range(4):
            a_idx = 3*i + 2
            b_idx = 16 + k
            c_idx = 2*k + i
            add_bilinear(a_idx, b_idx, c_idx, 1)
            
    # Block 2: B cols 4-7
    # A_L same
    # B_T1 (cols 4-5): B04, B05, B14, B15 -> 4, 5, 12, 13
    b_T1_2 = [4, 5, 12, 13]
    # C (cols 4-5): C04, C05, C14, C15 -> 8, 10, 9, 11
    c_2_1 = [8, 10, 9, 11]
    
    add_strassen(a_L_1, b_T1_2, c_2_1)
    
    # B_T2 (cols 6-7): B06, B07, B16, B17 -> 6, 7, 14, 15
    b_T2_2 = [6, 7, 14, 15]
    # C (cols 6-7): C06, C07, C16, C17 -> 12, 14, 13, 15
    c_2_2 = [12, 14, 13, 15]
    
    add_strassen(a_L_1, b_T2_2, c_2_2)
    
    # A_R same
    # B_B (cols 4-7): B24, B25, B26, B27 -> 20, 21, 22, 23
    for i in range(2):
        for k in range(4, 8):
            a_idx = 3*i + 2
            b_idx = 16 + k
            c_idx = 2*k + i
            add_bilinear(a_idx, b_idx, c_idx, 1)

    return {"U": U, "V": V, "W": W}
