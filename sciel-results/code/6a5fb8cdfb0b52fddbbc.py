"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 60 decomposition derived from:
    # m=4 -> 2+2
    # p=6 -> 3+3
    # For each (m_block, p_block):
    #   n=3 -> 1+2
    #   n=1 part: 1x2x3 (Rank 6)
    #   n=2 part: 2x2x3 -> p=3 -> 1+2
    #     p=1 part: 2x2x1 (Rank 2)
    #     p=2 part: 2x2x2 (Rank 7, Strassen)
    # Total Rank = 2 * ( 2 * (6 + 2 + 7) ) = 60
    
    R = 60
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will iterate through the blocks
    
    for bm in range(2): # m blocks: 0, 1
        m_start = bm * 2
        for bp in range(2): # p blocks: 0, 1
            p_start = bp * 3
            
            # Part 1: n=1 (global n=0)
            # A is 1x2, B is 2x3
            # Rank 6
            for j in range(2):
                for k in range(3):
                    u_idx = 0 * m + (m_start + j) # n=0
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + 0
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)
                    col += 1
            
            # Part 2: n=2 (global n=1,2)
            # A is 2x2, B is 2x3
            # Split p into 1+2
            
            # Part 2a: p=1 (global p=p_start)
            # A is 2x2, B is 2x1
            # Rank 2
            for j in range(2):
                for i in range(2):
                    u_idx = (1 + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + p_start
                    w_idx = p_start * n + (1 + i)
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)
                    col += 1
            
            # Part 2b: p=2 (global p=p_start+1, p_start+2)
            # A is 2x2, B is 2x2
            # Rank 7 (Strassen)
            
            # Indices for A (2x2):
            # A[0,0] -> n=1, m=m_start
            # A[0,1] -> n=1, m=m_start+1
            # A[1,0] -> n=2, m=m_start
            # A[1,1] -> n=2, m=m_start+1
            
            # Indices for B (2x2):
            # B[0,0] -> m=m_start, p=p_start+1
            # B[0,1] -> m=m_start, p=p_start+2
            # B[1,0] -> m=m_start+1, p=p_start+1
            # B[1,1] -> m=m_start+1, p=p_start+2
            
            # Strassen M1: (A00+A11)(B00+B11)
            # A00: n=1, m=m_start
            # A11: n=2, m=m_start+1
            # B00: m=m_start, p=p_start+1
            # B11: m=m_start+1, p=p_start+2
            # This is a single multiplication term? No, Strassen combines them.
            # The decomposition adds 7 terms, each corresponding to one multiplication.
            # For each term, U and V contain the linear combinations of inputs.
            
            # Term 1: M1
            # U: A00 (1), A11 (1)
            # V: B00 (1), B11 (1)
            # W: C00 (1), C01 (0), C10 (0), C11 (-1)? No, W is the output coefficient.
            # Wait, the decomposition is:
            # C = sum_k U_k * V_k * W_k
            # Where U_k, V_k are vectors, W_k is a tensor (or matrix in this case).
            # Actually, W[k*n+i, r] is the coefficient for output C[i, k] (or similar).
            # In Strassen, we have 7 multiplications.
            # For each multiplication k, we add a rank-1 tensor.
            # The rank-1 tensor is $u \otimes v \otimes w$.
            # $u$ is the vector of A coefficients.
            # $v$ is the vector of B coefficients.
            # $w$ is the vector of C coefficients.
            
            # Let's define the 7 Strassen terms explicitly.
            
            # Term 1: (A00+A11) * (B00+B11)
            # U: A00=1, A11=1
            # V: B00=1, B11=1
            # W: C00=1, C01=0, C10=0, C11=-1 (Wait, C11 = M1 - M2 + M7. So coeff for M1 in C11 is 1? No, C11 = M1 - M2 + M7. So coeff is 1. C00 = M1 - M4 + M5. Coeff 1. C01 = M3 + M4. Coeff 0. C10 = M2 + M6. Coeff 0.)
            # Correction:
            # C00 = M1 - M4 + M5
            # C01 = M3 + M4
            # C10 = M2 + M6
            # C11 = M1 - M2 + M7
            
            # So for M1:
            # W coeffs: C00=1, C01=0, C10=0, C11=1
            
            def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
                # u_coeffs: dict of (n_local, m_local) -> val
                # v_coeffs: dict of (m_local, p_local) -> val
                # w_coeffs: dict of (n_local, p_local) -> val
                # Local indices: n in 0,1; m in 0,1; p in 0,1
                # Global mapping:
                # n_global = 1 + n_local
                # m_global = m_start + m_local
                # p_global = p_start + 1 + p_local (since this is the p=2 part, starting at p_start+1)
                
                # We add to the current column
                for (nl, ml), uv in u_coeffs.items():
                    u_idx = (1 + nl) * m + (m_start + ml)
                    U[u_idx][col] += uv
                for (ml, pl), vv in v_coeffs.items():
                    v_idx = (m_start + ml) * p + (p_start + 1 + pl)
                    V[v_idx][col] += vv
                for (nl, pl), ww in w_coeffs.items():
                    w_idx = (p_start + 1 + pl) * n + (1 + nl)
                    W[w_idx][col] += ww
            
            # M1: (A00+A11)(B00+B11)
            add_strassen_term(
                {(0,0):1, (1,1):1},
                {(0,0):1, (1,1):1},
                {(0,0):1, (1,1):1}
            )
            col += 1
            
            # M2: (A10+A11)B00
            add_strassen_term(
                {(1,0):1, (1,1):1},
                {(0,0):1},
                {(1,0):1} # C10 gets M2
            )
            col += 1
            
            # M3: A00(B10-B11)
            add_strassen_term(
                {(0,0):1},
                {(1,0):1, (1,1):-1},
                {(0,1):1} # C01 gets M3
            )
            col += 1
            
            # M4: A11(B01-B00)
            add_strassen_term(
                {(1,1):1},
                {(0,1):1, (0,0):-1},
                {(0,0):-1, (0,1):1} # C00 gets -M4, C01 gets M4
            )
            col += 1
            
            # M5: (A00+A01)B11
            add_strassen_term(
                {(0,0):1, (0,1):1},
                {(1,1):1},
                {(0,0):1} # C00 gets M5
            )
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            add_strassen_term(
                {(1,0):1, (0,0):-1},
                {(0,0):1, (0,1):1},
                {(1,0):1} # C10 gets M6
            )
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            add_strassen_term(
                {(0,1):1, (1,1):-1},
                {(1,0):1, (1,1):1},
                {(1,1):1} # C11 gets M7
            )
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
