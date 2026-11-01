"""Optimized matrix multiplication 2x2 x 2x13 using Strassen for column pairs."""
DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add a single multiplication term to U, V, W columns."""
        # u_coeffs: dict {(i,j): coeff} for A entries
        # v_coeffs: dict {(j,k): coeff} for B entries
        # w_coeffs: dict {(i,k): coeff} for C entries
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        
        for (i, j), c in u_coeffs.items():
            U_row[i * m + j] = c
        for (j, k), c in v_coeffs.items():
            V_row[j * p + k] = c
        for (i, k), c in w_coeffs.items():
            W_row[k * n + i] = c
        
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # Process 6 pairs of columns (0-1, 2-3, 4-5, 6-7, 8-9, 10-11)
    for pair in range(6):
        k0 = pair * 2      # first column in pair
        k1 = pair * 2 + 1  # second column in pair
        
        # Strassen's 7 multiplications for 2x2 block
        # M1 = (a00+a11)(b00+b11)
        add_term(
            {(0,0): 1, (1,1): 1},
            {(0, k0): 1, (1, k1): 1},
            {}
        )
        # M2 = (a10+a11)b00
        add_term(
            {(1,0): 1, (1,1): 1},
            {(0, k0): 1},
            {}
        )
        # M3 = a00(b10-b11)
        add_term(
            {(0,0): 1},
            {(1, k0): 1, (1, k1): -1},
            {}
        )
        # M4 = a11(b01-b00)
        add_term(
            {(1,1): 1},
            {(0, k1): 1, (0, k0): -1},
            {}
        )
        # M5 = (a00+a01)b11
        add_term(
            {(0,0): 1, (0,1): 1},
            {(1, k1): 1},
            {}
        )
        # M6 = (a10-a00)(b00+b01)
        add_term(
            {(1,0): 1, (0,0): -1},
            {(0, k0): 1, (0, k1): 1},
            {}
        )
        # M7 = (a01-a11)(b10+b11)
        add_term(
            {(0,1): 1, (1,1): -1},
            {(1, k0): 1, (1, k1): 1},
            {}
        )
        
        # Now add the W coefficients for reconstruction
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        # We need to add W terms that combine these
        
        # For c00 (i=0, k=k0): +M1 +M4 -M5 +M7
        # For c01 (i=0, k=k1): +M3 +M5
        # For c10 (i=1, k=k0): +M2 +M4
        # For c11 (i=1, k=k1): +M1 -M2 +M3 +M6
        
        # Since we already added the U,V for each M, we need to add W entries
        # that reference which M contributes to which output
        # Actually, in the tensor decomposition, each column of U,V,W corresponds
        # to one multiplication. The W column should have 1s where that product
        # contributes to the output.
        
        # Let me redo this: each Strassen term M_r contributes to multiple outputs.
        # The W vector for term r should have coefficients indicating how M_r 
        # appears in the output.
        
        pass  # This approach needs restructuring
    
    # Let me restructure: build U, V, W as full matrices
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for pairs
    for pair in range(6):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1 = (a00+a11)(b00+b11)
        # Contributes: c00 += 1*M1, c11 += 1*M1
        U[0*2+0][col] = 1   # a00
        U[1*2+1][col] = 1   # a11
        V[0*13+k0][col] = 1 # b0,k0
        V[1*13+k1][col] = 1 # b1,k1
        W[k0*2+0][col] = 1  # c0,k0
        W[k1*2+1][col] = 1  # c1,k1
        col += 1
        
        # M2 = (a10+a11)b00
        # Contributes: c10 += 1*M2, c11 += -1*M2
        U[1*2+0][col] = 1   # a10
        U[1*2+1][col] = 1   # a11
        V[0*13+k0][col] = 1 # b0,k0
        W[k0*2+1][col] = 1  # c1,k0
        W[k1*2+1][col] = -1 # c1,k1
        col += 1
        
        # M3 = a00(b10-b11)
        # Contributes: c01 += 1*M3, c11 += 1*M3
        U[0*2+0][col] = 1   # a00
        V[1*13+k0][col] = 1 # b1,k0
        V[1*13+k1][col] = -1# b1,k1
        W[k1*2+0][col] = 1  # c0,k1
        W[k1*2+1][col] = 1  # c1,k1
        col += 1
        
        # M4 = a11(b01-b00)
        # Contributes: c00 += 1*M4, c10 += 1*M4
        U[1*2+1][col] = 1   # a11
        V[0*13+k1][col] = 1 # b0,k1
        V[0*13+k0][col] = -1# b0,k0
        W[k0*2+0][col] = 1  # c0,k0
        W[k0*2+1][col] = 1  # c1,k0
        col += 1
        
        # M5 = (a00+a01)b11
        # Contributes: c00 += -1*M5, c01 += 1*M5
        U[0*2+0][col] = 1   # a00
        U[0*2+1][col] = 1   # a01
        V[1*13+k1][col] = 1 # b1,k1
        W[k0*2+0][col] = -1 # c0,k0
        W[k1*2+0][col] = 1  # c0,k1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        # Contributes: c11 += 1*M6
        U[1*2+0][col] = 1   # a10
        U[0*2+0][col] = -1  # a00
        V[0*13+k0][col] = 1 # b0,k0
        V[0*13+k1][col] = 1 # b0,k1
        W[k1*2+1][col] = 1  # c1,k1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        # Contributes: c00 += 1*M7
        U[0*2+1][col] = 1   # a01
        U[1*2+1][col] = -1  # a11
        V[1*13+k0][col] = 1 # b1,k0
        V[1*13+k1][col] = 1 # b1,k1
        W[k0*2+0][col] = 1  # c0,k0
        col += 1
    
    # Last column k=12: schoolbook
    k = 12
    # c_i,k = sum_j a_i,j * b_j,k
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*13+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
