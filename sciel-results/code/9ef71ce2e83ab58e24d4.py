import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We assume n=m=p for this task
    U, V, W = strassen_decomp(n)
    # Convert to list of lists as required by interface? 
    # The interface says "factor matrices of shapes ... with integer/half-integer entries"
    # The seed returns lists of lists. I should probably return lists of lists or numpy arrays.
    # The seed returns lists. I will return lists to be safe, or check if numpy is accepted.
    # "Python standard library, NumPy and SciPy are available."
    # The seed returns lists. I will return lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

def strassen_decomp(n):
    if n == 1:
        U = np.array([[1]])
        V = np.array([[1]])
        W = np.array([[1]])
        return U, V, W
    
    if n == 2:
        # Rank 7 decomposition
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: 0:00, 1:01, 2:10, 3:11
        # M1 = (A00+A11)(B00+B11)
        # U col 0: A00, A11 -> 1, 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # W: C00 += M1, C11 += M1
        # C00 is W[0, 0] (k=0, i=0)? 
        # Convention W[k*n+i]. n=2.
        # C00 -> k=0, i=0 -> idx 0.
        # C11 -> k=1, i=1 -> idx 3.
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 += M2, C11 -= M2
        # C10 -> k=1, i=0 -> idx 2.
        # C11 -> k=1, i=1 -> idx 3.
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += M3, C11 += M3
        # C01 -> k=0, i=1 -> idx 1.
        # C11 -> k=1, i=1 -> idx 3.
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 += M4, C10 += M4
        # C00 -> idx 0.
        # C10 -> idx 2.
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00 -= M5, C01 += M5
        # C00 -> idx 0.
        # C01 -> idx 1.
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C11 += M6
        # C11 -> idx 3.
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C00 += M7
        # C00 -> idx 0.
        W[0, 6] = 1
        
        return U, V, W

    # Recursive case
    m = n // 2
    U_m, V_m, W_m = strassen_decomp(m)
    rank_m = U_m.shape[1]
    rank_n = 7 * rank_m
    
    U = np.zeros((n*n, rank_n))
    V = np.zeros((n*n, rank_n))
    W = np.zeros((n*n, rank_n))
    
    # Strassen coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # M1: (A00+A11)(B00+B11)
    # A coeffs: A00:1, A11:1
    # B coeffs: B00:1, B11:1
    # C coeffs: C00:1, C11:1
    
    # Helper to add a term
    def add_term(r, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict {(a_block, a_idx): coeff}
        # Actually, a_coeffs is a 2x2 matrix of coefficients for A blocks
        # a_coeffs[ab] is the coefficient for block A_ab
        for a_block in range(2):
            for b_block in range(2):
                coeff_a = a_coeffs[a_block, b_block]
                if coeff_a == 0: continue
                for i2 in range(m):
                    for j2 in range(m):
                        # U index: (a_block, i2), (b_block, j2)
                        u_idx = (a_block * m + i2) * (2 * m) + (b_block * m + j2)
                        # Wait, U index is i*m + j.
                        # i = a_block * m + i2
                        # j = b_block * m + j2
                        u_idx = (a_block * m + i2) * m + (b_block * m + j2) # Wrong, n=2m. i ranges 0..2m-1.
                        # i = a_block * m + i2. j = b_block * m + j2.
                        # U index = i * m + j? No, U shape is (n*m, rank). n=m=p=4.
                        # U index = i * m + j.
                        u_idx = (a_block * m + i2) * m + (b_block * m + j2)
                        
                        # Recursive U index: i2 * m + j2
                        u_rec_idx = i2 * m + j2
                        U[u_idx, r * rank_m + s] += coeff_a * U_m[u_rec_idx, s] # This loop is over s?
                        # No, s is the rank index of the recursive call.
                        # We need to loop over s.
                        
    # This approach is getting messy with nested loops. 
    # Let's use matrix operations.
    
    # For each Strassen term r:
    # A_r = sum_{ab} a_coeffs[ab] A_ab
    # B_r = sum_{bc} b_coeffs[bc] B_bc
    # C_r = A_r B_r
    # C = sum_r sum_{ba} c_coeffs[ba] C_r[ba]
    
    # U_full[:, r*rank_m + s] is formed by combining U_m[:, s] with the block coefficients.
    
    for r in range(7):
        # Define coefficients for A, B, C blocks
        if r == 0: # M1
            a_c = np.array([[1, 0], [0, 1]])
            b_c = np.array([[1, 0], [0, 1]])
            c_c = np.array([[1, 0], [0, 1]])
        elif r == 1: # M2
            a_c = np.array([[0, 0], [1, 1]])
            b_c = np.array([[1, 0], [0, 0]])
            c_c = np.array([[0, 0], [1, -1]]) # C10 +=, C11 -=
        elif r == 2: # M3
            a_c = np.array([[1, 0], [0, 0]])
            b_c = np.array([[0, 1], [0, -1]])
            c_c = np.array([[0, 1], [0, 1]]) # C01 +=, C11 +=
        elif r == 3: # M4
            a_c = np.array([[0, 0], [0, 1]])
            b_c = np.array([[0, -1], [1, 0]])
            c_c = np.array([[1, 0], [1, 0]]) # C00 +=, C10 +=
        elif r == 4: # M5
            a_c = np.array([[1, 1], [0, 0]])
            b_c = np.array([[0, 0], [0, 1]])
            c_c = np.array([[-1, 1], [0, 0]]) # C00 -=, C01 +=
        elif r == 5: # M6
            a_c = np.array([[-1, 0], [1, 0]])
            b_c = np.array([[1, 1], [0, 0]])
            c_c = np.array([[0, 0], [0, 1]]) # C11 +=
        elif r == 6: # M7
            a_c = np.array([[0, 1], [0, -1]])
            b_c = np.array([[0, 0], [1, 1]])
            c_c = np.array([[1, 0], [0, 0]]) # C00 +=
            
        # For each recursive rank s:
        for s in range(rank_m):
            col = r * rank_m + s
            
            # U construction
            # U_full[i, j, col] = sum_{ab} a_c[ab] U_m[i2, j2, s] where i = a*m+i2, j = b*m+j2
            # We can do this by reshaping.
            # U_m is (m*m, rank_m). Take column s.
            # u_m = U_m[:, s] -> shape (m*m,)
            # Reshape to (m, m)
            u_m = U_m[:, s].reshape(m, m)
            
            # Create U_block of shape (2m, 2m)
            # U_block[a*m+i2, b*m+j2] = a_c[a, b] * u_m[i2, j2]
            U_block = np.zeros((2*m, 2*m))
            for a in range(2):
                for b in range(2):
                    U_block[a*m:(a+1)*m, b*m:(b+1)*m] = a_c[a, b] * u_m
            
            U[:, col] = U_block.flatten()
            
            # V construction
            v_m = V_m[:, s].reshape(m, m)
            V_block = np.zeros((2*m, 2*m))
            for b in range(2):
                for c in range(2):
                    V_block[b*m:(b+1)*m, c*m:(c+1)*m] = b_c[b, c] * v_m
            V[:, col] = V_block.flatten()
            
            # W construction
            w_m = W_m[:, s].reshape(m, m)
            W_block = np.zeros((2*m, 2*m))
            # W indices are (k, i). Block indices (b, a).
            # W_block[k, i] where k = b*m+k2, i = a*m+i2.
            # W_block[b*m+k2, a*m+i2] = c_c[b, a] * w_m[k2, i2]
            for b in range(2):
                for a in range(2):
                    W_block[b*m:(b+1)*m, a*m:(a+1)*m] = c_c[b, a] * w_m
            W[:, col] = W_block.flatten()
            
    return U, V, W
