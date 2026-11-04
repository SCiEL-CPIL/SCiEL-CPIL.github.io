import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # We will build U, V, W as lists of lists.
    # U shape: (n*m, Rank) = (16, Rank)
    # V shape: (m*p, Rank) = (20, Rank)
    # W shape: (p*n, Rank) = (20, Rank)
    
    # Part 1: 4x4x4 multiplication using 8 Strassen 2x2x2 blocks. Rank 56.
    # Part 2: 4x4x1 multiplication using schoolbook. Rank 16.
    # Total Rank = 72.
    
    R = 72
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # Strassen formulas for C = A*B where A, B, C are 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map this to the global indices.
    # A block (br, bc) corresponds to A[br*2:br*2+2, bc*2:bc*2+2]
    # Global index for A[i, j] is i*m + j.
    # For 4x4, m=4.
    
    def add_strassen_block(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds a Strassen decomposition for the 2x2 block multiplication.
        A block is at rows A_base_i..A_base_i+1, cols A_base_j..A_base_j+1.
        B block is at rows B_base_j..B_base_j+1, cols B_base_k..B_base_k+1.
        C block is at rows C_base_k..C_base_k+1, cols C_base_i..C_base_i+1.
        
        Note: The tensor interface is:
        U[i*m+j, c]
        V[j*p+k, c]
        W[k*n+i, c]
        
        For a 2x2 block at A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[k0:k0+2, i0:i0+2]:
        We need to add contributions to U, V, W.
        """
        nonlocal col
        
        # Strassen variables M1..M7
        # Each M is a linear combination of A and B entries.
        # We need to add 7 rank-1 terms.
        
        # M1 = (A00 + A11)(B00 + B11)
        # U part: A00, A11
        # V part: B00, B11
        # W part: C00 (coef 1), C11 (coef 1)
        
        def add_term(A_coefs, B_coefs, C_coefs):
            nonlocal col
            # A_coefs: list of (global_idx, coeff)
            # B_coefs: list of (global_idx, coeff)
            # C_coefs: list of (global_idx, coeff)
            
            # Combine A terms into U
            # Since U is a matrix, we just add the coefficients to the specific row/col
            # But we need to handle the tensor structure:
            # T = sum_c U_c \otimes V_c \otimes W_c
            # This means for a fixed c, U[:, c] is a vector, V[:, c] is a vector, W[:, c] is a vector.
            # The term is U[:, c] V[:, c] W[:, c].
            # This corresponds to the outer product.
            
            # We just set the non-zero entries in U, V, W for column col.
            
            for idx, coeff in A_coefs:
                U[idx][col] += coeff
            for idx, coeff in B_coefs:
                V[idx][col] += coeff
            for idx, coeff in C_coefs:
                W[idx][col] += coeff
                
            col += 1

        # Indices for the 2x2 blocks
        # A: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # Global A idx: i*m + j
        a00 = A_base_i * m + A_base_j
        a01 = A_base_i * m + A_base_j + 1
        a10 = (A_base_i + 1) * m + A_base_j
        a11 = (A_base_i + 1) * m + A_base_j + 1
        
        # B: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # Global B idx: j*p + k
        b00 = B_base_j * p + B_base_k
        b01 = B_base_j * p + B_base_k + 1
        b10 = (B_base_j + 1) * p + B_base_k
        b11 = (B_base_j + 1) * p + B_base_k + 1
        
        # C: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        # Global C idx: k*n + i
        c00 = C_base_k * n + C_base_i
        c01 = C_base_k * n + C_base_i + 1
        c10 = (C_base_k + 1) * n + C_base_i
        c11 = (C_base_k + 1) * n + C_base_i + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            [(a00, 1), (a11, 1)],
            [(b00, 1), (b11, 1)],
            [(c00, 1), (c11, 1)]
        )
        
        # M2 = (A10 + A11)B00
        add_term(
            [(a10, 1), (a11, 1)],
            [(b00, 1)],
            [(c10, 1)]
        )
        
        # M3 = A00(B01 - B11)
        add_term(
            [(a00, 1)],
            [(b01, 1), (b11, -1)],
            [(c01, 1)]
        )
        
        # M4 = A11(B10 - B00)
        add_term(
            [(a11, 1)],
            [(b10, 1), (b00, -1)],
            [(c00, 1), (c10, 1)]
        )
        
        # M5 = (A00 + A01)B11
        add_term(
            [(a00, 1), (a01, 1)],
            [(b11, 1)],
            [(c00, -1), (c01, 1)]
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            [(a10, 1), (a00, -1)],
            [(b00, 1), (b01, 1)],
            [(c11, 1)]
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            [(a01, 1), (a11, -1)],
            [(b10, 1), (b11, 1)],
            [(c00, 1), (c11, -1)]
        )

    # Define the 8 blocks for 4x4x4
    # A blocks: (0,0), (0,1), (1,0), (1,1) in terms of 2x2 blocks
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # C00 = A00 B00 + A01 B10
    add_strassen_block(0, 0, 0, 0, 0, 0) # A00 B00 -> C00
    add_strassen_block(0, 2, 2, 0, 0, 0) # A01 B10 -> C00
    
    # C01 = A00 B01 + A01 B11
    add_strassen_block(0, 0, 0, 2, 0, 2) # A00 B01 -> C01
    add_strassen_block(0, 2, 2, 2, 0, 2) # A01 B11 -> C01
    
    # C10 = A10 B00 + A11 B10
    add_strassen_block(2, 0, 0, 0, 2, 0) # A10 B00 -> C10
    add_strassen_block(2, 2, 2, 0, 2, 0) # A11 B10 -> C10
    
    # C11 = A10 B01 + A11 B11
    add_strassen_block(2, 0, 0, 2, 2, 2) # A10 B01 -> C11
    add_strassen_block(2, 2, 2, 2, 2, 2) # A11 B11 -> C11
    
    # Part 2: 4x4x1 for the last column (index 4 in B and C)
    # C[:, 4] = A B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # This is rank 16 (schoolbook).
    for i in range(n):
        for j in range(m):
            # U[i*m + j, col] = 1
            # V[j*p + 4, col] = 1
            # W[4*n + i, col] = 1
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

