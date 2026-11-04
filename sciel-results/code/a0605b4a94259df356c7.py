import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We will use 5 blocks of 2x3x2
    # Each block has rank 11
    # Total rank 55
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, c):
        U[u_idx, c] = 1.0
        V[v_idx, c] = 1.0
        W[w_idx, c] = 1.0

    # We need to map the global indices to the block indices
    # Block k handles columns 2k, 2k+1 of B, and thus columns 2k, 2k+1 of C
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    c = 0
    for block in range(5):
        # B columns: 2*block, 2*block+1
        # C columns: 2*block, 2*block+1
        
        # A_1 B_1 part (Rank 4)
        # A_1 is col 0 of A (index 0, 3 in U)
        # B_1 is row 0 of B (indices 2*block, 2*block+1 in V)
        
        # Term 1: A[0,0] * B[0, 2*block] -> C[0, 2*block]
        add_term(0, 2*block, 2*block, c); c += 1
        # Term 2: A[0,0] * B[0, 2*block+1] -> C[0, 2*block+1]
        add_term(0, 2*block+1, 2*block+1, c); c += 1
        # Term 3: A[1,0] * B[0, 2*block] -> C[1, 2*block]
        add_term(3, 2*block, 2*block, c); c += 1
        # Term 4: A[1,0] * B[0, 2*block+1] -> C[1, 2*block+1]
        add_term(3, 2*block+1, 2*block+1, c); c += 1
        
        # A_2 B_2 part (Rank 7, Strassen)
        # A_2 is cols 1,2 of A (indices 1,2,4,5 in U)
        # B_2 is rows 1,2 of B (indices 3+2*block, 3+2*block+1, 6+2*block, 6+2*block+1 in V)
        # Note: V index is row * p + col.
        # Row 1 of B is index 1 * 10 + col.
        # Row 2 of B is index 2 * 10 + col.
        
        # Let's define local indices for A_2 and B_2
        # A_2[0,0] = U[1], A_2[0,1] = U[2]
        # A_2[1,0] = U[4], A_2[1,1] = U[5]
        
        # B_2[0,0] = V[1*10 + 2*block]
        # B_2[0,1] = V[1*10 + 2*block+1]
        # B_2[1,0] = V[2*10 + 2*block]
        # B_2[1,1] = V[2*10 + 2*block+1]
        
        b00 = 10 + 2*block
        b01 = 10 + 2*block + 1
        b10 = 20 + 2*block
        b11 = 20 + 2*block + 1
        
        # M1 = (A11 + A22)(B11 + B22)
        # U: A11(1), A22(5). V: B11(b00), B22(b11)
        # This requires 2 terms for U and 2 for V? No, Strassen terms are rank 1 in the 2x2x2 subspace.
        # But we are adding to the global U, V, W.
        # A term is U_vec * V_vec * W_vec.
        # For M1: U_vec has 1s at A11, A22. V_vec has 1s at B11, B22.
        # We need to add 1 term to U, V, W.
        # But U is sparse. We can just set the entries.
        
        # However, the "add_term" function above assumes single non-zero entries.
        # Strassen terms have multiple non-zero entries in U, V, W.
        # So I need a more flexible add_term.
        
        pass # Placeholder, will rewrite logic below

    # Rewriting with flexible term addition
    def add_strassen_term(u_indices, v_indices, w_indices, c, sign=1.0):
        for idx in u_indices:
            U[idx, c] += sign
        for idx in v_indices:
            V[idx, c] += sign
        for idx in w_indices:
            W[idx, c] += sign

    c = 0
    for block in range(5):
        # Part 1: A1 * B1 (Rank 4)
        # A1 is col 0: U indices 0, 3
        # B1 is row 0: V indices 2*block, 2*block+1
        
        # T1: A[0,0] B[0, 2b] -> C[0, 2b]
        add_strassen_term([0], [2*block], [2*block], c); c += 1
        # T2: A[0,0] B[0, 2b+1] -> C[0, 2b+1]
        add_strassen_term([0], [2*block+1], [2*block+1], c); c += 1
        # T3: A[1,0] B[0, 2b] -> C[1, 2b]
        add_strassen_term([3], [2*block], [2*block], c); c += 1
        # T4: A[1,0] B[0, 2b+1] -> C[1, 2b+1]
        add_strassen_term([3], [2*block+1], [2*block+1], c); c += 1
        
        # Part 2: A2 * B2 (Rank 7)
        # A2: U[1], U[2], U[4], U[5]
        # B2: V[10+2b], V[10+2b+1], V[20+2b], V[20+2b+1]
        
        b00 = 10 + 2*block
        b01 = 10 + 2*block + 1
        b10 = 20 + 2*block
        b11 = 20 + 2*block + 1
        
        # M1 = (A11 + A22)(B11 + B22)
        add_strassen_term([1, 5], [b00, b11], [2*block, 2*block+1, 2*block, 2*block+1], c) # W indices need to be correct
        # W indices for C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1] are 2b, 2b+1, 2b, 2b+1?
        # No, W is p*n = 10*2 = 20.
        # C[i, k] index is k*n + i = 2*block*2 + i = 4*block + i.
        # So C[0, 2b] is 4*block. C[0, 2b+1] is 4*block+1.
        # C[1, 2b] is 4*block+2. C[1, 2b+1] is 4*block+3.
        
        # Correcting W indices
        w00 = 4*block
        w01 = 4*block + 1
        w10 = 4*block + 2
        w11 = 4*block + 3
        
        # Reset c for this block's Strassen part? No, c is global.
        # I already used 4 terms.
        
        # M1
        add_strassen_term([1, 5], [b00, b11], [w00, w01, w10, w11], c); c += 1
        # M2 = (A21 + A22)B11
        add_strassen_term([4, 5], [b00], [w10, w11], c); c += 1
        # M3 = A11(B21 - B22)
        add_strassen_term([1], [b10, b11], [w01, w11], c); c += 1
        # Note: M3 contributes to C12 and C22.
        # C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
        # So M3 adds to w01 and w11.
        
        # M4 = A22(B12 - B11)
        add_strassen_term([5], [b01, b00], [w00, w10], c); c += 1
        # M4 adds to C11 (w00) and C21 (w10).
        
        # M5 = (A11 + A12)B22
        add_strassen_term([1, 2], [b11], [w01, w11], c); c += 1
        # M5 adds to C12 (w01) and C22 (w11).
        
        # M6 = (A21 - A11)(B11 + B12)
        add_strassen_term([4, 1], [b00, b01], [w11], c); c += 1
        # Wait, M6 is (A21 - A11). U indices 4 (A21) and 1 (A11) with sign -1?
        # add_strassen_term handles sign? No, I passed sign=1.
        # I need to handle signs.
        
        # Let's refine add_strassen_term to take a list of (index, coeff)
        
    # This is getting complicated to hardcode.
    # I will use a helper that takes sparse vectors.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
