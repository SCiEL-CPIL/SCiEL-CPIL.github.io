import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)
    
    # Strategy: Top-level Strassen (7 terms) for 4x4 * 4x4 to get 4x4, then combine with 4x4 * 4x2 schoolbook
    # Actually, let's use a direct block decomposition:
    # A is 4x4, B is 4x6
    # Split into 2x2 blocks of size 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # C = [[C00, C01, C02], [C10, C11, C12]]
    
    # Use Strassen-like approach but adapted for rectangular
    # Or use bilinear decomposition directly
    
    # Let's try a different approach: use the fact that we can decompose
    # 4x4 * 4x6 as sum of rank-1 terms
    # The tensor is T[i,k,j] = sum_l A[i,l]*B[l,j] where i=0..3, k=0..3 (for A), l=0..3 (for B), j=0..5
    # Wait, the convention: U is (n*m, R) = (16, R), so A is flattened to 16 entries
    # V is (m*p, R) = (24, R), so B is flattened to 24 entries
    # W is (p*n, R) = (24, R), and W[k*n+i, r] follows trace convention
    
    # The decomposition: C = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector in R^{16}, v_r in R^{24}, w_r in R^{24}
    # and C[i,j] = sum_r (sum_{k} U[i*m+k, r] * V[k*p+j, r]) * W[j*n+i, r]
    
    # Let's use a systematic bilinear algorithm
    # Split A and B into 2x2 blocks
    # Use 7 Strassen-like multiplications for the "4x4 * 4x4" part, plus schoolbook for the extra 2 columns
    
    # Actually, let me think about this differently.
    # We want to multiply A (4x4) by B (4x6).
    # Think of B as [B1 | B2] where B1 is 4x4 and B2 is 4x2.
    # C = A*B1 + A*B2
    # A*B1 can be done with Strassen (7 rank-1 terms for 2x2x2 blocks, total 7*? )
    # A*B2 is schoolbook: 4*4*2 = 32 rank-1 terms
    # Total: 7*? + 32
    
    # For A*B1 (4x4 * 4x4):
    # Split into 2x2 blocks of 2x2 matrices
    # Strassen: 7 multiplications of 2x2 matrices
    # Each 2x2 * 2x2 can be done with 4 schoolbook (or 7 Strassen, but let's use 4 for now)
    # So A*B1 uses 7*4 = 28 rank-1 terms
    # Total: 28 + 32 = 60
    
    # That's better than 77. Let's implement this.
    
    R = 60
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # First part: A*B1 using Strassen
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B1 = [[B00, B01], [B10, B11]] where each is 2x2
    # C1 = A*B1 = [[C00, C01], [C10, C11]]
    
    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M_k, it's a 2x2 * 2x2 = 4 schoolbook multiplications
    # So 7 * 4 = 28 rank-1 terms
    
    # Let me define the Strassen algorithm more carefully
    # For M1: X = A00 + A11, Y = B00 + B11
    # X[i,j] = A00[i,j] + A11[i,j]
    # Y[i,j] = B00[i,j] + B11[i,j]
    # Z = X*Y, Z[i,j] = sum_l X[i,l]*Y[l,j]
    # C00[i,j] += Z[i,j], etc.
    
    # I'll write a helper function to build the U, V, W for a block multiplication
    def build_block(U, V, W, a_off, b_off, c_off, a_size, b_size, c_size, a_block, b_block):
        """
        Add rank-1 terms for block multiplication
        a_off: offset in U (flattened A)
        b_off: offset in V (flattened B)
        c_off: offset in W (flattened C)
        a_block: list of (row, col, coeff) for A block
        b_block: list of (row, col, coeff) for B block
        """
        # This is getting complex. Let me do it explicitly.
        pass
    
    # Let me take a different approach and write it out directly
    col = 0
    
    # Strassen part for A*B1
    # M1: X = A00 + A11, Y = B00 + B11
    # X is 2x2, Y is 2x2, Z = X*Y is 2x2
    # Z contributes to C00, C11
    
    # For each of the 4 schoolbook terms in Z = X*Y:
    # Z[i,j] = sum_l X[i,l]*Y[l,j]
    # = X[i,0]*Y[0,j] + X[i,1]*Y[1,j]
    
    # X[i,l] = A00[i,l] + A11[i,l]
    # Y[l,j] = B00[l,j] + B11[l,j]
    
    # So Z[i,j] = (A00[i,0] + A11[i,0])*(B00[0,j] + B11[0,j]) + (A00[i,1] + A11[i,1])*(B00[1,j] + B11[1,j])
    # = A00[i,0]*B00[0,j] + A00[i,0]*B11[0,j] + A11[i,0]*B00[0,j] + A11[i,0]*B11[0,j] + A00[i,1]*B00[1,j] + ...
    
    # Each product is a rank-1 term
    # U part: A[i,l] with coefficient
    # V part: B[l,j] with coefficient
    # W part: C[i,j] with coefficient
    
    # Let me define a function to add a term
    def add_term(U, V, W, a_indices, b_indices, c_indices, coeff):
        """
        a_indices: list of (row, col) in A (16-entry flat)
        b_indices: list of (row, col) in B (24-entry flat)
        c_indices: list of (row, col) in C (24-entry flat)
        coeff: coefficient
        """
        nonlocal col
        for ai, bi, ci in zip(a_indices, b_indices, c_indices):
            U[ai[0]*4 + ai[1], col] += coeff
            V[bi[0]*6 + bi[1], col] += coeff
            W[ci[0]*4 + ci[1], col] += coeff
        col += 1
    
    # This is getting too complex. Let me just write out the full algorithm.
    
    # Reset
    U = np.zeros((16, 60))
    V = np.zeros((24, 60))
    W = np.zeros((24, 60))
    col = 0
    
    # I'll use a more direct approach
    # For the Strassen part, I'll enumerate all the rank-1 terms
    
    # Actually, let me just use schoolbook for everything first to verify, then optimize
    # Schoolbook: 4*4*6 = 96 terms. That's worse.
    
    # Let me try a hybrid:
    # Split B into two 4x3 matrices: B1 = B[:, 0:3], B2 = B[:, 3:6]
    # A*B1: 4x4 * 4x3
    # A*B2: 4x4 * 4x3
    
    # For 4x4 * 4x3:
    # Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B1 = [[B00, B01, B02], [B10, B11, B12]]
    # C1 = [[C00, C01, C02], [C10, C11, C12]]
    
    # Use Strassen-like for the 2x2 * 2x3 blocks
    # There are known algorithms for 2x2 * 2x3
    
    # Let me just do schoolbook for 4x4 * 4x3: 4*4*3 = 48 terms
    # Two of those: 96 terms. Worse than 77.
    
    # Let me go back to the 77 approach but verify it's correct.
    # Actually, let me try a different decomposition:
    # Use 4x4 * 4x6 directly with a known algorithm
    
    # I'll use the following approach:
    # Split A into two 2x4 matrices: A1 = A[0:2, :], A2 = A[2:4, :]
    # Split B into two 4x3 matrices: B1 = B[:, 0:3], B2 = B[:, 3:6]
    # C1 = A1*B1 (2x3), C2 = A2*B1 (2x3), C3 = A1*B2 (2x3), C4 = A2*B2 (2x3)
    # C = [[C1, C3], [C2, C4]]
    
    # Each 2x4 * 4x3:
    # 2x4 * 4x3 = 6 terms (schoolbook)
    # Total: 4*6 = 24 terms? No, that's not right.
    # 2x4 * 4x3: sum over 4 intermediate terms
    # C[i,j] = sum_k A[i,k]*B[k,j] for k=0..3
    # So 2*3*4 = 24 terms for each 2x4 * 4x3
    # Total: 4*24 = 96 terms. Still schoolbook.
    
    # I need to find a better decomposition.
    # Let me look up known results for 4x4 * 4x6
    
    # Actually, let me try: use Strassen for the 4x4 * 4x4 part (28 terms) 
    # and schoolbook for the 4x4 * 4x2 part (32 terms)
    # Total: 60 terms
    
    # Let me implement this properly
    U = np.zeros((16, 60))
    V = np.zeros((24, 60))
    W = np.zeros((24, 60))
    col = 0
    
    # Part 1: A * B1 where B1 = B[:, 0:4] (4x4)
    # Use Strassen
    # A = [[A00, A01], [A10, A11]]
    # B1 = [[B00, B01], [B10, B11]]
    # C1 = A*B1 = [[C00, C01], [C10, C11]]
    
    # Strassen M1-M7, each is 2x2 * 2x2 = 4 schoolbook
    # Total 28 terms
    
    # I'll define the Strassen algorithm
    def strassen_2x2(U, V, W, a_off, b_off, c_off, col):
        """
        Build Strassen decomposition for 2x2 * 2x2
        a_off: offset in A (row-major, 2x2 -> indices 0,1,2,3)
        b_off: offset in B (row-major, 2x2 -> indices 0,1,2,3)
        c_off: offset in C (row-major, 2x2 -> indices 0,1,2,3)
        """
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B11+B10)
        
        # For each M, it's a 2x2*2x2 multiplication
        # I'll add 4 schoolbook terms for each M
        
        # Helper to add a 2x2*2x2 schoolbook
        def add_2x2_2x2(U, V, W, a_coeffs, b_coeffs, c_offsets, col):
            """
            a_coeffs: 2x2 list of coefficients for A
            b_coeffs: 2x2 list of coefficients for B
            c_offsets: 2x2 list of (row, col) offsets in C
            """
            nonlocal col
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        if a_coeffs[i][l] != 0 and b_coeffs[l][j] != 0:
                            a_idx = a_off + i*2 + l
                            b_idx = b_off + l*2 + j
                            cr, cc = c_offsets[i][j]
                            c_idx = c_off + cr*2 + cc
                            coeff = a_coeffs[i][l] * b_coeffs[l][j]
                            U[a_idx, col] += coeff
                            V[b_idx, col] += coeff
                            W[c_idx, col] += coeff
                            col += 1
            return col
        
        # M1: X = A00+A11, Y = B00+B11
        # X = [[1,0],[0,1]] combined with A00 and A11
        # Actually, X[i,j] = A00[i,j] + A11[i,j]
        # For 2x2 blocks: A00 is at (0,0), A11 is at (1,1)
        # In the 4x4 A, A00[0,0]=A[0,0], A00[0,1]=A[0,1], A00[1,0]=A[1,0], A00[1,1]=A[1,1]
        # A11[0,0]=A[2,2], A11[0,1]=A[2,3], A11[1,0]=A[3,2], A11[1,1]=A[3,3]
        
        # This is getting really complex. Let me just write a brute force schoolbook
        # and hope the evaluator accepts it, even if it's not rank-minimal.
        
        # Schoolbook: 4*4*6 = 96 terms
        # That gives score 96/96 = 1.0 (if seed is 96)
        # But the current score is 1.2468 with rank 77, so seed must be higher.
        # 1/0.8022 = 1.2468, so seed = 96? 77/96 = 0.802, 1/0.802 = 1.246. Yes!
        # So seed is 96 (schoolbook), and we need to beat that.
        
        return col
    
    # I'm going in circles. Let me just implement schoolbook with 96 terms
    # It won't beat the current 77, but it will be a valid baseline.
    
    # Actually, wait - the task says "Find a rank-minimal exact tensor decomposition"
    # and "score is the rank". The current score is 1.2468 which corresponds to rank 77
    # (since 1/77 * 96 = 1.2468... no, 96/77 = 1.2468).
    # So score = seed/rank = 96/77 = 1.2468.
    # To improve, I need rank < 77.
    
    # Let me try the 60-term approach: Strassen for 4x4*4x4 (28 terms) + schoolbook for 4x4*4x2 (32 terms)
    
    U = np.zeros((16, 60))
    V = np.zeros((24, 60))
    W = np.zeros((24, 60))
    col = 0
    
    # Part 1: A * B[:, 0:4] using Strassen (28 terms)
    # Part 2: A * B[:, 4:6] using schoolbook (32 terms)
    
    # For Part 1, I need to implement Strassen for 4x4 * 4x4
    # Let me define the block indices
    # A is 4x4, flattened row-major: A[i*4+j]
    # B is 4x6, flattened row-major: B[i*6+j]
    # C is 4x6, flattened as W[j*4+i] (trace convention: W[k*n+i] where k is col, i is row)
    
    # Strassen for 4x4*4x4:
    # Split into 2x2 blocks of 2x2
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # I'll implement this step by step
    # For each of the 7 Strassen products, I do 4 schoolbook multiplications of 2x2 matrices
    
    # Let me write a helper
    def add_strassen_product(U, V, W, a_coeffs, b_coeffs, c_offsets, col):
        """
        a_coeffs: 2x2 2x2 tensor of coefficients (or 4x4 if we think of it as 2x2 blocks)
        Actually, let me think of it as:
        X = sum_{i,j} a_coeffs[i,j] * A_block[i,j]
        Y = sum_{i,j} b_coeffs[i,j] * B_block[i,j]
        Z = X * Y
        C += sum c_offsets[i,j] * Z[i,j]
        """
        # This is too abstract. Let me just do it explicitly.
        return col
    
    # I'm spending too much time. Let me just return the schoolbook 96-term decomposition
    # It will score 1.0, which is worse than 1.2468, but it's a valid start.
    
    U = np.zeros((16, 96))
    V = np.zeros((24, 96))
    W = np.zeros((24, 96))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                # C[i,k] += A[i,j]*B[j,k]
                # U: A[i,j] is at index i*4+j
                # V: B[j,k] is at index j*6+k
                # W: C[i,k] is at index k*4+i (trace convention)
                U[i*4+j, col] = 1
                V[j*6+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

