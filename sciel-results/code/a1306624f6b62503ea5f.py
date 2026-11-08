import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Generate Strassen factors for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # Indices: A: 0:a, 1:b, 2:c, 3:d; B: 0:e, 1:f, 2:g, 3:h; C: 0:i, 1:j, 2:k, 3:l
    
    U2 = []
    V2 = []
    W2 = []
    
    # M1 = (a+d)(e+h) -> C11 += 1, C22 += 1
    U2.append([1, 0, 0, 1])
    V2.append([1, 0, 0, 1])
    W2.append([1, 0, 0, 1])
    
    # M2 = (c+d)e -> C21 += 1, C22 -= 1
    U2.append([0, 0, 1, 1])
    V2.append([1, 0, 0, 0])
    W2.append([0, 0, 1, -1])
    
    # M3 = a(f-h) -> C12 += 1, C22 += 1
    U2.append([1, 0, 0, 0])
    V2.append([0, 1, 0, -1])
    W2.append([0, 1, 0, 1])
    
    # M4 = d(g-e) -> C11 += 1, C21 += 1
    U2.append([0, 0, 0, 1])
    V2.append([0, 0, 1, -1])
    W2.append([1, 0, 1, 0])
    
    # M5 = (a+b)h -> C11 -= 1, C12 += 1
    U2.append([1, 1, 0, 0])
    V2.append([0, 0, 0, 1])
    W2.append([-1, 1, 0, 0])
    
    # M6 = (c-a)(e+f) -> C22 += 1
    U2.append([-1, 0, 1, 0])
    V2.append([1, 1, 0, 0])
    W2.append([0, 0, 0, 1])
    
    # M7 = (b-d)(g+h) -> C11 += 1
    U2.append([0, 1, 0, -1])
    V2.append([0, 0, 1, 1])
    W2.append([1, 0, 0, 0])
    
    U2 = np.array(U2).T # 4x7
    V2 = np.array(V2).T # 4x7
    W2 = np.array(W2).T # 4x7
    
    # Generate Strassen factors for 4x4x4
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks.
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    
    U4 = []
    V4 = []
    W4 = []
    
    # Define the 7 block linear combinations
    # Each is a 2x2 matrix of coefficients for the 4 blocks
    # Block order: 11, 12, 21, 22
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients for A blocks (2x2 grid of blocks, each block 2x2)
    # We need to map the 2x2 block coefficients to the 4x4 matrix indices.
    
    def expand_block_coeffs(coeffs_2x2, matrix_shape):
        # coeffs_2x2 is 2x2 array of scalars
        # Returns 16x1 vector of coefficients for the 4x4 matrix
        res = np.zeros(16)
        for i in range(2):
            for j in range(2):
                c = coeffs_2x2[i, j]
                if c != 0:
                    # Block (i,j) is at rows 2i:2i+2, cols 2j:2j+2
                    for r in range(2):
                        for k in range(2):
                            idx = (2*i + r) * 4 + (2*j + k)
                            res[idx] += c
        return res
    
    # For each of the 7 terms, we have a block multiplication.
    # We use the 2x2 Strassen factors (U2, V2, W2) for the block multiplication.
    # But we need to account for the linear combinations of blocks.
    
    # Term 1: (A11+A22)(B11+B22)
    # A_block = A11 + A22. Coeffs: [[1, 0], [0, 1]]
    # B_block = B11 + B22. Coeffs: [[1, 0], [0, 1]]
    # C_block = C11 + C22 (partially).
    # Actually, the Strassen formula gives M1.
    # M1 contributes to C11 and C22.
    # We need to decompose M1 = A' B' where A' and B' are 2x2 matrices.
    # The decomposition of M1 into 7 scalar terms is given by U2, V2, W2.
    # However, A' and B' are themselves linear combinations of blocks.
    # So we substitute A' and B' into the 2x2 Strassen terms.
    
    # Let's do it systematically.
    # For each of the 7 Strassen terms at level 4:
    # 1. Define A_lin (2x2 coeffs for A blocks)
    # 2. Define B_lin (2x2 coeffs for B blocks)
    # 3. Define C_lin (2x2 coeffs for C blocks) - this is how M contributes to C.
    
    # Wait, the W2 matrix already handles the distribution of the scalar product to C entries.
    # The C_lin here would be the mapping from the 2x2 block result to the 4x4 C.
    # But M1 is a 2x2 block. It maps to C11 and C22?
    # No, M1 is just one of the 7 intermediate matrices.
    # The final C is computed as:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each M_k, we need to know which C blocks it contributes to and with what sign.
    # And M_k is a 2x2 block product.
    
    # Let's define the contribution of each M_k to the C blocks.
    # M1: C11 += 1, C22 += 1
    # M2: C21 += 1, C22 -= 1
    # M3: C12 += 1, C22 += 1
    # M4: C11 += 1, C21 += 1
    # M5: C11 -= 1, C12 += 1
    # M6: C22 += 1
    # M7: C11 += 1
    
    # Now, for each M_k, we decompose the block product into 7 scalar products using U2, V2, W2.
    # The scalar products are between entries of A' and B'.
    # A' is a linear combo of A blocks.
    # B' is a linear combo of B blocks.
    
    # We iterate through the 7 terms of level 4.
    # For each term, we have A_lin and B_lin.
    # We iterate through the 7 terms of level 2 (from U2, V2, W2).
    # For each level-2 term, we have u_vec (4 coeffs for A' entries), v_vec (4 coeffs for B' entries), w_vec (4 coeffs for M entries).
    # We need to expand u_vec to 16 coeffs for A.
    # u_vec[0] corresponds to A'_{11}. A'_{11} is the (1,1) entry of the 2x2 matrix A'.
    # A' is defined by A_lin.
    # A'_{11} = sum_{i,j} A_lin[0,0] * A_block_{ij}? No.
    # A' is a 2x2 matrix. Its entries are linear combinations of the 4 blocks of A.
    # Wait, A_lin is 2x2. A_lin[i,j] is the coefficient of Block_{ij} in A'.
    # But A' is a 2x2 matrix of blocks? No, A' is a 2x2 matrix of *scalars* in the Strassen formula?
    # No, in block Strassen, A' is a 2x2 matrix of *blocks*.
    # So A'_{11} is a block.
    # My U2, V2, W2 are for scalar 2x2 multiplication.
    # So I should apply U2, V2, W2 to the *blocks*.
    
    # Let's re-evaluate.
    # Level 4: 7 block multiplications.
    # Each block multiplication is a 2x2 matrix product.
    # We use Level 2 Strassen (rank 7) for each 2x2 block product.
    # So for each Level 4 term, we add 7 Level 2 terms.
    
    # For a Level 4 term k:
    # A'_k is a 2x2 matrix of blocks.
    # B'_k is a 2x2 matrix of blocks.
    # M_k = A'_k B'_k.
    # We decompose M_k using U2, V2, W2.
    # U2[:, r] gives coefficients for the 4 entries of A'_k (which are blocks).
    # V2[:, r] gives coefficients for the 4 entries of B'_k (which are blocks).
    # W2[:, r] gives coefficients for the 4 entries of M_k (which are blocks).
    
    # To get the scalar coefficients for A (16x1):
    # For each of the 4 entries of A'_k (blocks), we have a coefficient from U2.
    # Each block is a linear combination of 4 blocks of A.
    # So we multiply the coefficient by the block expansion.
    
    # We need A_lin_k (2x2 matrix of block coefficients for A'_k).
    # And B_lin_k (2x2 matrix of block coefficients for B'_k).
    
    # Let's define A_lin and B_lin for the 7 terms.
    
    terms = []
    # M1: (A11+A22)(B11+B22)
    A_lin = np.array([[1, 0], [0, 1]])
    B_lin = np.array([[1, 0], [0, 1]])
    terms.append((A_lin, B_lin))
    
    # M2: (A21+A22)B11
    A_lin = np.array([[0, 0], [1, 1]])
    B_lin = np.array([[1, 0], [0, 0]])
    terms.append((A_lin, B_lin))
    
    # M3: A11(B12-B22)
    A_lin = np.array([[1, 0], [0, 0]])
    B_lin = np.array([[0, 1], [0, -1]])
    terms.append((A_lin, B_lin))
    
    # M4: A22(B21-B11)
    A_lin = np.array([[0, 0], [0, 1]])
    B_lin = np.array([[-1, 0], [1, 0]])
    terms.append((A_lin, B_lin))
    
    # M5: (A11+A12)B22
    A_lin = np.array([[1, 1], [0, 0]])
    B_lin = np.array([[0, 0], [0, 1]])
    terms.append((A_lin, B_lin))
    
    # M6: (A21-A11)(B11+B12)
    A_lin = np.array([[-1, 0], [1, 0]])
    B_lin = np.array([[1, 1], [0, 0]])
    terms.append((A_lin, B_lin))
    
    # M7: (A12-A22)(B21+B22)
    A_lin = np.array([[0, 1], [0, -1]])
    B_lin = np.array([[0, 0], [1, 1]])
    terms.append((A_lin, B_lin))
    
    # Now construct U4, V4, W4
    for k in range(7):
        A_lin, B_lin = terms[k]
        # For each of the 7 scalar terms in the 2x2 block multiplication
        for r in range(7):
            u_block = U2[:, r] # 4 coeffs for A' blocks
            v_block = V2[:, r] # 4 coeffs for B' blocks
            w_block = W2[:, r] # 4 coeffs for M blocks
            
            # Expand u_block to 16 coeffs for A
            u_full = np.zeros(16)
            for i in range(2):
                for j in range(2):
                    coeff = u_block[i*2 + j] # Index in u_block for block (i,j)
                    # Wait, U2 is 4x7. The 4 entries correspond to A11, A12, A21, A22 of the 2x2 matrix.
                    # So u_block[0] is A'11, u_block[1] is A'12, u_block[2] is A'21, u_block[3] is A'22.
                    # A'11 is the (0,0) entry of the 2x2 matrix A'.
                    # A' is defined by A_lin.
                    # A'_{ij} = sum_{p,q} A_lin[i,j] * A_block_{pq}? No.
                    # A' is a 2x2 matrix. Its entries are blocks.
                    # A'_{00} = A_lin[0,0]*A11 + A_lin[0,1]*A12 + ...?
                    # No, A_lin[i,j] is the coefficient of Block_{ij} in the *entire* matrix A'.
                    # This is not correct.
                    # In Strassen, A' = A11 + A22.
                    # A' is a 2x2 matrix? No, A11 and A22 are 2x2 matrices.
                    # So A' is a 2x2 matrix.
                    # A'_{11} = A11_{11} + A22_{11}.
                    # So the (i,j) entry of A' is the sum of (i,j) entries of the contributing blocks.
                    
                    # So u_block[0] is the coefficient for A'_{11} (top-left entry of the 2x2 matrix A').
                    # A'_{11} = sum_{p,q} A_lin[p,q] * A_block_{p,q, 0,0}.
                    # Here A_block_{p,q} is the block at (p,q).
                    # So u_full[idx] += u_block[0] * A_lin[p,q] for idx corresponding to A_block_{p,q, 0,0}.
                    
                    pass # This logic is getting messy. Let's use a simpler mapping.
            
            # Simpler mapping:
            # u_block[0] corresponds to A'_{0,0}.
            # A'_{0,0} = sum_{p,q} A_lin[p,q] * A_{p,q, 0,0}.
            # So for each p,q:
            #   coeff = u_block[0] * A_lin[p,q]
            #   idx = (2*p) * 4 + (2*q)
            #   u_full[idx] += coeff
            
            # Similarly for u_block[1] (A'_{0,1}), u_block[2] (A'_{1,0}), u_block[3] (A'_{1,1}).
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            
            # A' entries
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for entry_idx in range(4):
                coeff = u_block[entry_idx]
                if coeff == 0: continue
                r_prime = entry_idx // 2
                c_prime = entry_idx % 2
                for p in range(2):
                    for q in range(2):
                        a_coeff = A_lin[p, q]
                        if a_coeff == 0: continue
                        idx = (2*p + r_prime) * 4 + (2*q + c_prime)
                        u_full[idx] += coeff * a_coeff
                        
            # B' entries
            for entry_idx in range(4):
                coeff = v_block[entry_idx]
                if coeff == 0: continue
                r_prime = entry_idx // 2
                c_prime = entry_idx % 2
                for p in range(2):
                    for q in range(2):
                        b_coeff = B_lin[p, q]
                        if b_coeff == 0: continue
                        idx = (2*p + r_prime) * 4 + (2*q + c_prime)
                        v_full[idx] += coeff * b_coeff
                        
            # W expansion
            # w_block[0] is M_{0,0}, w_block[1] is M_{0,1}, etc.
            # M is a 2x2 block.
            # We need to map M blocks to C blocks.
            # This depends on which Strassen term k it is.
            # We need C_lin_k.
            
            # Let's define C_lin for the 7 terms.
            # M1: C11 += 1, C22 += 1. So M_{0,0} -> C11, M_{1,1} -> C22.
            # M2: C21 += 1, C22 -= 1.
            # M3: C12 += 1, C22 += 1.
            # M4: C11 += 1, C21 += 1.
            # M5: C11 -= 1, C12 += 1.
            # M6: C22 += 1.
            # M7: C11 += 1.
            
            C_lin = np.zeros((2,2))
            if k == 0: # M1
                C_lin[0,0] = 1; C_lin[1,1] = 1
            elif k == 1: # M2
                C_lin[1,0] = 1; C_lin[1,1] = -1
            elif k == 2: # M3
                C_lin[0,1] = 1; C_lin[1,1] = 1
            elif k == 3: # M4
                C_lin[0,0] = 1; C_lin[1,0] = 1
            elif k == 4: # M5
                C_lin[0,0] = -1; C_lin[0,1] = 1
            elif k == 5: # M6
                C_lin[1,1] = 1
            elif k == 6: # M7
                C_lin[0,0] = 1
            
            w_full = np.zeros(16)
            for entry_idx in range(4):
                coeff = w_block[entry_idx]
                if coeff == 0: continue
                r_prime = entry_idx // 2
                c_prime = entry_idx % 2
                for p in range(2):
                    for q in range(2):
                        c_coeff = C_lin[p, q]
                        if c_coeff == 0: continue
                        idx = (2*p + r_prime) * 4 + (2*q + c_prime)
                        w_full[idx] += coeff * c_coeff
                        
            U4.append(u_full)
            V4.append(v_full)
            W4.append(w_full)
            
    U4 = np.array(U4).T # 16x49
    V4 = np.array(V4).T # 16x49
    W4 = np.array(W4).T # 16x49
    
    # Now combine for 4x4x8
    # U is 16x98
    # V is 32x98
    # W is 32x98
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    return {"U": U, "V": V, "W": W}

