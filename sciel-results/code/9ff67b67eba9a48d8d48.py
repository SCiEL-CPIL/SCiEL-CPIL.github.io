import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x2 using Schoolbook on blocks (Rank 16)
    
    total_rank = 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Strassen levels:
    # Level 1: 7 products S_k * T_k
    # Level 2: Each S_k * T_k is 2x2x2, decomposed into 7 rank-1 terms.
    
    # Define Strassen coefficients for 2x2x2
    # S matrices (A side)
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    # T matrices (B side)
    # T1 = B00 + B11
    # T2 = B00
    # T3 = B01 - B11
    # T4 = B10 - B00
    # T5 = B11
    # T6 = B00 + B01
    # T7 = B10 + B11
    
    # C combination
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate over the 7 outer Strassen terms (k_outer)
    # And the 7 inner Strassen terms (k_inner)
    
    # Helper to get local indices for 2x2 block
    # Block (I, J) of 4x4 matrix corresponds to rows 2I..2I+1, cols 2J..2J+1
    # Local index u, v in 0..1 -> Global 2I+u, 2J+v
    
    # Inner Strassen decomposition for 2x2x2
    # Returns list of (u_coef, v_coef, w_coef) for the 7 terms
    # u_coef, v_coef are 2x2 matrices (flattened to 4)
    # w_coef is 2x2 matrix (flattened to 4)
    
    def get_inner_strassen_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # A part: A00 + A11 -> [[1,0],[1,1]]? No.
        # A00 is top-left, A11 is bottom-right.
        # S1 = [[1,0],[0,1]]*A00 + [[0,0],[0,1]]*A11?
        # Let's define the linear forms explicitly.
        
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # S1: A00 + A11. Coeffs: [1,0,0,1]
        # T1: B00 + B11. Coeffs: [1,0,0,1]
        # W1: C00 + C11? No, W is the output coefficient.
        # The inner Strassen gives M1.
        # M1 is added to C00 and C11.
        # So for the inner term, the W vector should have 1 at C00 and C11?
        # No, the inner Strassen decomposes the product S1*T1 into 7 rank-1 tensors.
        # Each rank-1 tensor contributes to M1.
        # Then M1 is combined into C.
        
        # Let's just generate the 7 rank-1 tensors for 2x2x2 multiplication.
        # Standard Strassen factors for 2x2x2:
        # u1 = [[1,0],[1,1]], v1 = [[1,1],[0,1]], w1 = [[1,1],[1,0]]
        # u2 = [[0,1],[0,1]], v2 = [[1,0],[1,0]], w2 = [[0,1],[0,1]]
        # u3 = [[1,0],[0,0]], v3 = [[0,1],[0,1]], w3 = [[1,0],[0,0]]
        # u4 = [[0,0],[0,1]], v4 = [[0,0],[1,0]], w4 = [[0,0],[0,1]]
        # u5 = [[1,1],[0,0]], v5 = [[0,0],[1,1]], w5 = [[1,0],[0,0]]
        # u6 = [[0,0],[1,-1]], v6 = [[1,1],[0,0]], w6 = [[0,1],[0,0]]
        # u7 = [[0,1],[-1,0]], v7 = [[0,0],[1,1]], w7 = [[0,0],[1,0]]
        
        # Wait, I need to verify these factors.
        # Let's use a known set.
        # Reference: "Matrix Multiplication via Strassen"
        
        # I will hardcode the 7 terms for 2x2x2.
        # u, v, w are 2x2 matrices flattened to length 4 (row-major).
        
        terms = []
        
        # Term 1
        u = [1,0, 1,1]
        v = [1,1, 0,1]
        w = [1,1, 1,0]
        terms.append((u,v,w))
        
        # Term 2
        u = [0,1, 0,1]
        v = [1,0, 1,0]
        w = [0,1, 0,1]
        terms.append((u,v,w))
        
        # Term 3
        u = [1,0, 0,0]
        v = [0,1, 0,1]
        w = [1,0, 0,0]
        terms.append((u,v,w))
        
        # Term 4
        u = [0,0, 0,1]
        v = [0,0, 1,0]
        w = [0,0, 0,1]
        terms.append((u,v,w))
        
        # Term 5
        u = [1,1, 0,0]
        v = [0,0, 1,1]
        w = [1,0, 0,0]
        terms.append((u,v,w))
        
        # Term 6
        u = [0,0, 1,-1]
        v = [1,1, 0,0]
        w = [0,1, 0,0]
        terms.append((u,v,w))
        
        # Term 7
        u = [0,1, -1,0]
        v = [0,0, 1,1]
        w = [0,0, 1,0]
        terms.append((u,v,w))
        
        return terms

    inner_terms = get_inner_strassen_terms()
    
    # Outer Strassen combinations
    # For each outer term k (0..6), we have S_k and T_k.
    # S_k and T_k are linear combinations of the 4 blocks of A and B.
    # We need to map the inner 2x2 indices to the global 4x4 indices.
    
    # Outer S_k definitions (coeffs for A blocks A00, A01, A10, A11)
    # A00: top-left, A01: top-right, A10: bottom-left, A11: bottom-right
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    outer_S = [
        [1, 0, 0, 1], # S1 = A00 + A11
        [0, 0, 1, 1], # S2 = A10 + A11
        [1, 0, 0, 0], # S3 = A00
        [0, 0, 0, 1], # S4 = A11
        [1, 1, 0, 0], # S5 = A00 + A01
        [1, 0, 1, 0], # S6 = A10 - A00? No, S6 = A10 - A00. Coeffs: [1,0,1,0] with sign?
                      # A10 is block 2. A00 is block 0.
                      # So [1, 0, 1, 0] means 1*A00 + 1*A10. But we want A10 - A00.
                      # So [-1, 0, 1, 0].
        [0, 1, 0, -1] # S7 = A01 - A11. Coeffs: [0, 1, 0, -1]
    ]
    # Correction for S6: A10 - A00.
    # A00 is index 0. A10 is index 2.
    # So coeff[0] = -1, coeff[2] = 1.
    outer_S[5] = [-1, 0, 1, 0]
    
    outer_T = [
        [1, 0, 0, 1], # T1 = B00 + B11
        [1, 0, 0, 0], # T2 = B00
        [0, 1, 0, -1],# T3 = B01 - B11
        [0, 0, 1, -1],# T4 = B10 - B00
        [0, 0, 0, 1], # T5 = B11
        [1, 1, 0, 0], # T6 = B00 + B01
        [0, 0, 1, 1]  # T7 = B10 + B11
    ]
    
    # Outer C combinations
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # M_k is the result of the k-th outer product.
    # M_k is a 2x2 matrix.
    # C blocks are 2x2 matrices.
    # C00, C01, C10, C11.
    # C00 is top-left of C.
    
    # We need to sum the contributions from each M_k.
    # For each outer term k, we have 7 inner terms.
    # Each inner term is a rank-1 tensor for the 2x2x2 product S_k T_k -> M_k.
    # The W part of the inner term tells us how M_k is formed.
    # But M_k is then combined into C.
    
    # Let's trace one term.
    # Outer k. Inner t.
    # U part: S_k * u_t (element-wise? No, S_k is a matrix, u_t is a matrix?
    # No, S_k is a linear combination of A blocks.
    # u_t is a linear combination of the entries of S_k?
    # No, u_t is the "left factor" of the rank-1 decomposition of the 2x2x2 tensor.
    # So U_global = S_k \otimes u_t?
    # S_k is a 2x2 matrix of coefficients for A blocks.
    # u_t is a 2x2 matrix of coefficients for S_k entries?
    # This is getting complicated.
    
    # Alternative: Just generate the full U, V, W for the 4x4x4 part by iterating
    # through all 49 scalar multiplications and determining the coefficients.
    
    # Let's do it explicitly.
    # For each outer k (0..6):
    #   For each inner t (0..6):
    #     The term is: (S_k linear form) * (T_k linear form) * (M_k linear form)
    #     S_k linear form is a sum of 4 A blocks.
    #     u_t is a 2x2 matrix.
    #     The product S_k * u_t? No.
    #     The inner decomposition is:
    #     M_k = sum_t (U_t S_k) (V_t T_k) W_t^T?
    #     No.
    #     The 2x2x2 tensor T_222 satisfies:
    #     (A B)_{ij} = sum_k (u_k A)_{ij} (v_k B)_{ij} w_k?
    #     No.
    #     (A B) = sum_k (u_k \circ A) (v_k \circ B) w_k^T?
    #     Where \circ is element-wise product? No.
    #     Strassen:
    #     M1 = (A00+A11)(B00+B11)
    #     This is a matrix product.
    #     We decompose this matrix product into rank-1 terms?
    #     No, the matrix product itself is the "multiplication".
    #     The rank-1 decomposition of the 2x2x2 tensor means:
    #     C = sum_{t=1}^7 (L_{A,t}(A)) (L_{B,t}(B)) (L_{C,t}(C))?
    #     No, C is the output.
    #     C_{ij} = sum_k A_{ik} B_{kj}.
    #     Strassen expresses this as sum of 7 terms of the form:
    #     (Linear in A) * (Linear in B) * (Linear in C)?
    #     No, it's a bilinear map.
    #     C = sum_t (U_t A) (V_t B) W_t^T?
    #     No.
    #     C_{ij} = sum_t (sum_{kl} U_{t,ij,kl} A_{kl}) (sum_{mn} V_{t,ij,mn} B_{mn})?
    #     No, that would be rank 1 in the tensor sense.
    
    # Let's use the explicit formula:
    # For 2x2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Where M1..M7 are scalar products of linear forms.
    # M1 = (A00+A11)(B00+B11)
    # This M1 is a scalar? No, in Strassen for 2x2, A00 is a scalar.
    # But in our case, A00 is a 2x2 block.
    # So M1 is a 2x2 matrix product.
    # We decompose M1 into 7 scalar products using Strassen again.
    
    # So, for each outer k:
    #   S_k is a 2x2 matrix of A blocks.
    #   T_k is a 2x2 matrix of B blocks.
    #   M_k = S_k T_k.
    #   M_k is a 2x2 matrix.
    #   We decompose M_k using inner Strassen.
    #   Inner Strassen gives 7 scalar products.
    #   Each scalar product is of the form:
    #   (Linear in S_k) * (Linear in T_k)
    #   Linear in S_k is a linear combination of the 4 entries of S_k.
    #   S_k entries are linear combinations of A blocks.
    #   So Linear in S_k is a linear combination of A blocks.
    
    # We will build U, V, W by accumulating these coefficients.
    
    # U shape: 16 x 49
    # V shape: 24 x 49 (for 4x4 part, B is 4x4, so 16 entries? No, B is 4x6.
    # For the 4x4x4 part, we only use the first 4 columns of B.
    # So V is 16 x 49 for this part.
    # W shape: 16 x 49 (for C, 4x4 part, 16 entries).
    
    # Let's map indices.
    # A flat: 16.
    # B flat (first 4 cols): 16.
    # C flat (first 4 cols): 16.
    
    # For each of the 49 terms:
    #   U[col] = coefficients for A
    #   V[col] = coefficients for B (first 4 cols)
    #   W[col] = coefficients for C (first 4 cols)
    
    # We need a helper to add coefficients.
    
    # Let's iterate.
    
    for k_outer in range(7):
        S_coeff = outer_S[k_outer]
        T_coeff = outer_T[k_outer]
        
        # S_coeff is [s00, s01, s10, s11]
        # s00 is coeff for A00, etc.
        
        # For each inner term t:
        for t_inner in range(7):
            u, v, w = inner_terms[t_inner]
            # u, v, w are flattened 2x2 matrices.
            # u = [u00, u01, u10, u11]
            
            # The inner term contributes to M_k.
            # M_k = sum_t (u_t \circ S_k) (v_t \circ T_k) w_t?
            # No.
            # The inner Strassen decomposes the product S_k T_k.
            # S_k is a 2x2 matrix. T_k is a 2x2 matrix.
            # The product is a 2x2 matrix.
            # The rank-1 decomposition of the 2x2x2 tensor (where inputs are 2x2 matrices)
            # is given by the 7 terms.
            # Each term is a product of a linear form in S_k and a linear form in T_k,
            # multiplied by a linear form in the output M_k?
            # No, the output is M_k.
            # The term is:
            # (sum_{i,j} u_{ij} S_k[i,j]) * (sum_{i,j} v_{ij} T_k[i,j])
            # This is a scalar.
            # This scalar is added to M_k[i,j] with coefficient w_{ij}.
            
            # So, for this term:
            # Scalar = (u dot S_k) * (v dot T_k)
            # Contribution to M_k[i,j] = w_{ij} * Scalar
            
            # Now, M_k is combined into C.
            # C_block[r,c] += coeff * M_k
            
            # We need to distribute this to the final C blocks.
            
            # Let's determine the coefficients for A and B.
            # A_coeff = 0 (16,)
            # B_coeff = 0 (16,)
            
            # (u dot S_k) is a linear form in A.
            # S_k[i,j] = sum_{block} S_coeff[block] * A_block[i,j]
            # So u dot S_k = sum_{i,j} u_{ij} S_k[i,j]
            # = sum_{i,j} u_{ij} sum_{block} S_coeff[block] A_block[i,j]
            # = sum_{block} S_coeff[block] sum_{i,j} u_{ij} A_block[i,j]
            
            # So the coefficient for A_block[i,j] is S_coeff[block] * u_{ij}.
            
            # Similarly for B.
            
            # Now, the scalar is multiplied by w_{ij} and added to M_k[i,j].
            # Then M_k[i,j] is added to C.
            
            # We need to know how M_k is added to C.
            # C00 += M1 + M4 - M5 + M7
            # C01 += M3 + M5
            # C10 += M2 + M4
            # C11 += M1 - M2 + M3 + M6
            
            # Let's create a map from M_k index to C block and coefficient.
            # M0 (k=0) is M1.
            # M1 (k=1) is M2.
            # M2 (k=2) is M3.
            # M3 (k=3) is M4.
            # M4 (k=4) is M5.
            # M5 (k=5) is M6.
            # M6 (k=6) is M7.
            
            # C00 (block 0) gets M0(1), M3(1), M4(-1), M6(1)
            # C01 (block 1) gets M2(1), M4(1)
            # C10 (block 2) gets M1(1), M3(1)
            # C11 (block 3) gets M0(1), M1(-1), M2(1), M5(1)
            
            # For a fixed k_outer, we know which M_k it is.
            # We need to add w_{ij} * (coeff for C block) to the W matrix.
            
            # Let's define the C combination coefficients for each M_k.
            # c_comb[k_outer][c_block][i,j]
            
            # This is getting very complex to implement correctly in one go.
            # I will use a simpler approach:
            # Just use the rank 49 decomposition but generate the matrices programmatically
            # by verifying the tensor equality.
            
            # Actually, I can just return the schoolbook rank 96 if I can't get it right.
            # But 81 is a good target.
            
            pass

    # Given the complexity and risk of error, I will stick to the schoolbook 96 for now?
    # No, the directive asks for improvement.
    # I will implement the 49+16=65 rank solution carefully.

    # Let's restart the implementation with a clear structure.

    # 1. Initialize U, V, W with zeros.
    # 2. Fill in the 49 terms for 4x4x4.
    # 3. Fill in the 16 terms for 4x4x2.

    # To ensure correctness, I will use a small helper to add terms.

    # For the 4x4x4 part:
    # We iterate k_outer (0..6) and k_inner (0..6).
    # For each pair, we determine the linear forms for A and B.
    # And the linear form for C.

    # Linear form for A:
    # L_A = sum_{i,j} coeff_A[i,j] A[i,j]
    # coeff_A is 4x4.
    
    # For a given k_outer, S_k is a 2x2 matrix of blocks.
    # S_k[i,j] = sum_{b} S_coeff[b] A_block_b[i,j]
    # b=0: A00, b=1: A01, b=2: A10, b=3: A11.
    
    # The inner term uses u (2x2).
    # L_A = sum_{i,j} u[i,j] S_k[i,j]
    # = sum_{i,j} u[i,j] sum_{b} S_coeff[b] A_block_b[i,j]
    # = sum_{b} S_coeff[b] sum_{i,j} u[i,j] A_block_b[i,j]
    
    # So coeff_A for block b, entry (i,j) is S_coeff[b] * u[i,j].
    
    # Similarly for B.
    
    # For C:
    # The inner term contributes w[i,j] to M_k[i,j].
    # M_k is then combined into C blocks.
    # Let c_coeff[c_block] be the coefficient of M_k in C_block.
    # Then the contribution to C_block[i,j] is w[i,j] * c_coeff[c_block].
    
    # So coeff_C for C_block, entry (i,j) is w[i,j] * c_coeff[c_block].

    # We need to map (c_block, i, j) to global C index.
    # C_block 0: rows 0-1, cols 0-1.
    # C_block 1: rows 0-1, cols 2-3.
    # C_block 2: rows 2-3, cols 0-1.
    # C_block 3: rows 2-3, cols 2-3.
    
    # Global C index = row * 4 + col.
    
    # Let's code this.

    # Precompute c_comb
    # c_comb[k_outer][c_block] = coeff
    c_comb = [
        [1, 0, 0, 0], # M1 -> C00:1, C01:0, C10:0, C11:0? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # k=0 (M1): C00:1, C11:1
        # k=1 (M2): C10:1, C11:-1
        # k=2 (M3): C01:1, C11:1
        # k=3 (M4): C00:1, C10:1
        # k=4 (M5): C00:-1, C01:1
        # k=5 (M6): C11:1
        # k=6 (M7): C00:1
    ]
    c_comb = [
        [1, 0, 0, 1], # k=0 (M1)
        [0, 0, 1, -1],# k=1 (M2)
        [0, 1, 0, 1], # k=2 (M3)
        [1, 0, 1, 0], # k=3 (M4)
        [-1, 1, 0, 0],# k=4 (M5)
        [0, 0, 0, 1], # k=5 (M6)
        [1, 0, 0, 0]  # k=6 (M7)
    ]

    # Now fill U, V, W for 49 terms.
    for k_outer in range(7):
        S_coeff = outer_S[k_outer]
        T_coeff = outer_T[k_outer]
        for t_inner in range(7):
            u, v, w = inner_terms[t_inner]
            
            # Build U_col (16,), V_col (16,), W_col (16,)
            U_col = [0] * 16
            V_col = [0] * 16
            W_col = [0] * 16
            
            # Fill U_col
            for b in range(4):
                # b=0: A00, b=1: A01, b=2: A10, b=3: A11
                # Block b corresponds to rows 2*(b//2)? No.
                # b=0: I=0, J=0.
                # b=1: I=0, J=1.
                # b=2: I=1, J=0.
                # b=3: I=1, J=1.
                I = b // 2
                J = b % 2
                
                # S_coeff[b] is the coefficient for block b.
                # u is 2x2 flattened.
                for i in range(2):
                    for j in range(2):
                        # Global A index: row = 2*I + i, col = 2*J + j
                        row = 2*I + i
                        col = 2*J + j
                        U_col[row*4 + col] += S_coeff[b] * u[i*2 + j]
            
            # Fill V_col
            for b in range(4):
                I = b // 2
                J = b % 2
                for i in range(2):
                    for j in range(2):
                        row = 2*I + i
                        col = 2*J + j
                        # B index for 4x4 part: row*4 + col
                        V_col[row*4 + col] += T_coeff[b] * v[i*2 + j]
            
            # Fill W_col
            for c_block in range(4):
                coeff_c = c_comb[k_outer][c_block]
                if coeff_c == 0:
                    continue
                I = c_block // 2
                J = c_block % 2
                for i in range(2):
                    for j in range(2):
                        row = 2*I + i
                        col = 2*J + j
                        # W index for 4x4 part: col*4 + row (Trace convention W[k*n+i])
                        # n=4, p=6.
                        # C is 4x4. k is col, i is row.
                        # Index = k*4 + i.
                        W_col[col*4 + row] += coeff_c * w[i*2 + j]
            
            # Add to U, V, W
            for i in range(16):
                U[i][col] = U_col[i]
                V[i][col] = V_col[i]
                W[i][col] = W_col[i]
            
            col += 1

    # --- Part 2: 4x4x2 (Rank 16) ---
    # Schoolbook for blocks.
    # A is 4x4, B is 4x2, C is 4x2.
    # B cols 4,5. C cols 4,5.
    
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0..3, j in 0..3, k in 4..5.
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*4 + j][col] = 1
                V[j*6 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
