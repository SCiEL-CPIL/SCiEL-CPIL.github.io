import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _get_strassen_2x2_factors():
    """
    Returns the 7 Strassen factors for 2x2 matrix multiplication.
    Each factor is a tuple (u, v, w) where u, v, w are numpy arrays of shape (4,).
    u corresponds to A indices: (0,0), (0,1), (1,0), (1,1)
    v corresponds to B indices: (0,0), (0,1), (1,0), (1,1)
    w corresponds to C indices: (0,0), (0,1), (1,0), (1,1)
    """
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00 + A11)(B00 + B11)
    u1 = np.array([1, 0, 0, 1], dtype=float)
    v1 = np.array([1, 0, 0, 1], dtype=float)
    # C00 += M1, C11 += M1
    # C01 -= M1? No, let's use the standard formulas
    # C00 = M1 - M2 + M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M4 + M6 - M5
    
    # We need to define w vectors such that sum(w_k * M_k) = C
    # Let's define the w vectors for each M_k
    
    w1 = np.array([1, 0, 0, 1], dtype=float) # C00 and C11 get +1 from M1
    
    # M2 = (A10 + A11)B00
    u2 = np.array([0, 0, 1, 1], dtype=float)
    v2 = np.array([1, 0, 0, 0], dtype=float)
    # C00 -= M2, C10 += M2
    w2 = np.array([-1, 0, 1, 0], dtype=float)
    
    # M3 = A00(B01 - B11)
    u3 = np.array([1, 0, 0, 0], dtype=float)
    v3 = np.array([0, 1, 0, -1], dtype=float)
    # C01 += M3
    w3 = np.array([0, 1, 0, 0], dtype=float)
    
    # M4 = A11(B10 - B00)
    u4 = np.array([0, 0, 0, 1], dtype=float)
    v4 = np.array([0, 0, 1, -1], dtype=float)
    # C00 += M4, C11 -= M4
    w4 = np.array([1, 0, 0, -1], dtype=float)
    
    # M5 = (A00 + A01)B11
    u5 = np.array([1, 1, 0, 0], dtype=float)
    v5 = np.array([0, 0, 0, 1], dtype=float)
    # C01 += M5, C11 -= M5
    w5 = np.array([0, 1, 0, -1], dtype=float)
    
    # M6 = (A10 - A00)(B00 + B01)
    u6 = np.array([-1, 0, 1, 0], dtype=float)
    v6 = np.array([1, 1, 0, 0], dtype=float)
    # C10 += M6, C11 += M6
    w6 = np.array([0, 0, 1, 1], dtype=float)
    
    # M7 = (A01 - A11)(B10 + B11)
    u7 = np.array([0, 1, 0, -1], dtype=float)
    v7 = np.array([0, 0, 1, 1], dtype=float)
    # C00 -= M7
    w7 = np.array([-1, 0, 0, 0], dtype=float)
    
    return [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Get base 2x2 Strassen factors
    base_factors = _get_strassen_2x2_factors()
    
    # Define the 7 Strassen combinations for the 4x4 level
    # Each entry is (U_comb_coeffs, V_comb_coeffs)
    # U_comb is a 2x2 matrix of coefficients for the 4 blocks of A
    # V_comb is a 2x2 matrix of coefficients for the 4 blocks of B
    # Blocks are indexed 00, 01, 10, 11
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # T0: (A00 + A11)(B00 + B11)
    U_comb_0 = np.array([
        [1, 0],
        [0, 1]
    ])
    V_comb_0 = np.array([
        [1, 0],
        [0, 1]
    ])
    
    # T1: (A10 + A11)B00
    U_comb_1 = np.array([
        [0, 0],
        [1, 1]
    ])
    V_comb_1 = np.array([
        [1, 0],
        [0, 0]
    ])
    
    # T2: A00(B01 - B11)
    U_comb_2 = np.array([
        [1, 0],
        [0, 0]
    ])
    V_comb_2 = np.array([
        [0, 1],
        [0, -1]
    ])
    
    # T3: A11(B10 - B00)
    U_comb_3 = np.array([
        [0, 0],
        [0, 1]
    ])
    V_comb_3 = np.array([
        [0, 0],
        [1, -1]
    ])
    
    # T4: (A00 + A01)B11
    U_comb_4 = np.array([
        [1, 1],
        [0, 0]
    ])
    V_comb_4 = np.array([
        [0, 0],
        [0, 1]
    ])
    
    # T5: (A10 - A00)(B00 + B01)
    U_comb_5 = np.array([
        [-1, 0],
        [1, 0]
    ])
    V_comb_5 = np.array([
        [1, 1],
        [0, 0]
    ])
    
    # T6: (A01 - A11)(B10 + B11)
    U_comb_6 = np.array([
        [0, 1],
        [0, -1]
    ])
    V_comb_6 = np.array([
        [0, 0],
        [1, 1]
    ])
    
    U_combs = [U_comb_0, U_comb_1, U_comb_2, U_comb_3, U_comb_4, U_comb_5, U_comb_6]
    V_combs = [V_comb_0, V_comb_1, V_comb_2, V_comb_3, V_comb_4, V_comb_5, V_comb_6]
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for t_idx in range(7):
        U_comb = U_combs[t_idx]
        V_comb = V_combs[t_idx]
        
        for b_idx in range(7):
            u_base, v_base, w_base = base_factors[b_idx]
            
            # Expand u_base (4 coeffs for A_block entries) to 16 coeffs for A
            # A_block entries are (0,0), (0,1), (1,0), (1,1)
            # These correspond to global indices:
            # Block (I, J) has entries at global (2I+r, 2J+c)
            
            # u_base[0] is coeff for A_block(0,0)
            # u_base[1] is coeff for A_block(0,1)
            # u_base[2] is coeff for A_block(1,0)
            # u_base[3] is coeff for A_block(1,1)
            
            # For each of these, we add the contribution to the global U vector
            # A_block(0,0) corresponds to global indices 0,1,4,5
            # A_block(0,1) corresponds to global indices 2,3,6,7
            # A_block(1,0) corresponds to global indices 8,9,12,13
            # A_block(1,1) corresponds to global indices 10,11,14,15
            
            # The linear combination for A is:
            # sum_{b} u_base[b] * (sum of A entries in block b weighted by U_comb)
            # Wait, U_comb is the linear combination of blocks to form the matrix X.
            # X is a 2x2 matrix.
            # X[0,0] = sum_{i,j} U_comb[0,0] * A_block(i,j) ? No.
            # U_comb is a 2x2 matrix where U_comb[r,c] is the coefficient of Block(r,c).
            # So X[r,c] = U_comb[r,c] * A_block(r,c) ?
            # No, Strassen combines blocks.
            # e.g. T0: X = A00 + A11.
            # X is a 2x2 matrix.
            # X[0,0] = A00[0,0] + A11[0,0]
            # X[0,1] = A00[0,1] + A11[0,1]
            # etc.
            
            # So for each entry (r,c) of the 2x2 matrix X:
            # X[r,c] = sum_{I,J} U_comb[I,J] * A_block(I,J)[r,c]
            
            # u_base[k] is the coefficient for X[k] (where k maps to (r,c) of X)
            # k=0 -> X[0,0]
            # k=1 -> X[0,1]
            # k=2 -> X[1,0]
            # k=3 -> X[1,1]
            
            # So the global coefficient for A[i,j] is:
            # sum_{k} u_base[k] * (coeff of A[i,j] in X[k])
            
            # Let's compute the global U vector for this term
            U_vec = np.zeros(16)
            V_vec = np.zeros(16)
            W_vec = np.zeros(16)
            
            # Map base index k to (r,c)
            base_indices = [(0,0), (0,1), (1,0), (1,1)]
            
            for k in range(4):
                if u_base[k] == 0 and v_base[k] == 0:
                    continue
                
                r, c = base_indices[k]
                
                # X[r,c] = sum_{I,J} U_comb[I,J] * A_block(I,J)[r,c]
                # Y[r,c] = sum_{I,J} V_comb[I,J] * B_block(I,J)[r,c]
                
                # For each block (I,J):
                for I in range(2):
                    for J in range(2):
                        coeff_U = U_comb[I, J]
                        coeff_V = V_comb[I, J]
                        
                        if coeff_U == 0 and coeff_V == 0:
                            continue
                        
                        # A_block(I,J)[r,c] is at global index (2I+r, 2J+c)
                        global_idx_A = (2*I + r) * 4 + (2*J + c)
                        global_idx_B = (2*I + r) * 4 + (2*J + c)
                        
                        U_vec[global_idx_A] += u_base[k] * coeff_U
                        V_vec[global_idx_B] += v_base[k] * coeff_V
            
            # Now handle W
            # C_block(r,c) = sum_{k} w_base[k] * Z_k
            # Z_k is the k-th entry of the product matrix Z = X Y
            # But we are decomposing the product.
            # The output of the base Strassen is a 2x2 matrix Z.
            # Z[r,c] = sum_{k} w_base[k] * M_k
            # Wait, w_base is defined such that C = sum M_k * w_k?
            # In my base_factors, w_k is the vector of coeffs for C.
            # So C[r,c] = sum_k w_base[k][r,c] * M_k
            # Here M_k is the scalar product.
            # So the contribution to C_block(I,J) is:
            # C_block(I,J)[r,c] += sum_k w_base[k][r,c] * M_k
            # But M_k is the scalar product for the k-th term of the base Strassen.
            # And M_k is multiplied by 1 (it's the scalar multiplication).
            
            # So for the global W vector:
            # The scalar multiplication M_k contributes to C_block(I,J)[r,c] with coefficient w_base[k][r,c].
            # But which block I,J?
            # The Strassen at the top level produces C_blocks.
            # C_block(I,J) is formed by the combination of M_top_level?
            # No, the top level Strassen produces 7 matrices M_top.
            # Each M_top is a 2x2 matrix.
            # We decompose each M_top using base Strassen.
            # The result of M_top decomposition is the matrix M_top.
            # So C_block(I,J) is NOT directly M_top.
            # C = AB.
            # The top level Strassen computes C from M_top.
            # C_block(I,J) = linear combination of M_top.
            
            # I need the W_comb for the top level.
            # C00 = M0 - M1 + M3 - M6
            # C01 = M2 + M4
            # C10 = M1 + M5
            # C11 = M0 - M3 + M5 - M4
            
            # Let's define W_comb for each top level term t_idx
            # W_comb[t_idx] is a 2x2 matrix of coefficients for C_blocks
            
            if t_idx == 0: # M0
                W_comb = np.array([
                    [1, 0],
                    [0, 1]
                ])
            elif t_idx == 1: # M1
                W_comb = np.array([
                    [-1, 0],
                    [1, 0]
                ])
            elif t_idx == 2: # M2
                W_comb = np.array([
                    [0, 1],
                    [0, 0]
                ])
            elif t_idx == 3: # M3
                W_comb = np.array([
                    [1, 0],
                    [0, -1]
                ])
            elif t_idx == 4: # M4
                W_comb = np.array([
                    [0, 0],
                    [0, -1]
                ])
            elif t_idx == 5: # M5
                W_comb = np.array([
                    [0, 0],
                    [1, -1]
                ])
            elif t_idx == 6: # M6
                W_comb = np.array([
                    [0, 0],
                    [1, 1]
                ])
            
            # Now, for the base Strassen term b_idx:
            # The scalar multiplication M_k contributes to Z[r,c] with coeff w_base[k][r,c]
            # Z is the matrix M_top.
            # M_top contributes to C_block(I,J) with coeff W_comb[I,J]
            
            # So the global coefficient for C[i,j] is:
            # sum_{I,J} W_comb[I,J] * (coeff of C_block(I,J)[r,c] in M_top)
            # Coeff of C_block(I,J)[r,c] in M_top is w_base[k][r,c]
            
            # Wait, w_base[k] is a vector of 4 coeffs for the 4 entries of C (which is M_top).
            # So w_base[k][r,c] is the coeff for M_top[r,c].
            
            for k in range(4):
                if w_base[k] == 0:
                    continue
                r, c = base_indices[k]
                
                for I in range(2):
                    for J in range(2):
                        coeff_W = W_comb[I, J]
                        if coeff_W == 0:
                            continue
                        
                        global_idx_C = (2*I + r) * 4 + (2*J + c)
                        W_vec[global_idx_C] += w_base[k][r,c] * coeff_W
            
            U[:, col] = U_vec
            V[:, col] = V_vec
            W[:, col] = W_vec
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
