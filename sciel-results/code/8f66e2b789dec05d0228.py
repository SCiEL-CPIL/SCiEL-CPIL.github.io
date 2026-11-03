import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    dim_U = n * m  # 16
    dim_V = m * p  # 20
    dim_W = p * n  # 20
    
    # We will build the decomposition in two parts:
    # 1. 4x4 * 4x4 part (columns 0-3 of B, rows 0-3 of C) using Strassen (Rank 49)
    # 2. 4x4 * 4x1 part (column 4 of B, column 4 of C) using Schoolbook (Rank 16)
    
    # Total Rank = 49 + 16 = 65
    rank = 65
    
    U = np.zeros((dim_U, rank), dtype=int)
    V = np.zeros((dim_V, rank), dtype=int)
    W = np.zeros((dim_W, rank), dtype=int)
    
    idx = 0
    
    # --- Part 1: Strassen for 4x4 * 4x4 ---
    
    # Helper to add a term to U, V, W
    # a_coeffs: dict {global_U_index: coeff}
    # b_coeffs: dict {global_V_index: coeff}
    # c_coeffs: dict {global_W_index: coeff}
    def add_term(a_coeffs, b_coeffs, c_coeffs, idx):
        for u_idx, coeff_a in a_coeffs.items():
            for v_idx, coeff_b in b_coeffs.items():
                for w_idx, coeff_c in c_coeffs.items():
                    # We need to add coeff_a * coeff_b * coeff_c to the tensor
                    # But U, V, W are factors.
                    # The tensor is sum_r U[:,r] V[:,r] W[:,r]
                    # So for a specific term r, U[i,r] = coeff_a, V[j,r] = coeff_b, W[k,r] = coeff_c
                    # This works if each (a,b,c) combination is unique for a term.
                    # In Strassen, the terms are structured such that we can just assign.
                    pass
        # Actually, the standard way is:
        # For each term in the decomposition, we define U[:, idx], V[:, idx], W[:, idx].
        # The term is U[:, idx] * V[:, idx] * W[:, idx] (outer product)
        # This corresponds to a rank-1 tensor.
        # We need to sum these rank-1 tensors.
        
        # Wait, the add_term logic above is wrong for the factor matrices.
        # The factor matrices U, V, W define the rank-1 terms directly.
        # U[i, idx] is the coefficient of A[i] in term idx.
        # V[j, idx] is the coefficient of B[j] in term idx.
        # W[k, idx] is the coefficient of C[k] in term idx.
        
        # So, if a term is (sum a_i A_i) * (sum b_j B_j) -> (sum c_k C_k)
        # Then U[i, idx] = a_i, V[j, idx] = b_j, W[k, idx] = c_k.
        
        # Let's refine add_term to take these dicts and fill the column.
        for u_idx, coeff_a in a_coeffs.items():
            U[u_idx, idx] = coeff_a
        for v_idx, coeff_b in b_coeffs.items():
            V[v_idx, idx] = coeff_b
        for w_idx, coeff_c in c_coeffs.items():
            W[w_idx, idx] = coeff_c
        return idx + 1

    # Helper to get global index for A
    def A_idx(i, j):
        return i * m + j

    # Helper to get global index for B
    def B_idx(j, k):
        return j * p + k

    # Helper to get global index for C
    def C_idx(k, i):
        return k * n + i

    # Define the 7 Strassen terms for 2x2 * 2x2
    # Each term is a tuple: (X_coeffs, Y_coeffs, Z_coeffs)
    # X_coeffs: {X_idx: coeff} for X = [[0,0],[0,1],[1,0],[1,1]]
    # Y_coeffs: {Y_idx: coeff} for Y
    # Z_coeffs: {Z_idx: coeff} for Z
    # Note: The term is (dot(X, X_coeffs)) * (dot(Y, Y_coeffs)) -> dot(Z, Z_coeffs)
    # This is a rank-1 tensor for the 2x2 multiplication.
    
    strassen_2x2_terms = [
        # M1 = (X00+X11)(Y00+Y11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2 = (X10+X11)Y00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # M3 = X00(Y01-Y11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1}
        ),
        # M4 = X11(Y10-Y00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M5 = (X00+X10)Y11
        (
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(0,0): -1, (1,1): 1}
        ),
        # M6 = (X01-X00)(Y00+Y10)
        (
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1},
            {(1,1): 1}
        ),
        # M7 = (X11-X01)(Y01+Y11)
        (
            {(1,1): 1, (0,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1}
        ),
    ]
    
    # The 7 Macro terms for 4x4 * 4x4
    # Each macro term is defined by linear combinations of A and B blocks.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # Note: A_ij is a 2x2 matrix.
    # A00 corresponds to A[0:2, 0:2]
    
    # We need to map the local 2x2 indices (0,0)..(1,1) of X and Y
    # to the global 4x4 indices of A and B.
    
    # Macro Term 1: X = A00 + A11, Y = B00 + B11
    # X[i,j] = A[i,j] + A[i+2, j+2]
    # Y[i,j] = B[i,j] + B[i+2, j+2]
    
    # Let's create a list of 7 macro definitions.
    # Each definition is a function that maps local X(i,j) to global A indices,
    # and local Y(i,j) to global B indices.
    
    def get_macro_mappings():
        mappings = []
        
        # M1: X=A00+A11, Y=B00+B11
        mappings.append({
            'X': lambda i, j: [A_idx(i, j), A_idx(i+2, j+2)],
            'Y': lambda i, j: [B_idx(i, j), B_idx(i+2, j+2)],
            'signs': [1, 1] # X = 1*X + 1*X, Y = 1*Y + 1*Y
        })
        # Actually, the signs are embedded in the coefficients.
        # Let's just return the lists of global indices and their coefficients for X and Y.
        
        # This approach is getting messy. Let's just explicitly construct the 49 terms.
        pass
        
    # Simpler approach:
    # Iterate through the 7 Strassen 2x2 terms.
    # For each, we need to know which global A/B indices contribute.
    
    # Let's define the 7 macro "subproblems" explicitly.
    # Macro k defines X and Y as linear combinations of A and B.
    # Then we apply the 7 Strassen 2x2 decompositions to that Macro.
    
    # Macro 1: X = A00 + A11, Y = B00 + B11
    # X[i,j] = A[i,j] + A[i+2,j+2]
    # Y[i,j] = B[i,j] + B[i+2,j+2]
    
    # For a given Strassen 2x2 term (X_coeffs, Y_coeffs, Z_coeffs):
    # The global A coeffs are sum over (xi, xj) in X_coeffs of X_coeffs[xi,xj] * (coeff of A in X[xi,xj])
    
    # Let's pre-compute the global coefficient dictionaries for the 7 macros.
    
    macros = []
    
    # Helper to build global coeff dict from local coeff dict and mapping
    def expand_local_to_global(local_coeffs, mapping_func, sign_func=None):
        global_coeffs = {}
        for (li, lj), coeff in local_coeffs.items():
            # mapping_func(li, lj) returns list of global indices
            # sign_func(li, lj, idx_in_list) returns sign?
            # Let's assume mapping_func returns list of (global_idx, sign)
            for g_idx, g_sign in mapping_func(li, lj):
                global_coeffs[g_idx] = global_coeffs.get(g_idx, 0) + coeff * g_sign
        return global_coeffs

    # Define the 7 macros
    # Each macro is a dict with 'X_map' and 'Y_map'
    # X_map(li, lj) -> [(global_A_idx, sign), ...]
    
    # Macro 1: X = A00 + A11
    def X_map_1(li, lj):
        return [(A_idx(li, lj), 1), (A_idx(li+2, lj+2), 1)]
    def Y_map_1(li, lj):
        return [(B_idx(li, lj), 1), (B_idx(li+2, lj+2), 1)]
    macros.append({'X_map': X_map_1, 'Y_map': Y_map_1})
    
    # Macro 2: X = A10 + A11, Y = B00
    def X_map_2(li, lj):
        # A10 is A[2:4, 0:2], A11 is A[2:4, 2:4]
        # X[0,0] = A[2,0] + A[2,2]
        # X[0,1] = A[2,1] + A[2,3]
        # X[1,0] = A[3,0] + A[3,2]
        # X[1,1] = A[3,1] + A[3,3]
        return [(A_idx(2+li, lj), 1), (A_idx(2+li, 2+lj), 1)]
    def Y_map_2(li, lj):
        # Y = B00
        # Y[0,0] = B[0,0]
        # Y[0,1] = B[0,1]
        # Y[1,0] = B[1,0]
        # Y[1,1] = B[1,1]
        return [(B_idx(li, lj), 1)]
    macros.append({'X_map': X_map_2, 'Y_map': Y_map_2})
    
    # Macro 3: X = A00, Y = B01 - B11
    def X_map_3(li, lj):
        # X = A00
        return [(A_idx(li, lj), 1)]
    def Y_map_3(li, lj):
        # Y = B01 - B11
        # Y[0,0] = B[0,1] - B[2,1]
        # Y[0,1] = B[0,2] - B[2,2]
        # Y[1,0] = B[1,1] - B[3,1]
        # Y[1,1] = B[1,2] - B[3,2]
        return [(B_idx(li, 1+lj), 1), (B_idx(2+li, 1+lj), -1)]
    macros.append({'X_map': X_map_3, 'Y_map': Y_map_3})
    
    # Macro 4: X = A11, Y = B10 - B00
    def X_map_4(li, lj):
        # X = A11
        return [(A_idx(2+li, 2+lj), 1)]
    def Y_map_4(li, lj):
        # Y = B10 - B00
        # Y[0,0] = B[2,0] - B[0,0]
        # Y[0,1] = B[2,1] - B[0,1]
        # Y[1,0] = B[3,0] - B[1,0]
        # Y[1,1] = B[3,1] - B[1,1]
        return [(B_idx(2+li, lj), 1), (B_idx(li, lj), -1)]
    macros.append({'X_map': X_map_4, 'Y_map': Y_map_4})
    
    # Macro 5: X = A00 + A10, Y = B11
    def X_map_5(li, lj):
        # X = A00 + A10
        # X[0,0] = A[0,0] + A[2,0]
        # X[0,1] = A[0,1] + A[2,1]
        # X[1,0] = A[1,0] + A[3,0]
        # X[1,1] = A[1,1] + A[3,1]
        return [(A_idx(li, lj), 1), (A_idx(2+li, lj), 1)]
    def Y_map_5(li, lj):
        # Y = B11
        return [(B_idx(2+li, 2+lj), 1)]
    macros.append({'X_map': X_map_5, 'Y_map': Y_map_5})
    
    # Macro 6: X = A01 - A00, Y = B00 + B10
    def X_map_6(li, lj):
        # X = A01 - A00
        # X[0,0] = A[0,1] - A[0,0]
        # X[0,1] = A[0,2] - A[0,1]
        # X[1,0] = A[1,1] - A[1,0]
        # X[1,1] = A[1,2] - A[1,1]
        return [(A_idx(li, 1+lj), 1), (A_idx(li, lj), -1)]
    def Y_map_6(li, lj):
        # Y = B00 + B10
        # Y[0,0] = B[0,0] + B[2,0]
        # Y[0,1] = B[0,1] + B[2,1]
        # Y[1,0] = B[1,0] + B[3,0]
        # Y[1,1] = B[1,1] + B[3,1]
        return [(B_idx(li, lj), 1), (B_idx(2+li, lj), 1)]
    macros.append({'X_map': X_map_6, 'Y_map': Y_map_6})
    
    # Macro 7: X = A11 - A01, Y = B01 + B11
    def X_map_7(li, lj):
        # X = A11 - A01
        # X[0,0] = A[2,2] - A[0,2]
        # X[0,1] = A[2,3] - A[0,3]
        # X[1,0] = A[3,2] - A[1,2]
        # X[1,1] = A[3,3] - A[1,3]
        return [(A_idx(2+li, 2+lj), 1), (A_idx(li, 2+lj), -1)]
    def Y_map_7(li, lj):
        # Y = B01 + B11
        # Y[0,0] = B[0,1] + B[2,1]
        # Y[0,1] = B[0,2] + B[2,2]
        # Y[1,0] = B[1,1] + B[3,1]
        # Y[1,1] = B[1,2] + B[3,2]
        return [(B_idx(li, 1+lj), 1), (B_idx(2+li, 1+lj), 1)]
    macros.append({'X_map': X_map_7, 'Y_map': Y_map_7})

    # Now, iterate through macros and Strassen 2x2 terms
    for macro in macros:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_2x2_terms:
            # Expand X and Y to global A and B coeffs
            a_coeffs = expand_local_to_global(x_coeffs, macro['X_map'])
            b_coeffs = expand_local_to_global(y_coeffs, macro['Y_map'])
            
            # The Z coeffs are for the 2x2 result of the macro multiplication.
            # We need to map these to the global C indices.
            # The result of Macro k is added to the global C with specific signs.
            
            # Let's determine the global C coeffs.
            # This depends on the macro index.
            # We need the assembly formulas.
            
            # This part is tricky. We need to know which C block the macro result goes into, and with what sign.
            # Let's just use the standard Strassen assembly for the 4x4 case directly.
            
            # Actually, the 7 macros *are* the Strassen terms for the 4x4 case.
            # The "Z" in the 2x2 Strassen decomposition is the 2x2 result of that macro.
            # But the macro result is not a single C block, it contributes to all 4 C blocks.
            
            # Wait, no.
            # In the standard Strassen algorithm for 2x2 matrices of matrices:
            # We compute 7 matrix products M1..M7.
            # Each Mk is a 2x2 matrix.
            # The final C blocks are linear combinations of these Mk.
            
            # So, for each macro k, we compute the 2x2 matrix Mk.
            # Then we add the contributions of Mk to the global C.
            
            # How do we map the local Z (2x2) of the macro to the global C?
            # The local Z indices (0,0)..(1,1) correspond to the 2x2 entries of the macro result.
            # But the macro result Mk is a 2x2 matrix.
            # Its entries are added to the global C blocks.
            
            # Let's look at the assembly:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Here M1 is a 2x2 matrix.
            # So C00 (a 2x2 block) = M1 (2x2) + M4 (2x2) ...
            
            # So, for a specific macro k (which computes Mk), the local Z (2x2)
            # is added to the global C blocks with coefficients determined by the assembly formulas.
            
            # We need to define, for each macro k, the coefficients for the 4 global C blocks.
            # And for each of the 4 entries of the local Z (2x2), which global C entry it maps to.
            
            # The local Z entry (zi, zj) maps to global C entry (bi*2+zi, bj*2+zj) ?
            # No, the blocks are:
            # C00 is C[0:2, 0:2]
            # C01 is C[0:2, 2:4]
            # C10 is C[2:4, 0:2]
            # C11 is C[2:4, 2:4]
            
            # So local Z (zi, zj) in block (bi, bj) maps to global C (bi*2+zi, bj*2+zj).
            
            # Let's define the assembly coefficients for each macro.
            # macro_idx 0 (M1):
            #   C00: +1, C01: 0, C10: 0, C11: +1
            # macro_idx 1 (M2):
            #   C00: 0, C01: 0, C10: +1, C11: -1
            # macro_idx 2 (M3):
            #   C00: 0, C01: +1, C10: 0, C11: +1
            # macro_idx 3 (M4):
            #   C00: +1, C01: 0, C10: +1, C11: 0
            # macro_idx 4 (M5):
            #   C00: -1, C01: +1, C10: 0, C11: +1
            # macro_idx 5 (M6):
            #   C00: 0, C01: 0, C10: 0, C11: +1
            # macro_idx 6 (M7):
            #   C00: +1, C01: 0, C10: 0, C11: -1
            
            assembly = [
                # M1
                {(0,0): 1, (1,1): 1},
                # M2
                {(1,0): 1, (1,1): -1},
                # M3
                {(0,1): 1, (1,1): 1},
                # M4
                {(0,0): 1, (1,0): 1},
                # M5
                {(0,0): -1, (0,1): 1, (1,1): 1},
                # M6
                {(1,1): 1},
                # M7
                {(0,0): 1, (1,1): -1},
            ]
            
            current_assembly = assembly[macros.index(macro)]
            
            c_coeffs = {}
            for (zi, zj), coeff_z in z_coeffs.items():
                # zi, zj are local 2x2 indices
                # We need to add to global C blocks based on current_assembly
                for (bi, bj), coeff_block in current_assembly.items():
                    global_ci = bi * 2 + zi
                    global_cj = bj * 2 + zj
                    g_idx = C_idx(global_cj, global_ci) # C[k*n+i] -> C[global_cj * 4 + global_ci]
                    c_coeffs[g_idx] = c_coeffs.get(g_idx, 0) + coeff_z * coeff_block
            
            # Now we have global a_coeffs, b_coeffs, c_coeffs
            idx = add_term(a_coeffs, b_coeffs, c_coeffs, idx)

    # --- Part 2: Schoolbook for 4x4 * 4x1 ---
    for i in range(4):
        for j in range(4):
            U[A_idx(i, j), idx] = 1
            V[B_idx(j, 4), idx] = 1
            W[C_idx(4, i), idx] = 1
            idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
