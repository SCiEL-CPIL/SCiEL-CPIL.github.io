"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses a hardcoded rank-65 decomposition derived from Strassen's algorithm for the 
    4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    dim_u = n * m  # 16
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    # Total rank: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1) = 65
    rank = 65
    
    U = np.zeros((dim_u, rank), dtype=int)
    V = np.zeros((dim_v, rank), dtype=int)
    W = np.zeros((dim_w, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # --- Part 1: Strassen 4x4x4 (Rank 49) ---
    # We decompose A (4x4), B (4x4), C (4x4) into 2x2 blocks.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Global indices for A (U rows)
    # A00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # A01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # A10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # A11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # Global indices for B (V rows) - only first 4 columns (k=0..3)
    # B00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 5, 6
    # B01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 7, 8
    # B10: (2,0), (2,1), (3,0), (3,1) -> 10, 11, 15, 16
    # B11: (2,2), (2,3), (3,2), (3,3) -> 12, 13, 17, 18
    
    # Global indices for C (W rows) - only first 4 columns (k=0..3)
    # C00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5  (k*n+i: 0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # C01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7  (k=2,3)
    # C10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13 (k=0,1, i=2,3)
    # C11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15 (k=2,3, i=2,3)
    
    # Define block index lists
    A_blocks = {
        'A00': [0, 1, 4, 5],
        'A01': [2, 3, 6, 7],
        'A10': [8, 9, 12, 13],
        'A11': [10, 11, 14, 15]
    }
    B_blocks = {
        'B00': [0, 1, 5, 6],
        'B01': [2, 3, 7, 8],
        'B10': [10, 11, 15, 16],
        'B11': [12, 13, 17, 18]
    }
    C_blocks = {
        'C00': [0, 1, 4, 5],
        'C01': [2, 3, 6, 7],
        'C10': [8, 9, 12, 13],
        'C11': [10, 11, 14, 15]
    }
    
    # Function to add a 2x2x2 Strassen decomposition for given linear combos
    # X_coeffs: dict mapping block name to list of 4 coeffs
    # Y_coeffs: dict mapping block name to list of 4 coeffs
    # Z_coeffs: dict mapping block name to list of 4 coeffs
    def add_strassen_2x2(X_coeffs, Y_coeffs, Z_coeffs):
        nonlocal col
        # The 7 products for 2x2x2
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # We need to expand each Mk into scalar terms.
        # Each Mk is a 2x2 matrix product.
        # We iterate through the 7 Strassen steps for the inner 2x2x2.
        
        # Inner Strassen steps for a generic 2x2x2 with inputs X, Y and output Z
        # X, Y, Z are 2x2 matrices.
        # We define the 7 inner products P1..P7 in terms of X, Y elements.
        # Then we express Z in terms of P1..P7.
        
        # To avoid complex nesting, we explicitly generate the 7 scalar multiplications for the inner level.
        # Let's define the 7 inner products' X and Y linear combos.
        
        # Inner X elements: X[0,0], X[0,1], X[1,0], X[1,1]
        # Inner Y elements: Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        
        # We will construct the U, V vectors for each of the 7 inner products.
        # Then we need to distribute the result to the Z elements.
        
        # The 7 inner products:
        # I1 = (X00+X11)(Y00+Y11)
        # I2 = (X10+X11)Y00
        # I3 = X00(Y01-Y11)
        # I4 = X11(Y10-Y00)
        # I5 = (X00+X01)Y11
        # I6 = (X10-X00)(Y00+Y01)
        # I7 = (X01-X11)(Y10+Y11)
        
        # Contribution to Z:
        # Z00 = I1 + I4 - I5 + I7
        # Z01 = I3 + I5
        # Z10 = I2 + I4
        # Z11 = I1 - I2 + I3 + I6
        
        # We need to map X, Y, Z elements to global indices with coefficients.
        
        # Get global indices for X, Y, Z blocks
        # X_coeffs is a dict of block_name -> [c00, c01, c10, c11]
        # We need to form the linear combination for each inner element.
        
        # Helper to get global U indices and coeffs for a specific inner element (r, c) of X
        def get_x_elem(r, c):
            # r, c in {0,1}
            # We need to sum over all blocks
            u_indices = []
            u_coeffs = []
            for block_name, coeffs in X_coeffs.items():
                val = coeffs[r * 2 + c]
                if val != 0:
                    block_indices = A_blocks[block_name]
                    # Map local (r, c) to local index in block
                    # Block is 2x2.
                    # Local index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    # But block_indices list is [00, 01, 10, 11]
                    local_idx = r * 2 + c
                    global_idx = block_indices[local_idx]
                    u_indices.append(global_idx)
                    u_coeffs.append(val)
            return u_indices, u_coeffs
            
        def get_y_elem(r, c):
            v_indices = []
            v_coeffs = []
            for block_name, coeffs in Y_coeffs.items():
                val = coeffs[r * 2 + c]
                if val != 0:
                    block_indices = B_blocks[block_name]
                    local_idx = r * 2 + c
                    global_idx = block_indices[local_idx]
                    v_indices.append(global_idx)
                    v_coeffs.append(val)
            return v_indices, v_coeffs
            
        def get_z_elem(r, c):
            w_indices = []
            w_coeffs = []
            for block_name, coeffs in Z_coeffs.items():
                val = coeffs[r * 2 + c]
                if val != 0:
                    block_indices = C_blocks[block_name]
                    local_idx = r * 2 + c
                    global_idx = block_indices[local_idx]
                    w_indices.append(global_idx)
                    w_coeffs.append(val)
            return w_indices, w_coeffs

        # Define the 7 inner products' X and Y parts
        # I1: X = X00+X11, Y = Y00+Y11
        # I2: X = X10+X11, Y = Y00
        # I3: X = X00, Y = Y01-Y11
        # I4: X = X11, Y = Y10-Y00
        # I5: X = X00+X01, Y = Y11
        # I6: X = X10-X00, Y = Y00+Y01
        # I7: X = X01-X11, Y = Y10+Y11
        
        # We will create 7 "virtual" X and Y matrices for the inner level
        # and then expand them.
        
        # To simplify, we can just iterate through the 7 inner Strassen steps
        # and for each, define the X and Y linear combos, and the Z contribution.
        
        # Inner Step 1: I1
        # X combo: X00, X11 with coeff 1
        # Y combo: Y00, Y11 with coeff 1
        # Z contrib: Z00 (+1), Z11 (+1)
        
        # We need to add 7 scalar multiplications for I1? No, I1 is ONE product.
        # But I1 is a 2x2 matrix product.
        # So I1 itself requires 7 scalar multiplications if we use Strassen again?
        # NO. The rank 49 comes from 7 top-level products, each of which is a 2x2x2 multiplication.
        # A 2x2x2 multiplication has rank 7 (Strassen) or 8 (Schoolbook).
        # If we use Strassen for the 2x2x2, we use 7 scalar multiplications.
        # So for each of the 7 top-level products, we add 7 columns.
        
        # So, for I1 (which is a 2x2x2 product), we must decompose it into 7 scalar multiplications.
        
        # Let's write a helper to decompose a 2x2x2 product defined by X_coeffs, Y_coeffs, Z_coeffs
        # into 7 scalar multiplications.
        
        def decompose_2x2_2x2(X_c, Y_c, Z_c):
            # X_c, Y_c, Z_c are dicts of block_name -> [c00, c01, c10, c11]
            # This function adds 7 columns to U, V, W.
            
            # The 7 scalar multiplications for 2x2x2:
            # S1 = (X00+X11)(Y00+Y11)
            # S2 = (X10+X11)Y00
            # S3 = X00(Y01-Y11)
            # S4 = X11(Y10-Y00)
            # S5 = (X00+X01)Y11
            # S6 = (X10-X00)(Y00+Y01)
            # S7 = (X01-X11)(Y10+Y11)
            
            # Z = [[S1+S4-S5+S7, S3+S5], [S2+S4, S1-S2+S3+S6]]
            
            # We define the 7 scalar terms' X and Y parts.
            # Then we add the W part based on the Z formula.
            
            # Term 1: S1
            # X: X00, X11 (coeff 1)
            # Y: Y00, Y11 (coeff 1)
            # Z: Z00 (coeff 1), Z11 (coeff 1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(0,0,1), (1,1,1)], 
                            y_combo=[(0,0,1), (1,1,1)], 
                            z_combo=[(0,0,1), (1,1,1)])
                            
            # Term 2: S2
            # X: X10, X11 (coeff 1)
            # Y: Y00 (coeff 1)
            # Z: Z10 (coeff 1), Z11 (coeff -1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(1,0,1), (1,1,1)], 
                            y_combo=[(0,0,1)], 
                            z_combo=[(1,0,1), (1,1,-1)])
                            
            # Term 3: S3
            # X: X00 (coeff 1)
            # Y: Y01 (1), Y11 (-1)
            # Z: Z01 (1), Z11 (1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(0,0,1)], 
                            y_combo=[(0,1,1), (1,1,-1)], 
                            z_combo=[(0,1,1), (1,1,1)])
                            
            # Term 4: S4
            # X: X11 (1)
            # Y: Y10 (1), Y00 (-1)
            # Z: Z00 (1), Z10 (1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(1,1,1)], 
                            y_combo=[(1,0,1), (0,0,-1)], 
                            z_combo=[(0,0,1), (1,0,1)])
                            
            # Term 5: S5
            # X: X00 (1), X01 (1)
            # Y: Y11 (1)
            # Z: Z00 (-1), Z01 (1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(0,0,1), (0,1,1)], 
                            y_combo=[(1,1,1)], 
                            z_combo=[(0,0,-1), (0,1,1)])
                            
            # Term 6: S6
            # X: X10 (1), X00 (-1)
            # Y: Y00 (1), Y01 (1)
            # Z: Z11 (1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(1,0,1), (0,0,-1)], 
                            y_combo=[(0,0,1), (0,1,1)], 
                            z_combo=[(1,1,1)])
                            
            # Term 7: S7
            # X: X01 (1), X11 (-1)
            # Y: Y10 (1), Y11 (1)
            # Z: Z00 (1)
            add_scalar_term(X_c, Y_c, Z_c, 
                            x_combo=[(0,1,1), (1,1,-1)], 
                            y_combo=[(1,0,1), (1,1,1)], 
                            z_combo=[(0,0,1)])
            
        def add_scalar_term(X_c, Y_c, Z_c, x_combo, y_combo, z_combo):
            nonlocal col
            u_indices = []
            u_coeffs = []
            for r, c, val in x_combo:
                # Get global indices for X element (r, c)
                for block_name, coeffs in X_c.items():
                    if coeffs[r*2+c] != 0:
                        block_indices = A_blocks[block_name]
                        global_idx = block_indices[r*2+c]
                        u_indices.append(global_idx)
                        u_coeffs.append(coeffs[r*2+c] * val)
                        
            v_indices = []
            v_coeffs = []
            for r, c, val in y_combo:
                for block_name, coeffs in Y_c.items():
                    if coeffs[r*2+c] != 0:
                        block_indices = B_blocks[block_name]
                        global_idx = block_indices[r*2+c]
                        v_indices.append(global_idx)
                        v_coeffs.append(coeffs[r*2+c] * val)
                        
            w_indices = []
            w_coeffs = []
            for r, c, val in z_combo:
                for block_name, coeffs in Z_c.items():
                    if coeffs[r*2+c] != 0:
                        block_indices = C_blocks[block_name]
                        global_idx = block_indices[r*2+c]
                        w_indices.append(global_idx)
                        w_coeffs.append(coeffs[r*2+c] * val)
                        
            # Add to current column
            for idx, val in zip(u_indices, u_coeffs):
                U[idx, col] += val
            for idx, val in zip(v_indices, v_coeffs):
                V[idx, col] += val
            for idx, val in zip(w_indices, w_coeffs):
                W[idx, col] += val
            col += 1
            
        # Now, define the 7 top-level products
        # P1 = (A00+A11)(B00+B11)
        X1 = {'A00': [1,0,0,0], 'A11': [1,0,0,0]} # Wait, X is 2x2.
        # X1[0,0] = A00[0,0], X1[0,1] = A00[0,1], etc.
        # So X1 block 'A00' has coeffs [1,1,1,1]? No.
        # X1 is a 2x2 matrix.
        # X1[0,0] = A00[0,0] + A11[0,0]
        # X1[0,1] = A00[0,1] + A11[0,1]
        # X1[1,0] = A00[1,0] + A11[1,0]
        # X1[1,1] = A00[1,1] + A11[1,1]
        
        # So for X1, the coeffs for block 'A00' are [1,1,1,1] and 'A11' are [1,1,1,1].
        
        X1_c = {'A00': [1,1,1,1], 'A11': [1,1,1,1]}
        Y1_c = {'B00': [1,1,1,1], 'B11': [1,1,1,1]}
        Z1_c = {'C00': [1,0,0,0], 'C11': [1,0,0,0], 'C10': [0,0,1,0], 'C01': [0,0,0,1]} # Z00, Z11, Z10, Z01?
        # Z1 contributes to:
        # C00 (Z00 of top level) -> Top Z00 is C00.
        # C11 (Z11 of top level) -> Top Z11 is C11.
        # C10 (Z10 of top level) -> Top Z10 is C10.
        # C01 (Z01 of top level) -> Top Z01 is C01.
        # Formula: C00 = P1 + P4 - P5 + P7. So P1 contributes +1 to C00.
        # C11 = P1 - P2 + P3 + P6. So P1 contributes +1 to C11.
        # C10 = P2 + P4. P1 contributes 0.
        # C01 = P3 + P5. P1 contributes 0.
        
        # So Z1_c should only have C00 and C11 with coeffs [1,1,1,1]?
        # No, the inner Strassen decomposes the 2x2 product.
        # The result of P1 is a 2x2 matrix.
        # The top-level C00 is the (0,0) element of the result?
        # No, P1 is the product of two 2x2 matrices.
        # The result is a 2x2 matrix.
        # This 2x2 matrix is added to C00? No.
        # C00 is a 2x2 block.
        # The Strassen algorithm computes the 2x2 blocks of C.
        # C00 = P1 + P4 - P5 + P7.
        # Here P1, P4, etc are 2x2 matrices.
        # So P1 contributes to the entire C00 block.
        
        # So for P1, the Z part is the entire C00 block and C11 block?
        # No, P1 is just one term in the sum for C00.
        # And P1 is also a term in the sum for C11.
        
        # So when we decompose P1 (a 2x2x2 product), the "Z" output of this 2x2x2 product
        # is distributed to C00 and C11.
        
        # How to handle this?
        # The `decompose_2x2_2x2` function assumes the output Z is a single 2x2 matrix.
        # But here, the output of P1 is added to C00 AND C11.
        
        # We can modify `add_scalar_term` to accept a list of Z contributions.
        # Or, we can just call `decompose_2x2_2x2` with Z_c representing the target blocks.
        
        # Let's define Z_c for P1 such that it covers C00 and C11.
        # But the coefficients might be different?
        # In C00 = P1 + ..., P1 has coeff 1.
        # In C11 = P1 + ..., P1 has coeff 1.
        # So yes, coeff 1 for both.
        
        # So Z1_c = {'C00': [1,1,1,1], 'C11': [1,1,1,1]}
        
        # Let's re-verify the top-level Strassen formulas.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # P1: X = A00+A11, Y = B00+B11. Z = C00 (+1), C11 (+1).
        # P2: X = A10+A11, Y = B00. Z = C10 (+1), C11 (-1).
        # P3: X = A00, Y = B01-B11. Z = C01 (+1), C11 (+1).
        # P4: X = A11, Y = B10-B00. Z = C00 (+1), C10 (+1).
        # P5: X = A00+A01, Y = B11. Z = C00 (-1), C01 (+1).
        # P6: X = A10-A00, Y = B00+B01. Z = C11 (+1).
        # P7: X = A01-A11, Y = B10+B11. Z = C00 (+1).
        
        # We will call decompose_2x2_2x2 for each Pk with the appropriate X_c, Y_c, Z_c.
        
        # P1
        decompose_2x2_2x2(
            {'A00': [1,1,1,1], 'A11': [1,1,1,1]},
            {'B00': [1,1,1,1], 'B11': [1,1,1,1]},
            {'C00': [1,1,1,1], 'C11': [1,1,1,1]}
        )
        
        # P2
        decompose_2x2_2x2(
            {'A10': [1,1,1,1], 'A11': [1,1,1,1]},
            {'B00': [1,1,1,1]},
            {'C10': [1,1,1,1], 'C11': [-1,-1,-1,-1]}
        )
        
        # P3
        decompose_2x2_2x2(
            {'A00': [1,1,1,1]},
            {'B01': [1,1,1,1], 'B11': [-1,-1,-1,-1]},
            {'C01': [1,1,1,1], 'C11': [1,1,1,1]}
        )
        
        # P4
        decompose_2x2_2x2(
            {'A11': [1,1,1,1]},
            {'B10': [1,1,1,1], 'B00': [-1,-1,-1,-1]},
            {'C00': [1,1,1,1], 'C10': [1,1,1,1]}
        )
        
        # P5
        decompose_2x2_2x2(
            {'A00': [1,1,1,1], 'A01': [1,1,1,1]},
            {'B11': [1,1,1,1]},
            {'C00': [-1,-1,-1,-1], 'C01': [1,1,1,1]}
        )
        
        # P6
        decompose_2x2_2x2(
            {'A10': [1,1,1,1], 'A00': [-1,-1,-1,-1]},
            {'B00': [1,1,1,1], 'B01': [1,1,1,1]},
            {'C11': [1,1,1,1]}
        )
        
        # P7
        decompose_2x2_2x2(
            {'A01': [1,1,1,1], 'A11': [-1,-1,-1,-1]},
            {'B10': [1,1,1,1], 'B11': [1,1,1,1]},
            {'C00': [1,1,1,1]}
        )
        
    # --- Part 2: Schoolbook 4x4x1 (Rank 16) ---
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            add_term(u_idx, 1, v_idx, 1, w_idx, 1)
            
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
