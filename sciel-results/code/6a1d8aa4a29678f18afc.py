import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition via Strassen squared
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to map the 49 scalar multiplications to the U, V, W factors.
    # Strassen for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # For 4x4, we apply this to the 2x2 blocks.
    # Let A blocks be A00, A01, A10, A11 (each 2x2)
    # Let B blocks be B00, B01, B10, B11 (each 2x2)
    
    # The 7 products at the top level are:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Each Pi is a 2x2 matrix product.
    # We decompose each Pi using Strassen 2x2.
    # So P1 contributes 7 scalar products.
    # P2 contributes 7 scalar products.
    # ...
    # Total 49.
    
    # We need to construct U, V, W such that
    # C = U * diag(products) * V^T ? No.
    # The decomposition is C_{ij} = sum_k U_{i,k} * V_{j,k} * W_{k, (i,j)?}
    # Wait, the tensor is T_{ijk} = sum_c U_{i,c} V_{j,c} W_{k,c}
    # Here i, j, k are indices for A, B, C.
    # A is n x m (4x4). Index i (row), j (col).
    # B is m x p (4x4). Index j (row), k (col).
    # C is p x n (4x4). Index k (row), i (col).
    # Wait, standard matrix mult C = A B.
    # C_{k,i} = sum_j A_{i,j} B_{j,k}.
    # The tensor indices are (i, j, k) for A, B, C.
    # The decomposition is T_{ijk} = sum_c U_{i,c} V_{j,c} W_{k,c}.
    # Note the order in the prompt:
    # U shape (n*m, rank) -> Index (i*m+j)? No, prompt says U[i*m+j, c].
    # V shape (m*p, rank) -> Index (j*p+k)? Prompt says V[j*p+k, c].
    # W shape (p*n, rank) -> Index (k*n+i)? Prompt says W[k*n+i, c].
    # And the trace convention W[k*n+i, r].
    
    # So for each rank-1 term c, we have:
    # U[i, j, c] = u_{ij}
    # V[j, k, c] = v_{jk}
    # W[k, i, c] = w_{ki}
    # And the contribution to C_{k,i} is u_{ij} * v_{jk} * w_{ki}.
    # Summing over c and j?
    # No, the tensor decomposition of matrix multiplication is:
    # T_{ijk} = delta_{j,j} (identity in j)?
    # Actually, the tensor for matrix mult is T_{i j k} = 1 if the indices match appropriately?
    # No, the bilinear form is C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has indices i, j, k.
    # T_{i j k} = 1 if we are looking at the term A_{ij} B_{jk} contributing to C_{ik}.
    # So for a fixed j, the term is A_{ij} B_{jk}.
    # This corresponds to a rank-1 tensor in the space of A, B, C?
    # A_{ij} is a basis vector in A^*. B_{jk} is a basis vector in B^*.
    # The product is A_{ij} \otimes B_{jk}.
    # This contributes to C_{ik}.
    # So the rank-1 term is u_{ij} \otimes v_{jk} \otimes w_{ki} where w_{ki} is the coefficient for C_{ik}.
    # In the naive algorithm, for each (i,j,k), we have:
    # U[i,j] = 1
    # V[j,k] = 1
    # W[k,i] = 1
    # And C_{k,i} gets 1 * 1 * 1 = 1.
    
    # For Strassen, we have linear combinations.
    # For a specific rank-1 term c (one of the 49),
    # U_{ij}^{(c)} is the coefficient of A_{ij} in the first factor.
    # V_{jk}^{(c)} is the coefficient of B_{jk} in the second factor.
    # W_{ki}^{(c)} is the coefficient added to C_{ki} after multiplying the two factors.
    
    # So we need to define 49 such triples.
    
    # Let's generate the Strassen 2x2 coefficients first.
    # Strassen 2x2:
    # Inputs A (2x2), B (2x2).
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now for 4x4.
    # A is 2x2 blocks: A00, A01, A10, A11.
    # B is 2x2 blocks: B00, B01, B10, B11.
    
    # Top level Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each P is a 2x2 matrix.
    # We expand each P using Strassen 2x2.
    # This gives 7 scalar products per P.
    
    # We need to map these to U, V, W.
    # Let's iterate through the 7 top-level products.
    # For each top-level product, we have 7 bottom-level products.
    # Total 49.
    
    # We need to define the coefficients for A, B, C for each of the 49 terms.
    
    # Helper to get Strassen 2x2 coefficients for a specific product Mx.
    # Returns (coeff_A, coeff_B, coeff_C) for the 4 elements of A, B, C?
    # No, it returns the linear combination for A and B, and the contribution to C.
    
    # Let's define the 7 Strassen 2x2 patterns.
    # Pattern 1: (A00+A11)(B00+B11)
    # A coeffs: [1, 0, 0, 1] (for A00, A01, A10, A11)
    # B coeffs: [1, 0, 0, 1]
    # C coeffs: [1, 0, 0, 1] (Contributes to C00, C01, C10, C11? No, just C00 and C11? No, M1 is added to C00 and C11)
    # Wait, M1 is a scalar.
    # C00 += M1
    # C11 += M1
    # So C coeffs: [1, 0, 0, 1]
    
    # Pattern 2: (A10+A11)B00
    # A coeffs: [0, 0, 1, 1]
    # B coeffs: [1, 0, 0, 0]
    # C coeffs: [0, 0, 1, 0] (Contributes to C10)
    
    # Pattern 3: A00(B01-B11)
    # A coeffs: [1, 0, 0, 0]
    # B coeffs: [0, 1, 0, -1]
    # C coeffs: [0, 1, 0, 0] (Contributes to C01)
    
    # Pattern 4: A11(B10-B00)
    # A coeffs: [0, 0, 0, 1]
    # B coeffs: [0, 0, 1, -1]
    # C coeffs: [1, 0, 1, 0] (Contributes to C00 and C10)
    
    # Pattern 5: (A00+A01)B11
    # A coeffs: [1, 1, 0, 0]
    # B coeffs: [0, 0, 0, 1]
    # C coeffs: [0, 0, 0, 1] ? No.
    # C00 -= M5
    # C01 += M5
    # So C coeffs: [-1, 1, 0, 0] ?
    # Let's check C formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M5:
    # C00: -1
    # C01: +1
    # C10: 0
    # C11: 0
    # C coeffs: [-1, 1, 0, 0]
    
    # Pattern 6: (A10-A00)(B00+B01)
    # A coeffs: [-1, 0, 1, 0]
    # B coeffs: [1, 1, 0, 0]
    # C coeffs: [0, 0, 0, 1] (Contributes to C11)
    
    # Pattern 7: (A01-A11)(B10+B11)
    # A coeffs: [0, 1, 0, -1]
    # B coeffs: [0, 0, 1, 1]
    # C coeffs: [1, 0, 0, 0] (Contributes to C00)
    
    # Now we combine these.
    # For each top-level product P (1..7), we use the corresponding pattern.
    # But P is a product of 2x2 blocks.
    # So for P1, the "A" part is (A00+A11) and "B" part is (B00+B11).
    # We apply Strassen 2x2 to these 2x2 blocks.
    # So for P1, we generate 7 scalar products.
    # For each of these 7, we need to determine the coefficients in the global 4x4 A, B, C.
    
    # This is getting complex to hardcode.
    # I will write a function to generate the coefficients.
    
    def get_strassen_2x2_coeffs(pattern_id):
        # Returns (A_coeffs, B_coeffs, C_coeffs) for 2x2 case
        # A_coeffs is list of 4, B_coeffs list of 4, C_coeffs list of 4
        # Order: 00, 01, 10, 11
        if pattern_id == 0: # M1
            return ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
        elif pattern_id == 1: # M2
            return ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])
        elif pattern_id == 2: # M3
            return ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])
        elif pattern_id == 3: # M4
            return ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])
        elif pattern_id == 4: # M5
            return ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])
        elif pattern_id == 5: # M6
            return ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1])
        elif pattern_id == 6: # M7
            return ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        return None

    # Top level patterns for 4x4 (Strassen on 2x2 blocks)
    # P1: (A00+A11)(B00+B11) -> Pattern 0
    # P2: (A10+A11)B00 -> Pattern 1
    # P3: A00(B01-B11) -> Pattern 2
    # P4: A11(B10-B00) -> Pattern 3
    # P5: (A00+A01)B11 -> Pattern 4
    # P6: (A10-A00)(B00+B01) -> Pattern 5
    # P7: (A01-A11)(B10+B11) -> Pattern 6
    
    top_level_patterns = [0, 1, 2, 3, 4, 5, 6]
    
    # We need to map the 2x2 block indices to 4x4 indices.
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    def block_to_indices(block_id):
        # block_id: 0, 1, 2, 3
        # 0: 00, 1: 01, 2: 10, 3: 11
        r = block_id // 2
        c = block_id % 2
        return (r*2, r*2+1, c*2, c*2+1)

    col = 0
    for top_p in range(7):
        # Get the top-level pattern definition
        # This defines which blocks are combined and with what signs
        # We reuse the 2x2 Strassen patterns for the block level
        a_coeffs_block, b_coeffs_block, c_coeffs_block = get_strassen_2x2_coeffs(top_p)
        
        # Now, for each of the 7 bottom-level Strassen products within this top-level product
        for bottom_p in range(7):
            a_coeffs_sub, b_coeffs_sub, c_coeffs_sub = get_strassen_2x2_coeffs(bottom_p)
            
            # We need to construct U, V, W for this rank-1 term.
            # U[i, j] = sum of contributions from A blocks.
            # V[j, k] = sum of contributions from B blocks.
            # W[k, i] = sum of contributions to C blocks.
            
            U_row = [0] * (n*m)
            V_row = [0] * (m*p)
            W_row = [0] * (p*n)
            
            # Iterate over the 4 blocks of A
            for block_a in range(4):
                coeff_a_block = a_coeffs_block[block_a]
                if coeff_a_block == 0: continue
                
                # Iterate over the 4 elements of the block (sub-matrix)
                for sub_idx in range(4):
                    coeff_a_sub = a_coeffs_sub[sub_idx]
                    if coeff_a_sub == 0: continue
                    
                    val = coeff_a_block * coeff_a_sub
                    if val == 0: continue
                    
                    # Map block_a and sub_idx to global i, j
                    r0, r1, c0, c1 = block_to_indices(block_a)
                    sr = sub_idx // 2
                    sc = sub_idx % 2
                    i = r0 + sr
                    j = c0 + sc
                    
                    U_row[i*m + j] = val
            
            # Iterate over the 4 blocks of B
            for block_b in range(4):
                coeff_b_block = b_coeffs_block[block_b]
                if coeff_b_block == 0: continue
                
                for sub_idx in range(4):
                    coeff_b_sub = b_coeffs_sub[sub_idx]
                    if coeff_b_sub == 0: continue
                    
                    val = coeff_b_block * coeff_b_sub
                    if val == 0: continue
                    
                    # Map block_b and sub_idx to global j, k
                    r0, r1, c0, c1 = block_to_indices(block_b)
                    sr = sub_idx // 2
                    sc = sub_idx % 2
                    j = r0 + sr
                    k = c0 + sc
                    
                    V_row[j*p + k] = val
            
            # Iterate over the 4 blocks of C
            for block_c in range(4):
                coeff_c_block = c_coeffs_block[block_c]
                if coeff_c_block == 0: continue
                
                for sub_idx in range(4):
                    coeff_c_sub = c_coeffs_sub[sub_idx]
                    if coeff_c_sub == 0: continue
                    
                    val = coeff_c_block * coeff_c_sub
                    if val == 0: continue
                    
                    # Map block_c and sub_idx to global k, i
                    # Note: C is p x n. Indices k (row), i (col).
                    # Block indices are same as A/B.
                    r0, r1, c0, c1 = block_to_indices(block_c)
                    sr = sub_idx // 2
                    sc = sub_idx % 2
                    k = r0 + sr
                    i = c0 + sc
                    
                    W_row[k*n + i] = val
            
            U[col] = U_row
            V[col] = V_row
            W[col] = W_row
            col += 1
            
    return {"U": U, "V": V, "W": W}
