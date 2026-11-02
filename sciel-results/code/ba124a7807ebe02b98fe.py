"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into A1, A2 (2x2 each).
    # Split B into B11, B12, B21, B22 (2x2 each).
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # We use Strassen's algorithm (rank 7) for each 2x2 multiplication.
    # There are 4 such multiplications.
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen decomposition for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Strassen terms:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to map these to the global indices.
    # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
    # B is 4x4. 
    # B11 is rows 0,1 and cols 0,1.
    # B12 is rows 0,1 and cols 2,3.
    # B21 is rows 2,3 and cols 0,1.
    # B22 is rows 2,3 and cols 2,3.
    
    # C is 2x4.
    # C1 is cols 0,1.
    # C2 is cols 2,3.
    
    # Block 1: A1 * B11 -> contributes to C1
    # Block 2: A2 * B21 -> contributes to C1
    # Block 3: A1 * B12 -> contributes to C2
    # Block 4: A2 * B22 -> contributes to C2
    
    def add_strassen_block(a_offset, b_offset, c_offset, base_col):
        # a_offset: index in U for A (0 for A1, 2 for A2? No, U index is i*m+j)
        # A1: i=0,1; j=0,1. U indices: 0,1,2,3.
        # A2: i=0,1; j=2,3. U indices: 0*4+2=2? No.
        # U index = i * m + j. m=4.
        # A1: i=0,j=0 -> 0. i=0,j=1 -> 1. i=1,j=0 -> 4. i=1,j=1 -> 5.
        # A2: i=0,j=2 -> 2. i=0,j=3 -> 3. i=1,j=2 -> 6. i=1,j=3 -> 7.
        
        # B index = j * p + k. p=4.
        # B11: j=0,1; k=0,1.
        # j=0,k=0 -> 0. j=0,k=1 -> 1.
        # j=1,k=0 -> 4. j=1,k=1 -> 5.
        # B12: j=0,1; k=2,3.
        # j=0,k=2 -> 2. j=0,k=3 -> 3.
        # j=1,k=2 -> 6. j=1,k=3 -> 7.
        # B21: j=2,3; k=0,1.
        # j=2,k=0 -> 8. j=2,k=1 -> 9.
        # j=3,k=0 -> 12. j=3,k=1 -> 13.
        # B22: j=2,3; k=2,3.
        # j=2,k=2 -> 10. j=2,k=3 -> 11.
        # j=3,k=2 -> 14. j=3,k=3 -> 15.
        
        # C index = k * n + i. n=2.
        # C1: k=0,1; i=0,1.
        # k=0,i=0 -> 0. k=0,i=1 -> 2.
        # k=1,i=0 -> 1. k=1,i=1 -> 3.
        # C2: k=2,3; i=0,1.
        # k=2,i=0 -> 4. k=2,i=1 -> 6.
        # k=3,i=0 -> 5. k=3,i=1 -> 7.
        
        # Strassen terms for this block
        # We define the 7 terms
        # Each term has U coeffs, V coeffs, W coeffs
        
        # Term 1: (a00+a11)(b00+b11)
        # A indices in block: 00, 11.
        # B indices in block: 00, 11.
        # C contribution: C00 += 1, C11 += 1.
        
        # We need a helper to map local 2x2 indices to global indices.
        # Local A: 00, 01, 10, 11
        # Local B: 00, 01, 10, 11
        # Local C: 00, 01, 10, 11
        
        # Global A indices for this block
        # a_map: local (i,j) -> global U index
        # Global B indices for this block
        # b_map: local (j,k) -> global V index
        # Global C indices for this block
        # c_map: local (k,i) -> global W index
        
        # We pass the specific indices
        
        # For A1 (cols 0,1):
        # a_00 = U[0], a_01 = U[1], a_10 = U[4], a_11 = U[5]
        # For A2 (cols 2,3):
        # a_00 = U[2], a_01 = U[3], a_10 = U[6], a_11 = U[7]
        
        # For B11 (rows 0,1, cols 0,1):
        # b_00 = V[0], b_01 = V[1], b_10 = V[4], b_11 = V[5]
        # For B12 (rows 0,1, cols 2,3):
        # b_00 = V[2], b_01 = V[3], b_10 = V[6], b_11 = V[7]
        # For B21 (rows 2,3, cols 0,1):
        # b_00 = V[8], b_01 = V[9], b_10 = V[12], b_11 = V[13]
        # For B22 (rows 2,3, cols 2,3):
        # b_00 = V[10], b_01 = V[11], b_10 = V[14], b_11 = V[15]
        
        # For C1 (cols 0,1):
        # c_00 = W[0], c_01 = W[1], c_10 = W[2], c_11 = W[3]
        # For C2 (cols 2,3):
        # c_00 = W[4], c_01 = W[5], c_10 = W[6], c_11 = W[7]
        
        pass

    # We will hardcode the 4 blocks
    
    # Block 1: A1 * B11 -> C1
    # A1: U[0], U[1], U[4], U[5]
    # B11: V[0], V[1], V[4], V[5]
    # C1: W[0], W[1], W[2], W[3]
    
    # Block 2: A2 * B21 -> C1
    # A2: U[2], U[3], U[6], U[7]
    # B21: V[8], V[9], V[12], V[13]
    # C1: W[0], W[1], W[2], W[3]
    
    # Block 3: A1 * B12 -> C2
    # A1: U[0], U[1], U[4], U[5]
    # B12: V[2], V[3], V[6], V[7]
    # C2: W[4], W[5], W[6], W[7]
    
    # Block 4: A2 * B22 -> C2
    # A2: U[2], U[3], U[6], U[7]
    # B22: V[10], V[11], V[14], V[15]
    # C2: W[4], W[5], W[6], W[7]
    
    # Define Strassen terms for a generic 2x2 block
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_a_idx: val}
    # v_coeffs: dict {local_b_idx: val}
    # w_coeffs: dict {local_c_idx: val}
    
    def get_strassen_terms():
        terms = []
        # m1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
        terms.append((
            {0: 1, 3: 1}, 
            {0: 1, 3: 1}, 
            {0: 1, 3: 1}
        ))
        # m2 = (a10+a11)b00 -> c10+=1, c11+=-1
        terms.append((
            {2: 1, 3: 1}, 
            {0: 1}, 
            {2: 1, 3: -1}
        ))
        # m3 = a00(b01-b11) -> c01+=1, c11+=1
        terms.append((
            {0: 1}, 
            {1: 1, 3: -1}, 
            {1: 1, 3: 1}
        ))
        # m4 = a11(b10-b00) -> c00+=1, c10+=1
        terms.append((
            {3: 1}, 
            {2: 1, 0: -1}, 
            {0: 1, 2: 1}
        ))
        # m5 = (a00+a01)b11 -> c00+=-1, c01+=1
        terms.append((
            {0: 1, 1: 1}, 
            {3: 1}, 
            {0: -1, 1: 1}
        ))
        # m6 = (a10-a00)(b00+b01) -> c11+=1
        terms.append((
            {2: 1, 0: -1}, 
            {0: 1, 1: 1}, 
            {3: 1}
        ))
        # m7 = (a01-a11)(b10+b11) -> c00+=1, c11+=1
        terms.append((
            {1: 1, 3: -1}, 
            {2: 1, 3: 1}, 
            {0: 1, 3: 1}
        ))
        return terms

    strassen_terms = get_strassen_terms()
    
    col = 0
    # Block 1: A1, B11, C1
    a_map1 = {0: 0, 1: 1, 2: 4, 3: 5}
    b_map1 = {0: 0, 1: 1, 2: 4, 3: 5}
    c_map1 = {0: 0, 1: 1, 2: 2, 3: 3}
    
    for u_c, v_c, w_c in strassen_terms:
        for li, val in u_c.items():
            U[a_map1[li]][col] += val
        for li, val in v_c.items():
            V[b_map1[li]][col] += val
        for li, val in w_c.items():
            W[c_map1[li]][col] += val
        col += 1
        
    # Block 2: A2, B21, C1
    a_map2 = {0: 2, 1: 3, 2: 6, 3: 7}
    b_map2 = {0: 8, 1: 9, 2: 12, 3: 13}
    c_map2 = {0: 0, 1: 1, 2: 2, 3: 3}
    
    for u_c, v_c, w_c in strassen_terms:
        for li, val in u_c.items():
            U[a_map2[li]][col] += val
        for li, val in v_c.items():
            V[b_map2[li]][col] += val
        for li, val in w_c.items():
            W[c_map2[li]][col] += val
        col += 1
        
    # Block 3: A1, B12, C2
    a_map3 = {0: 0, 1: 1, 2: 4, 3: 5}
    b_map3 = {0: 2, 1: 3, 2: 6, 3: 7}
    c_map3 = {0: 4, 1: 5, 2: 6, 3: 7}
    
    for u_c, v_c, w_c in strassen_terms:
        for li, val in u_c.items():
            U[a_map3[li]][col] += val
        for li, val in v_c.items():
            V[b_map3[li]][col] += val
        for li, val in w_c.items():
            W[c_map3[li]][col] += val
        col += 1
        
    # Block 4: A2, B22, C2
    a_map4 = {0: 2, 1: 3, 2: 6, 3: 7}
    b_map4 = {0: 10, 1: 11, 2: 14, 3: 15}
    c_map4 = {0: 4, 1: 5, 2: 6, 3: 7}
    
    for u_c, v_c, w_c in strassen_terms:
        for li, val in u_c.items():
            U[a_map4[li]][col] += val
        for li, val in v_c.items():
            V[b_map4[li]][col] += val
        for li, val in w_c.items():
            W[c_map4[li]][col] += val
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
