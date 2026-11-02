import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _get_strassen_444_factors():
    """
    Returns U, V, W matrices for 4x4x4 matrix multiplication using Strassen's algorithm (Rank 49).
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    # Strassen 2x2x2 factors (Rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(g-h)
    # m4 = d(f-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # m1: (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 += m1, C11 += m1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # m2: (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 += m2, C11 -= m2
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # m3: a(g-h)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    # C01 += m3, C11 += m3
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # m4: d(f-e)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    # C00 += m4, C10 += m4
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # m5: (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 -= m5, C01 += m5
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # m6: (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C11 += m6
    W2[3, 5] = 1
    
    # m7: (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C00 += m7, C11 -= m7 (Wait, C11 = m1 - m2 + m3 + m6. Where is m7?
    # Standard Strassen:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    # So m7 is only in C00?
    # Let's check.
    # m7 = (b-d)(g+h).
    # Yes, m7 contributes to C00.
    W2[0, 6] = 1
    
    # Now compose for 4x4
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Block indices: 00:0-3, 01:4-7, 10:8-11, 11:12-15
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The 7 block products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each block product, we use the 2x2 Strassen factors U2, V2, W2.
    # The block product corresponds to a 2x2 matrix multiplication.
    # The result is added to the appropriate 2x2 block of C.
    
    # Helper to add a 2x2 Strassen decomposition to the 4x4 factors
    def add_block_product(u_offset, v_offset, w_offset, sign_u, sign_v, sign_w):
        # This is getting complicated with signs.
        # Better to just build the linear forms explicitly.
        pass

    # Alternative: Just use the known rank-49 decomposition from literature or generate via symbolic execution.
    # Given the complexity, I will use a simpler, verified rank-49 decomposition.
    # Actually, I can just use the schoolbook 64 for the blocks? No, 64*2+16 = 144.
    # I must use 49.
    
    # I will hardcode the 49 terms.
    # This is tedious but reliable.
    # Or, I can use the fact that the rank of 4x4 is 49 and use a known set of factors.
    
    # Let's try to generate them programmatically using the U2, V2, W2 above.
    # We need to map the 2x2 indices to 4x4 indices.
    
    # Block 00 (A00, B00, C00)
    # Block 01 (A01, B01, C01)
    # Block 10 (A10, B10, C10)
    # Block 11 (A11, B11, C11)
    
    # The 7 products are:
    # P1: (A00+A11) * (B00+B11) -> Contributes to C00, C11? No, P1 is a 2x2 matrix.
    # The result of P1 is a 2x2 matrix. It is added to C00?
    # No, the Strassen algorithm for 4x4 computes:
    # C00 = P1 + P4 - P5 + P7 (where Pk are 2x2 matrices)
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for each Pk, we need to compute the 2x2 product.
    # Each Pk is a product of two 2x2 matrices (which are linear combinations of blocks).
    # So for each Pk, we use the U2, V2, W2 factors.
    # The U2 factors are applied to the linear combination of A blocks.
    # The V2 factors are applied to the linear combination of B blocks.
    # The W2 factors are applied to the C block, with the sign from the C formula.
    
    # Let's define the linear combinations for A and B for each Pk.
    # P1: A1 = A00 + A11, B1 = B00 + B11
    # P2: A2 = A10 + A11, B2 = B00
    # P3: A3 = A00, B3 = B10 - B11
    # P4: A4 = A11, B4 = B01 - B00
    # P5: A5 = A00 + A01, B5 = B11
    # P6: A6 = A10 - A00, B6 = B00 + B01
    # P7: A7 = A01 - A11, B7 = B10 + B11
    
    # For each Pk, we have 7 scalar multiplications (from U2, V2, W2).
    # Total 49.
    
    # We will fill U4, V4, W4.
    # Column index in U4, V4, W4: k * 7 + j, where k is Pk index (0-6), j is 2x2 index (0-6).
    
    # Define block offsets
    # A blocks: 00:0-3, 01:4-7, 10:8-11, 11:12-15
    # B blocks: 00:0-3, 01:4-7, 10:8-11, 11:12-15
    # C blocks: 00:0-3, 01:4-7, 10:8-11, 11:12-15
    
    # Linear combinations for A (coefficients for A00, A01, A10, A11)
    A_combos = [
        [1, 0, 0, 1], # P1: A00 + A11
        [0, 0, 1, 1], # P2: A10 + A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00 + A01
        [-1, 0, 1, 0],# P6: A10 - A00
        [0, 1, 0, -1] # P7: A01 - A11
    ]
    
    # Linear combinations for B
    B_combos = [
        [1, 0, 0, 1], # P1: B00 + B11
        [1, 0, 0, 0], # P2: B00
        [0, 0, 1, -1],# P3: B10 - B11
        [0, 1, 0, -1],# P4: B01 - B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00 + B01
        [0, 0, 1, 1]  # P7: B10 + B11
    ]
    
    # Linear combinations for C (which Pk contributes to which C block with what sign)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    C_contrib = [
        {0: 1, 3: 1, 1: -1, 2: 1}, # P1 -> C00(1), C11(1)
        {2: 1, 3: -1},              # P2 -> C10(1), C11(-1)
        {1: 1, 3: 1},               # P3 -> C01(1), C11(1)
        {0: 1, 2: 1},               # P4 -> C00(1), C10(1)
        {0: -1, 1: 1},              # P5 -> C00(-1), C01(1)
        {3: 1},                     # P6 -> C11(1)
        {0: 1}                      # P7 -> C00(1)
    ]
    
    # Now fill the matrices
    for k in range(7):
        # Get the linear combinations for A and B for this Pk
        a_coeff = A_combos[k]
        b_coeff = B_combos[k]
        
        # For each of the 7 2x2 multiplications in this Pk
        for j in range(7):
            col = k * 7 + j
            
            # U factors
            # U2[:, j] gives coefficients for the 2x2 input A_sub.
            # A_sub is a linear combination of A00, A01, A10, A11.
            # So U4[u, col] = sum( a_coeff[b] * U2[u_block, j] )
            # where u_block is the index in the 2x2 sub-matrix.
            
            for b_idx in range(4):
                if a_coeff[b_idx] != 0:
                    # Block offset
                    offset = b_idx * 4
                    # Add a_coeff[b_idx] * U2[:, j] to U4[offset:offset+4, col]
                    U4[offset:offset+4, col] += a_coeff[b_idx] * U2[:, j]
                    
            # V factors
            for b_idx in range(4):
                if b_coeff[b_idx] != 0:
                    offset = b_idx * 4
                    V4[offset:offset+4, col] += b_coeff[b_idx] * V2[:, j]
                    
            # W factors
            # W2[:, j] gives coefficients for the 2x2 output C_sub.
            # C_sub is added to specific C blocks with specific signs.
            for c_block, sign in C_contrib[k].items():
                if sign != 0:
                    offset = c_block * 4
                    W4[offset:offset+4, col] += sign * W2[:, j]
                    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Get 4x4x4 Strassen factors
    U4, V4, W4 = _get_strassen_444_factors()
    
    # Rank for 4x4x4 is 49
    R444 = 49
    
    # We need to construct factors for 4x4x9
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # Split C (4x9) into C1 (4x4), C2 (4x4), C3 (4x1)
    
    # Total Rank = 49 + 49 + 16 = 114
    R_total = 114
    
    U = np.zeros((dim_u, R_total))
    V = np.zeros((dim_v, R_total))
    W = np.zeros((dim_w, R_total))
    
    # Block 1: B1 (cols 0-3), C1 (cols 0-3)
    # U is same as U4
    U[:, 0:49] = U4
    
    # V: V4 is 16x49. V is 36x114.
    # V4 indices: j*4 + k' where k' in 0..3.
    # V indices: j*9 + k where k in 0..3.
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + k_prime
            V[v_idx_main, 0:49] = V4[v_idx_444, :]
            
    # W: W4 is 16x49. W is 36x114.
    # W4 indices: k'*4 + i where k' in 0..3.
    # W indices: k*4 + i where k in 0..3.
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = k_prime * 4 + i
            W[w_idx_main, 0:49] = W4[w_idx_444, :]
            
    # Block 2: B2 (cols 4-7), C2 (cols 4-7)
    # U is same as U4
    U[:, 49:98] = U4
    
    # V: V4 indices: j*4 + k' where k' in 0..3.
    # V indices: j*9 + k where k in 4..7. k = 4 + k'.
    for j in range(4):
        for k_prime in range(4):
            v_idx_444 = j * 4 + k_prime
            v_idx_main = j * 9 + (4 + k_prime)
            V[v_idx_main, 49:98] = V4[v_idx_444, :]
            
    # W: W4 indices: k'*4 + i where k' in 0..3.
    # W indices: k*4 + i where k in 4..7. k = 4 + k'.
    for k_prime in range(4):
        for i in range(4):
            w_idx_444 = k_prime * 4 + i
            w_idx_main = (4 + k_prime) * 4 + i
            W[w_idx_main, 49:98] = W4[w_idx_444, :]
            
    # Block 3: B3 (col 8), C3 (col 8)
    # Schoolbook 4x4x1
    # Rank 16
    for i in range(4):
        for j in range(4):
            r = 98 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
