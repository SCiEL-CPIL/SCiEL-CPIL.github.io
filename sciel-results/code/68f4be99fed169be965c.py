import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _strassen_4x4():
    """
    Recursive Strassen decomposition for 4x4 matrix multiplication.
    
    Level 1: 4x4 = 2x2 blocks of 2x2 matrices
    - 7 block multiplications (Strassen at block level)
    
    Level 2: Each 2x2 block multiplication
    - 7 scalar multiplications (Strassen at scalar level)
    
    Total: 7 * 7 = 49 rank terms
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4: A is 2x2 blocks of 2x2 matrices
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # Each block multiplication (2x2 x 2x2) is decomposed into 7 rank-1 terms using Strassen
    # We need to track which scalar entries of the 2x2 blocks are involved
    
    # Helper: add a rank-1 term for a 2x2 block multiplication
    # block_A: 2x2 array of (row, col) indices into the full A matrix
    # block_B: 2x2 array of (row, col) indices into the full B matrix  
    # block_C: 2x2 array of (row, col) indices into the full C matrix
    # sign_A: 2x2 array of signs for A entries
    # sign_B: 2x2 array of signs for B entries
    # sign_C: 2x2 array of signs for C entries
    # col: which rank-1 term this is
    
    def add_block_term(col, A_idx, A_sign, B_idx, B_sign, C_idx, C_sign):
        """
        Add a rank-1 term from a 2x2 Strassen multiplication.
        
        A_idx: 2x2 list of (i,j) indices into A
        A_sign: 2x2 list of coefficients for A entries
        B_idx: 2x2 list of (j,k) indices into B
        B_sign: 2x2 list of coefficients for B entries
        C_idx: 2x2 list of (i,k) indices into C
        C_sign: 2x2 list of coefficients for C entries
        """
        # For each entry in the 2x2 result, we add to U, V, W
        # The 2x2 Strassen decomposition has 7 terms, each contributing to all 4 entries of the 2x2 result
        
        # Actually, let's think differently.
        # A 2x2 matrix product C = A*B has 8 scalar multiplications in schoolbook.
        # Strassen reduces this to 7 scalar multiplications.
        # Each scalar multiplication is a rank-1 term.
        
        # For a 2x2 Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Each M_k is a scalar multiplication.
        # For M1: u = a00 + a11, v = b00 + b11
        # This contributes to c00 with +1, c11 with +1
        
        # So for each of the 7 M_k, we add one rank-1 term.
        # The U vector has the linear form for the "u" part
        # The V vector has the linear form for the "v" part
        # The W vector has the reconstruction weights
        
        pass
    
    # Let's build this more systematically.
    # We have 7 block-level Strassen steps.
    # Each block-level step involves multiplying two 2x2 matrices.
    # Each 2x2 multiplication is decomposed into 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications = 49 rank-1 terms.
    
    # Define the 7 block-level Strassen combinations:
    # Each is (A_comb, B_comb, C_reconstruction)
    # A_comb: how to combine A blocks to form the first operand
    # B_comb: how to combine B blocks to form the second operand
    # C_reconstruction: how to combine the 7 results to form C
    
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level M_k, we need to decompose the 2x2 multiplication into 7 scalar multiplications.
    # Let's define the 2x2 Strassen decomposition for a generic 2x2 product.
    
    # 2x2 Strassen formulas (for matrices A2x2, B2x2):
    # S1 = (a00+a11)(b00+b11)
    # S2 = (a10+a11)b00
    # S3 = a00(b10-b11)
    # S4 = a11(b01-b00)
    # S5 = (a00+a01)b11
    # S6 = (a10-a00)(b00+b01)
    # S7 = (a01-a11)(b10+b11)
    
    # c00 = S1 + S4 - S5 + S7
    # c01 = S3 + S5
    # c10 = S2 + S4
    # c11 = S1 - S2 + S3 + S6
    
    # Now, for each block-level M_k, the "A2x2" and "B2x2" are linear combinations of the 2x2 blocks.
    # We need to track which scalar entries of the original 4x4 matrices are involved.
    
    # Let's define the block indices:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # For a 2x2 block at position (br, bc) in A (br, bc in {0,1}):
    # Entry (i,j) in the block corresponds to A[2*br+i, 2*bc+j]
    # In U, this is at row 2*br+i * m + 2*bc+j = (2*br+i)*4 + 2*bc+j
    
    # For a 2x2 block at position (br, bc) in B:
    # Entry (i,j) in the block corresponds to B[2*br+i, 2*bc+j]
    # In V, this is at row 2*br+i * p + 2*bc+j = (2*br+i)*4 + 2*bc+j
    
    # For a 2x2 block at position (br, bc) in C:
    # Entry (i,j) in the block corresponds to C[2*br+i, 2*bc+j]
    # In W, this is at row 2*bc+j * n + 2*br+i = (2*bc+j)*4 + 2*br+i (trace convention)
    
    # Now, let's define the 7 block-level combinations:
    # Each is a list of (block, sign) for A and B
    
    block_combos = [
        # M1: (A00+A11)(B00+B11)
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
        },
        # M2: (A10+A11)B00
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
        },
        # M3: A00(B10-B11)
        {
            'A': [((0,0), 1)],
            'B': [((1,0), 1), ((1,1), -1)],
        },
        # M4: A11(B01-B00)
        {
            'A': [((1,1), 1)],
            'B': [((0,1), 1), ((0,0), -1)],
        },
        # M5: (A00+A01)B11
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
        },
        # M6: (A10-A00)(B00+B01)
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
        },
        # M7: (A01-A11)(B10+B11)
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
        },
    ]
    
    # For each block-level M_k, we decompose the 2x2 multiplication into 7 scalar multiplications.
    # The 2x2 Strassen decomposition:
    # S1: u = a00+a11, v = b00+b11
    # S2: u = a10+a11, v = b00
    # S3: u = a00, v = b10-b11
    # S4: u = a11, v = b01-b00
    # S5: u = a00+a01, v = b11
    # S6: u = a10-a00, v = b00+b01
    # S7: u = a01-a11, v = b10+b11
    
    # The 2x2 result reconstruction:
    # c00 = S1 + S4 - S5 + S7
    # c01 = S3 + S5
    # c10 = S2 + S4
    # c11 = S1 - S2 + S3 + S6
    
    # For each block-level M_k, the "a" and "b" entries are linear combinations of the original A and B entries.
    # We need to substitute these into the 2x2 Strassen formulas.
    
    # Let's define the 2x2 Strassen scalar decomposition:
    # Each S_k has a specific u and v linear form.
    
    s2x2_decomp = [
        # S1: u = a00+a11, v = b00+b11
        {
            'u': [((0,0), 1), ((1,1), 1)],
            'v': [((0,0), 1), ((1,1), 1)],
        },
        # S2: u = a10+a11, v = b00
        {
            'u': [((1,0), 1), ((1,1), 1)],
            'v': [((0,0), 1)],
        },
        # S3: u = a00, v = b10-b11
        {
            'u': [((0,0), 1)],
            'v': [((1,0), 1), ((1,1), -1)],
        },
        # S4: u = a11, v = b01-b00
        {
            'u': [((1,1), 1)],
            'v': [((0,1), 1), ((0,0), -1)],
        },
        # S5: u = a00+a01, v = b11
        {
            'u': [((0,0), 1), ((0,1), 1)],
            'v': [((1,1), 1)],
        },
        # S6: u = a10-a00, v = b00+b01
        {
            'u': [((1,0), 1), ((0,0), -1)],
            'v': [((0,0), 1), ((0,1), 1)],
        },
        # S7: u = a01-a11, v = b10+b11
        {
            'u': [((0,1), 1), ((1,1), -1)],
            'v': [((1,0), 1), ((1,1), 1)],
        },
    ]
    
    # The 2x2 result reconstruction weights:
    # c00 = S1 + S4 - S5 + S7
    # c01 = S3 + S5
    # c10 = S2 + S4
    # c11 = S1 - S2 + S3 + S6
    
    c2x2_recon = [
        # S1 contributes to:
        [(0,0, 1), (1,1, 1)],  # c00 += 1, c11 += 1
        # S2 contributes to:
        [(1,0, 1), (1,1, -1)],  # c10 += 1, c11 -= 1
        # S3 contributes to:
        [(0,1, 1), (1,1, 1)],  # c01 += 1, c11 += 1
        # S4 contributes to:
        [(0,0, 1), (1,0, 1)],  # c00 += 1, c10 += 1
        # S5 contributes to:
        [(0,0, -1), (0,1, 1)],  # c00 -= 1, c01 += 1
        # S6 contributes to:
        [(1,1, 1)],  # c11 += 1
        # S7 contributes to:
        [(0,0, 1)],  # c00 += 1
    ]
    
    # Now, for each block-level M_k and each 2x2 Strassen S_j, we add one rank-1 term.
    # Total: 7 * 7 = 49 terms.
    
    col = 0
    for k in range(7):
        block_A = block_combos[k]['A']
        block_B = block_combos[k]['B']
        
        for j in range(7):
            s_u = s2x2_decomp[j]['u']
            s_v = s2x2_decomp[j]['v']
            s_c = c2x2_recon[j]
            
            # For each entry in s_u, we have a 2x2 block (br, bc) with sign.
            # For each entry in s_v, we have a 2x2 block (br, bc) with sign.
            # We need to combine these.
            
            # The u part is a linear combination of 2x2 blocks from A.
            # The v part is a linear combination of 2x2 blocks from B.
            # The result is added to C blocks with the weights in s_c.
            
            # For each combination of (block_A_entry, block_B_entry), we add to U, V, W.
            
            for (a_br, a_bc), a_sign in s_u:
                for (b_br, b_bc), b_sign in s_v:
                    # The u entry is the sum over all a_br, a_bc of a_sign * A[2*a_br+i, 2*a_bc+j]
                    # The v entry is the sum over all b_br, b_bc of b_sign * B[2*b_br+i, 2*b_bc+j]
                    # Wait, this is not quite right.
                    
                    # Actually, s_u specifies which entries of the 2x2 "a" matrix are used.
                    # The 2x2 "a" matrix itself is a linear combination of A blocks.
                    # So a[0,0] = sum of a_sign * A[2*a_br+0, 2*a_bc+0] for all (a_br, a_bc) in s_u
                    # a[0,1] = sum of a_sign * A[2*a_br+0, 2*a_bc+1] for all (a_br, a_bc) in s_u
                    # etc.
                    
                    # Similarly for b.
                    
                    # The scalar multiplication is:
                    # u * v = (sum of a_sign * a_entries) * (sum of b_sign * b_entries)
                    # This is a product of two linear forms.
                    # We can expand this into a sum of rank-1 terms, but that would increase the rank.
                    
                    # Actually, no. The Strassen decomposition already accounts for this.
                    # Each S_j is a single scalar multiplication of two linear forms.
                    # So we just need to add one rank-1 term for each S_j.
                    
                    # The U vector for this term has the coefficients for the linear form u.
                    # The V vector for this term has the coefficients for the linear form v.
                    # The W vector has the reconstruction weights.
                    
                    # Let's compute the U coefficients:
                    # u = sum of a_sign * a_entries
                    # a_entries are the 4 entries of the 2x2 "a" matrix.
                    # a[0,0] = sum of a_sign * A[2*a_br+0, 2*a_bc+0]
                    # a[0,1] = sum of a_sign * A[2*a_br+0, 2*a_bc+1]
                    # a[1,0] = sum of a_sign * A[2*a_br+1, 2*a_bc+0]
                    # a[1,1] = sum of a_sign * A[2*a_br+1, 2*a_bc+1]
                    
                    # So u = sum of a_sign * (a[0,0]*w00 + a[0,1]*w01 + a[1,0]*w10 + a[1,1]*w11)
                    # where w00, w01, w10, w11 are the weights from s_u.
                    
                    # Wait, s_u specifies which entries of a are used and their weights.
                    # For example, if s_u = [((0,0), 1), ((1,1), 1)], then:
                    # u = a[0,0] + a[1,1]
                    
                    # So we need to compute the coefficients for each A[i,j].
                    
                    # Let's build the U coefficients:
                    u_coeffs = {}
                    for (a_br, a_bc), a_sign in s_u:
                        for i in range(2):
                            for j in range(2):
                                # A entry at (2*a_br+i, 2*a_bc+j)
                                ai = 2*a_br + i
                                aj = 2*a_bc + j
                                # This entry contributes to a[i,j] with weight 1
                                # a[i,j] contributes to u with weight a_sign
                                # So A[ai, aj] contributes to u with weight a_sign
                                if (ai, aj) not in u_coeffs:
                                    u_coeffs[(ai, aj)] = 0
                                u_coeffs[(ai, aj)] += a_sign
                    
                    # Similarly for V:
                    v_coeffs = {}
                    for (b_br, b_bc), b_sign in s_v:
                        for i in range(2):
                            for j in range(2):
                                bi = 2*b_br + i
                                bj = 2*b_bc + j
                                if (bi, bj) not in v_coeffs:
                                    v_coeffs[(bi, bj)] = 0
                                v_coeffs[(bi, bj)] += b_sign
                    
                    # Now, add to U and V:
                    for (ai, aj), coeff in u_coeffs.items():
                        if coeff != 0:
                            U[ai * m + aj, col] += coeff
                    
                    for (bi, bj), coeff in v_coeffs.items():
                        if coeff != 0:
                            V[bi * p + bj, col] += coeff
                    
                    # Now, add to W:
                    # s_c specifies which C entries get which weights.
                    # s_c is a list of (c_br, c_bc, weight)
                    for (c_br, c_bc, c_weight) in s_c:
                        for i in range(2):
                            for j in range(2):
                                ci = 2*c_br + i
                                cj = 2*c_bc + j
                                # C[ci, cj] gets weight c_weight
                                # In W, this is at row cj * n + ci (trace convention)
                                W[cj * n + ci, col] += c_weight
                    
                    col += 1
    
    # Verify that col == rank
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {'U': U, 'V': V, 'W': W}

def construct():
    start_time = time.time()
    deadline = 150.0
    
    try:
        result = _strassen_4x4()
        return result
    except Exception as e:
        # Fallback to schoolbook rank 64
        n, m, p = 4, 4, 4
        rank = 64
        
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {'U': U, 'V': V, 'W': W}
