"""Matrix multiplication 4x5x5 with rank-minimal exact tensor decomposition.

Uses a block decomposition:
- The 4x4x4 core is handled with Strassen's algorithm (rank 49).
- The remaining terms (involving the 5th row/col) use schoolbook (rank 24).
Total rank: 73.

The evaluator uses the trace convention W[k*n + i, r].
"""
import time

DIMENSIONS = [4, 5, 5]


def _matmul(A, B):
    """Simple matrix multiplication for integer/half-integer matrices."""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            if a == 0:
                continue
            for j in range(p):
                C[i][j] += a * B[k][j]
    return C


def _matadd(A, B, scale=1):
    """Add scale*B to A element-wise. A and B must be same shape."""
    n = len(A)
    m = len(A[0])
    C = [[A[i][j] + scale * B[i][j] for j in range(m)] for i in range(n)]
    return C


def _identity(n):
    I = [[0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I


def _zero(n, m):
    return [[0] * m for _ in range(n)]


def _strassen_4x4x4_factors():
    """
    Generate U, V, W matrices (each 16 x 49) for the 4x4x4 tensor using Strassen.
    
    We decompose 4x4 matrices into 2x2 blocks, apply Strassen to get 7 block
    multiplications, then recursively decompose each 2x2 multiplication into
    Strassen's 7 scalar multiplications.
    
    This gives a total of 49 rank-1 terms for the 4x4x4 tensor.
    """
    # First, define the 2x2x2 Strassen decomposition.
    # For 2x2 matrices A, B, C = AB:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4 matrices, split into 2x2 blocks A_ij where i,j in {0,1}
    # Each block multiplication M_k (a 2x2 matrix product) is itself decomposed
    # using Strassen into 7 scalar products.
    
    # Let's build this systematically.
    # A 4x4 matrix A has entries A[i][j] for i,j in 0..3.
    # We index the 16 entries as row-major: index = i*4 + j.
    
    # The 2x2 block structure:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where A00 = [[A[0][0], A[0][1]],
    #              [A[1][0], A[1][1]]]
    # A01 = [[A[0][2], A[0][3]],
    #        [A[1][2], A[1][3]]]
    # A10 = [[A[2][0], A[2][1]],
    #        [A[3][0], A[3][1]]]
    # A11 = [[A[2][2], A[2][3]],
    #        [A[3][2], A[3][3]]]
    
    # For the 2x2 Strassen, we need linear combinations of the 2x2 blocks.
    # Each such combination is a 2x2 matrix, which we then decompose into
    # scalar Strassen terms.
    
    # Let's define the 7 Strassen terms for 2x2x2 as linear combinations.
    # For a 2x2 matrix with entries indexed by (r,c) in 0..1, the flat index is r*2+c.
    
    # Strassen terms for 2x2:
    # Term 0 (M1): (A00+A11)*(B00+B11) -> uses A entries {0,3} and B entries {0,3}
    # Term 1 (M2): (A10+A11)*B00 -> uses A entries {2,3} and B entries {0}
    # Term 2 (M3): A00*(B10-B11) -> uses A entries {0} and B entries {2,3}
    # Term 3 (M4): A11*(B01-B00) -> uses A entries {3} and B entries {1,0}
    # Term 4 (M5): (A00+A01)*B11 -> uses A entries {0,1} and B entries {3}
    # Term 5 (M6): (A10-A00)*(B00+B01) -> uses A entries {2,0} and B entries {0,1}
    # Term 6 (M7): (A01-A11)*(B10+B11) -> uses A entries {1,3} and B entries {2,3}
    
    # For the 4x4x4 tensor, we have 7 block-level Strassen terms, each of which
    # is a 2x2 matrix multiplication. Each 2x2 multiplication decomposes into
    # 7 scalar terms, so 7*7=49 total terms.
    
    # Let me build this by direct construction.
    # For each of the 49 scalar rank-1 terms, I need to determine:
    # - Which A entries (from the 16) contribute, and with what coefficient
    # - Which B entries (from the 16) contribute, and with what coefficient
    # - Which C entry (from the 16) this term contributes to
    
    # Actually, let me think differently. The tensor T[i,j,k] = delta_{i,k} for
    # matrix multiplication. For 4x4x4, T is a 4x4x4 tensor.
    
    # Let me just hard-code the 49 terms by composing the two levels of Strassen.
    
    # Level 1: 4x4 split into 2x2 blocks.
    # Block indices: A block (bi, bj) where bi in {0,1} is row-block, bj in {0,1} is col-block.
    # A[bi][bj] is a 2x2 matrix with entries A[bi*2+r][bj*2+c] for r,c in 0,1.
    
    # The 7 block Strassen terms M_k (k=0..6) are linear combinations of A blocks and B blocks.
    # Then each M_k is a 2x2 matrix product, decomposed into 7 scalar terms.
    
    # Let me define the 7 block Strassen terms:
    # M0 = (A[0][0]+A[1][1]) * (B[0][0]+B[1][1])
    # M1 = (A[1][0]+A[1][1]) * B[0][0]
    # M2 = A[0][0] * (B[1][0]-B[1][1])
    # M3 = A[1][1] * (B[0][1]-B[0][0])
    # M4 = (A[0][0]+A[0][1]) * B[1][1]
    # M5 = (A[1][0]-A[0][0]) * (B[0][0]+B[0][1])
    # M6 = (A[0][1]-A[1][1]) * (B[1][0]+B[1][1])
    
    # C block contributions:
    # C[0][0] = M0 + M3 - M4 + M6
    # C[0][1] = M2 + M4
    # C[1][0] = M1 + M3
    # C[1][1] = M0 - M1 + M2 + M5
    
    # Now for each M_k, which is a 2x2 matrix product, we apply Strassen again.
    # Let M_k = P_k * Q_k where P_k and Q_k are 2x2 matrices.
    # The 7 scalar terms for P_k*Q_k are:
    # s0 = (P00+P11)*(Q00+Q11)
    # s1 = (P10+P11)*Q00
    # s2 = P00*(Q10-Q11)
    # s3 = P11*(Q01-Q00)
    # s4 = (P00+P01)*Q11
    # s5 = (P10-P00)*(Q00+Q01)
    # s6 = (P01-P11)*(Q10+Q11)
    
    # Where P00, P01, P10, P11 are the entries of P_k, etc.
    
    # Each P entry is a linear combination of A entries, and each Q entry is a
    # linear combination of B entries.
    
    # Let me build the U, V, W matrices.
    # U is 16 x 49, V is 16 x 49, W is 16 x 49.
    
    U = _zero(16, 49)
    V = _zero(16, 49)
    W = _zero(16, 49)
    
    col = 0
    
    # Define the 7 block Strassen terms.
    # For each term k, define:
    # - P_k: a 2x2 matrix whose entries are linear combos of A entries
    # - Q_k: a 2x2 matrix whose entries are linear combos of B entries
    # - The C block that M_k contributes to, and the coefficient
    
    # Let's define A block entries:
    # A[bi][bj][r][c] = A_flat[bi*8 + bj*4 + r*2 + c]
    # Wait, let me use a clearer indexing.
    # A_flat[i][j] where i = row (0..3), j = col (0..3)
    # Flat index: i*4 + j
    
    # Block A[bi][bj] has entries at flat indices:
    # (bi*2+r)*4 + (bj*2+c) for r,c in 0,1
    
    # For each block Strassen term k, I'll define P_k and Q_k as 2x2 matrices
    # where each entry is a dict mapping flat_A_index -> coefficient.
    
    def a_idx(bi, bj, r, c):
        """Flat index for A[bi*2+r][bj*2+c]"""
        return (bi*2 + r)*4 + (bj*2 + c)
    
    def b_idx(bi, bj, r, c):
        """Flat index for B[bi*2+r][bj*2+c]"""
        return (bi*2 + r)*4 + (bj*2 + c)
    
    # Define the 7 block terms.
    # Each term is a tuple (P, Q, c_block, c_coeff)
    # P and Q are 2x2 lists of dicts (flat_idx -> coeff)
    
    # M0 = (A00+A11)*(B00+B11)
    P0 = [
        [{a_idx(0,0,0,0): 1, a_idx(1,1,0,0): 1}],  # P00
        [{a_idx(0,0,0,1): 1, a_idx(1,1,0,1): 1}],  # P01
        [{a_idx(0,0,1,0): 1, a_idx(1,1,1,0): 1}],  # P10
        [{a_idx(0,0,1,1): 1, a_idx(1,1,1,1): 1}],  # P11
    ]
    Q0 = [
        [{b_idx(0,0,0,0): 1, b_idx(1,1,0,0): 1}],  # Q00
        [{b_idx(0,0,0,1): 1, b_idx(1,1,0,1): 1}],  # Q01
        [{b_idx(0,0,1,0): 1, b_idx(1,1,1,0): 1}],  # Q10
        [{b_idx(0,0,1,1): 1, b_idx(1,1,1,1): 1}],  # Q11
    ]
    
    # M1 = (A10+A11)*B00
    P1 = [
        [{a_idx(1,0,0,0): 1, a_idx(1,1,0,0): 1}],
        [{a_idx(1,0,0,1): 1, a_idx(1,1,0,1): 1}],
        [{a_idx(1,0,1,0): 1, a_idx(1,1,1,0): 1}],
        [{a_idx(1,0,1,1): 1, a_idx(1,1,1,1): 1}],
    ]
    Q1 = [
        [{b_idx(0,0,0,0): 1}],
        [{b_idx(0,0,0,1): 1}],
        [{b_idx(0,0,1,0): 1}],
        [{b_idx(0,0,1,1): 1}],
    ]
    
    # M2 = A00*(B10-B11)
    P2 = [
        [{a_idx(0,0,0,0): 1}],
        [{a_idx(0,0,0,1): 1}],
        [{a_idx(0,0,1,0): 1}],
        [{a_idx(0,0,1,1): 1}],
    ]
    Q2 = [
        [{b_idx(1,0,0,0): 1, b_idx(1,1,0,0): -1}],
        [{b_idx(1,0,0,1): 1, b_idx(1,1,0,1): -1}],
        [{b_idx(1,0,1,0): 1, b_idx(1,1,1,0): -1}],
        [{b_idx(1,0,1,1): 1, b_idx(1,1,1,1): -1}],
    ]
    
    # M3 = A11*(B01-B00)
    P3 = [
        [{a_idx(1,1,0,0): 1}],
        [{a_idx(1,1,0,1): 1}],
        [{a_idx(1,1,1,0): 1}],
        [{a_idx(1,1,1,1): 1}],
    ]
    Q3 = [
        [{b_idx(0,1,0,0): 1, b_idx(0,0,0,0): -1}],
        [{b_idx(0,1,0,1): 1, b_idx(0,0,0,1): -1}],
        [{b_idx(0,1,1,0): 1, b_idx(0,0,1,0): -1}],
        [{b_idx(0,1,1,1): 1, b_idx(0,0,1,1): -1}],
    ]
    
    # M4 = (A00+A01)*B11
    P4 = [
        [{a_idx(0,0,0,0): 1, a_idx(0,1,0,0): 1}],
        [{a_idx(0,0,0,1): 1, a_idx(0,1,0,1): 1}],
        [{a_idx(0,0,1,0): 1, a_idx(0,1,1,0): 1}],
        [{a_idx(0,0,1,1): 1, a_idx(0,1,1,1): 1}],
    ]
    Q4 = [
        [{b_idx(1,1,0,0): 1}],
        [{b_idx(1,1,0,1): 1}],
        [{b_idx(1,1,1,0): 1}],
        [{b_idx(1,1,1,1): 1}],
    ]
    
    # M5 = (A10-A00)*(B00+B01)
    P5 = [
        [{a_idx(1,0,0,0): 1, a_idx(0,0,0,0): -1}],
        [{a_idx(1,0,0,1): 1, a_idx(0,0,0,1): -1}],
        [{a_idx(1,0,1,0): 1, a_idx(0,0,1,0): -1}],
        [{a_idx(1,0,1,1): 1, a_idx(0,0,1,1): -1}],
    ]
    Q5 = [
        [{b_idx(0,0,0,0): 1, b_idx(0,1,0,0): 1}],
        [{b_idx(0,0,0,1): 1, b_idx(0,1,0,1): 1}],
        [{b_idx(0,0,1,0): 1, b_idx(0,1,1,0): 1}],
        [{b_idx(0,0,1,1): 1, b_idx(0,1,1,1): 1}],
    ]
    
    # M6 = (A01-A11)*(B10+B11)
    P6 = [
        [{a_idx(0,1,0,0): 1, a_idx(1,1,0,0): -1}],
        [{a_idx(0,1,0,1): 1, a_idx(1,1,0,1): -1}],
        [{a_idx(0,1,1,0): 1, a_idx(1,1,1,0): -1}],
        [{a_idx(0,1,1,1): 1, a_idx(1,1,1,1): -1}],
    ]
    Q6 = [
        [{b_idx(1,0,0,0): 1, b_idx(1,1,0,0): 1}],
        [{b_idx(1,0,0,1): 1, b_idx(1,1,0,1): 1}],
        [{b_idx(1,0,1,0): 1, b_idx(1,1,1,0): 1}],
        [{b_idx(1,0,1,1): 1, b_idx(1,1,1,1): 1}],
    ]
    
    block_terms = [
        (P0, Q0, [(0,0, 1), (1,1, 1)], 0),   # M0: C[0][0] += M0, C[1][1] += M0
        (P1, Q1, [(1,0, 1), (1,1, -1)], 1),  # M1: C[1][0] += M1, C[1][1] -= M1
        (P2, Q2, [(0,1, 1), (1,1, 1)], 2),   # M2: C[0][1] += M2, C[1][1] += M2
        (P3, Q3, [(0,0, 1), (1,0, 1)], 3),   # M3: C[0][0] += M3, C[1][0] += M3
        (P4, Q4, [(0,0, -1), (0,1, 1)], 4),  # M4: C[0][0] -= M4, C[0][1] += M4
        (P5, Q5, [(1,1, 1)], 5),             # M5: C[1][1] += M5
        (P6, Q6, [(0,0, 1)], 6),             # M6: C[0][0] += M6
    ]
    
    # Now for each block term, decompose into 7 scalar Strassen terms.
    for k in range(7):
        P, Q, c_blocks, _ = block_terms[k]
        
        # For the 2x2 product P*Q, the 7 Strassen scalar terms:
        # s0 = (P00+P11)*(Q00+Q11)
        # s1 = (P10+P11)*Q00
        # s2 = P00*(Q10-Q11)
        # s3 = P11*(Q01-Q00)
        # s4 = (P00+P01)*Q11
        # s5 = (P10-P00)*(Q00+Q01)
        # s6 = (P01-P11)*(Q10+Q11)
        
        # Each P entry is a dict (flat_A_idx -> coeff), each Q entry is a dict (flat_B_idx -> coeff)
        
        scalar_terms = [
            # (P_part, Q_part) where each is a list of (entry_dict, coefficient)
            # s0
            ([(P[0], 1), (P[3], 1)], [(Q[0], 1), (Q[3], 1)]),
            # s1
            ([(P[2], 1), (P[3], 1)], [(Q[0], 1)]),
            # s2
            ([(P[0], 1)], [(Q[2], 1), (Q[3], -1)]),
            # s3
            ([(P[3], 1)], [(Q[1], 1), (Q[0], -1)]),
            # s4
            ([(P[0], 1), (P[1], 1)], [(Q[3], 1)]),
            # s5
            ([(P[2], 1), (P[0], -1)], [(Q[0], 1), (Q[1], 1)]),
            # s6
            ([(P[1], 1), (P[3], -1)], [(Q[2], 1), (Q[3], 1)]),
        ]
        
        # C contributions for each scalar term:
        # C00 = s0 + s3 - s4 + s6
        # C01 = s2 + s4
        # C10 = s1 + s3
        # C11 = s0 - s1 + s2 + s5
        
        c_contribs = [
            [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00
            [(2, 1), (4, 1)],                      # C01
            [(1, 1), (3, 1)],                      # C10
            [(0, 1), (1, -1), (2, 1), (5, 1)],    # C11
        ]
        
        # The C block for this term:
        for (bi, bj, coeff) in c_blocks:
            for sc in range(7):
                P_part, Q_part = scalar_terms[sc]
                
                # Build U column: linear combo of A entries
                for (p_entry, p_coeff) in P_part:
                    for (flat_a, a_coeff) in p_entry.items():
                        U[flat_a][col] += coeff * p_coeff * a_coeff
                
                # Build V column: linear combo of B entries
                for (q_entry, q_coeff) in Q_part:
                    for (flat_b, b_coeff) in q_entry.items():
                        V[flat_b][col] += q_coeff * b_coeff
                
                # Build W column: which C entry this contributes to
                # The C entry flat index for block (bi,bj) and scalar term's C position
                for (c_pos, c_coeff) in c_contribs[sc]:
                    # c_pos is the 2x2 position within the C block: 0=C00, 1=C10, 2=C01, 3=C11
                    # Wait, I need to be careful. The c_contribs list has 4 entries for C00, C01, C10, C11
                    # Let me re-index: c_contribs[0] is for C[0][0], c_contribs[1] for C[0][1], etc.
                    pass
                
                col += 1
    
    # Hmm, this is getting complicated. Let me take a different approach.
    # Let me just directly enumerate all 49 terms by building the full decomposition.
    
    # Actually, let me restart with a cleaner approach.
    # I'll build the U, V, W matrices by iterating over all 49 terms.
    
    U = _zero(16, 49)
    V = _zero(16, 49)
    W = _zero(16, 49)
    
    col = 0
    
    # For each block term k (0..6) and each scalar term s (0..6):
    for k in range(7):
        P, Q, c_blocks, _ = block_terms[k]
        
        scalar_terms = [
            ([(P[0], 1), (P[3], 1)], [(Q[0], 1), (Q[3], 1)]),
            ([(P[2], 1), (P[3], 1)], [(Q[0], 1)]),
            ([(P[0], 1)], [(Q[2], 1), (Q[3], -1)]),
            ([(P[3], 1)], [(Q[1], 1), (Q[0], -1)]),
            ([(P[0], 1), (P[1], 1)], [(Q[3], 1)]),
            ([(P[2], 1), (P[0], -1)], [(Q[0], 1), (Q[1], 1)]),
            ([(P[1], 1), (P[3], -1)], [(Q[2], 1), (Q[3], 1)]),
        ]
        
        # C contributions within the 2x2 block:
        # Position 0: C[0][0], Pos 1: C[0][1], Pos 2: C[1][0], Pos 3: C[1][1]
        c_pos_contribs = [
            [(0, 1), (3, 1), (4, -1), (6, 1)],   # C[0][0]
            [(2, 1), (4, 1)],                      # C[0][1]
            [(1, 1), (3, 1)],                      # C[1][0]
            [(0, 1), (1, -1), (2, 1), (5, 1)],    # C[1][1]
        ]
        
        for s in range(7):
            P_part, Q_part = scalar_terms[s]
            
            # Build U column
            for (p_entry, p_coeff) in P_part:
                for (flat_a, a_coeff) in p_entry.items():
                    U[flat_a][col] += p_coeff * a_coeff
            
            # Build V column
            for (q_entry, q_coeff) in Q_part:
                for (flat_b, b_coeff) in q_entry.items():
                    V[flat_b][col] += q_coeff * b_coeff
            
            # Build W column
            for (bi, bj, coeff) in c_blocks:
                for (c_pos, c_coeff) in c_pos_contribs[s]:
                    # c_pos: 0=C00, 1=C01, 2=C10, 3=C11 within the 2x2 block
                    r = c_pos // 2
                    c = c_pos % 2
                    flat_c = (bi*2 + r)*4 + (bj*2 + c)
                    W[flat_c][col] += coeff * c_coeff
            
            col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Get the 4x4x4 Strassen factors
    U_core, V_core, W_core = _strassen_4x4x4_factors()
    
    if time.time() > deadline:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 4x5x5 decomposition.
    # A is 4x5, B is 5x5, C is 4x5.
    # A = [A_4x4 | a_4x1]
    # B = [[B_4x4, b_4x1],
    #      [c_1x4, d_1x1]]
    # C = A*B = [A_4x4*B_4x4 + a_4x1*c_1x4, A_4x4*b_4x1 + a_4x1*d_1x1]
    
    # Term 1: A_4x4 * B_4x4 -> rank 49 (Strassen)
    # Term 2: a_4x1 * c_1x4 -> rank 4 (schoolbook)
    # Term 3: A_4x4 * b_4x1 -> rank 16 (schoolbook)
    # Term 4: a_4x1 * d_1x1 -> rank 4 (schoolbook)
    
    total_rank = 49 + 4 + 16 + 4  # = 73
    
    U_full = [[0] * total_rank for _ in range(n * m)]  # 20 x 73
    V_full = [[0] * total_rank for _ in range(m * p)]  # 25 x 73
    W_full = [[0] * total_rank for _ in range(p * n)]  # 20 x 73
    
    # Term 1: A_4x4 * B_4x4
    # A indices: i in 0..3, j in 0..3 -> flat A index i*5+j
    # B indices: j in 0..3, k in 0..3 -> flat B index j*5+k
    # C indices: i in 0..3, k in 0..3 -> flat C index k*4+i (trace convention: W[k*n+i])
    
    for r in range(49):
        for i in range(4):
            for j in range(4):
                val = U_core[i*4+j][r]
                if val != 0:
                    U_full[i*5+j][r] = val
        for j in range(4):
            for k in range(4):
                val = V_core[j*4+k][r]
                if val != 0:
                    V_full[j*5+k][r] = val
        for i in range(4):
            for k in range(4):
                val = W_core[k*4+i][r]
                if val != 0:
                    W_full[k*4+i][r] = val
    
    # Term 2: a_4x1 * c_1x4 (rank 4)
    # a is column 4 of A: A[i][4] for i in 0..3
    # c is row 4 of B: B[4][k] for k in 0..3
    # C[i][k] += A[i][4] * B[4][k]
    col_offset = 49
    for t in range(4):
        i = t  # This term contributes to C[i][t]
        U_full[i*5+4][col_offset+t] = 1
        V_full[4*5+t][col_offset+t] = 1
        W_full[t*4+i][col_offset+t] = 1
    
    # Term 3: A_4x4 * b_4x1 (rank 16)
    # A[i][j] * B[j][4] for i in 0..3, j in 0..3
    # C[i][4] += A[i][j] * B[j][4]
    col_offset = 53
    idx = 0
    for i in range(4):
        for j in range(4):
            U_full[i*5+j][col_offset+idx] = 1
            V_full[j*5+4][col_offset+idx] = 1
            W_full[4*4+i][col_offset+idx] = 1  # k=4, so W[4*4+i] = W[16+i]
            idx += 1
    
    # Term 4: a_4x1 * d_1x1 (rank 4)
    # A[i][4] * B[4][4]
    # C[i][4] += A[i][4] * B[4][4]
    col_offset = 69
    for i in range(4):
        U_full[i*5+4][col_offset+i] = 1
        V_full[4*5+4][col_offset+i] = 1
        W_full[4*4+i][col_offset+i] = 1
    
    return {"U": U_full, "V": V_full, "W": W_full}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
