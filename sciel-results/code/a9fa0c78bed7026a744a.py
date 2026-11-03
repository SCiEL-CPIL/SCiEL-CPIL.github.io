"""Optimized matrix multiplication decomposition for 4x4 * 4x9."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # ---------- 2x2x2 rank-7 decomposition (Strassen) ----------
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For 2x2x2: i,j,k in {0,1}
    # U, V, W are each 2x2x2 (flattened to (4,7))
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))

    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Term 1: M1
    U_222[0, 0] = 1; U_222[3, 0] = 1  # A00, A11
    V_222[0, 0] = 1; V_222[2, 0] = 1  # B00, B11
    W_222[0, 0] = 1; W_222[3, 0] = -1  # C00 += M1, C11 -= M1 ... wait
    # Let me be more careful. W[k,i,r] contributes to C[i,k] (since C is indexed by row i, col k)
    # Actually C[i,k] = sum_j A[i,j]*B[j,k]
    # In our convention: C[i,k] gets contribution W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # For 2x2x2: C[i,k] = sum_r W[k*2+i, r] * U[i*2+j, r] * V[j*2+k, r]
    #
    # M1 should contribute to C00 and C11:
    # C00 = M1 + M4 - M5 + M7  => W[0*2+0, 0] = 1 (for C[0,0])
    # C11 = M1 - M2 + M3 + M6  => W[1*2+1, 0] = 1 (for C[1,1])
    W_222[0, 0] = 1
    W_222[3, 0] = 1  # k=1, i=1 -> index 1*2+1 = 3

    # Term 2: M2 = (A10+A11)B00
    U_222[2, 1] = 1; U_222[3, 1] = 1  # A10, A11
    V_222[0, 1] = 1                    # B00
    # C10 = M2 + M4 => W[0*2+1, 1] = 1 (k=0, i=1)
    W_222[1, 1] = 1
    # C11 = ... - M2 => W[1*2+1, 1] = -1
    W_222[3, 1] = -1

    # Term 3: M3 = A00(B01-B11)
    U_222[0, 2] = 1                    # A00
    V_222[1, 2] = 1; V_222[3, 2] = -1  # B01 - B11
    # C01 = M3 + M5 => W[1*2+0, 2] = 1 (k=1, i=0)
    W_222[2, 2] = 1
    # C11 = ... + M3 => W[1*2+1, 2] = 1 (k=1, i=1)
    W_222[3, 2] = 1

    # Term 4: M4 = A11(B10-B00)
    U_222[3, 3] = 1                    # A11
    V_222[2, 3] = 1; V_222[0, 3] = -1  # B10 - B00
    # C00 = ... + M4 => W[0*2+0, 3] = 1
    W_222[0, 3] = 1
    # C10 = ... + M4 => W[0*2+1, 3] = 1 (k=0, i=1)
    W_222[1, 3] = 1

    # Term 5: M5 = (A00+A10)B11
    U_222[0, 4] = 1; U_222[2, 4] = 1  # A00, A10
    V_222[3, 4] = 1                    # B11
    # C00 = ... - M5 => W[0*2+0, 4] = -1
    W_222[0, 4] = -1
    # C01 = ... + M5 => W[1*2+0, 4] = 1 (k=1, i=0)
    W_222[2, 4] = 1

    # Term 6: M6 = (A11-A10)(B00+B10)
    U_222[3, 5] = 1; U_222[2, 5] = -1  # A11 - A10
    V_222[0, 5] = 1; V_222[2, 5] = 1   # B00 + B10
    # C11 = ... + M6 => W[1*2+1, 5] = 1
    W_222[3, 5] = 1

    # Term 7: M7 = (A01-A11)(B10+B11)
    U_222[1, 6] = 1; U_222[3, 6] = -1  # A01 - A11
    V_222[2, 6] = 1; V_222[3, 6] = 1   # B10 + B11
    # C00 = ... + M7 => W[0*2+0, 6] = 1
    W_222[0, 6] = 1

    # Verify 2x2x2 decomposition
    def verify_222():
        U3 = U_222.reshape(2, 2, 7)
        V3 = V_222.reshape(2, 2, 7)
        W3 = W_222.reshape(2, 2, 7)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        expected = np.ones((2, 2, 2))
        return np.max(np.abs(T - expected))

    # ---------- Build 4x4x9 decomposition ----------
    # Strategy: Split A (4x4) into 2x2 blocks, B (4x9) into 2x9 blocks.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B0], [B1]] where B0, B1 are 2x9
    # C = [[C0], [C1]] where C0, C1 are 2x9
    #
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    #
    # We need to compute: A2x2 * B2x9 for various combinations.
    # 
    # Alternative approach: Use the known rank-49 decomposition for 4x4x4,
    # and handle the remaining columns (4-8) with schoolbook.
    #
    # 4x4x4 has rank 49 (Strassen squared).
    # 4x4x1 has rank 16 (schoolbook).
    # Total for 4x4x9: split B into B[:, :4] (4x4), B[:, 4:8] (4x4), B[:, 8:9] (4x1)
    # Rank = 49 + 49 + 16 = 114
    #
    # But we can do better by using a 4x4x2 decomposition.
    # 4x4x2: Split B into two 4x1 columns. Each 4x4x1 has rank 16.
    # But we can share structure: rank(4x4x2) <= rank(4x4x4) = 49? No, that's not right.
    # Actually rank(4x4x2) <= 2*16 = 32 (schoolbook for two columns).
    # Can we do better? Possibly, but let's use 49+49+16=114 first.
    
    # Let me construct the 4x4x4 rank-49 decomposition.
    # For 4x4x4: A (4x4) * B (4x4) = C (4x4)
    # Use block Strassen: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # Each 2x2 block multiplication uses rank-7 decomposition.
    # Total: 7 * 7 = 49 terms.
    
    def build_4x4x4_decomposition(k_offset=0):
        """Build U, V, W for 4x4x4 with B's columns offset by k_offset."""
        U = np.zeros((16, 49))
        V = np.zeros((36, 49))
        W = np.zeros((16, 49))  # 4x4 output: 4*4=16 entries
        
        # For each of the 7 block-Strassen terms, we have 7 inner terms.
        # Block Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11-A10)(B00+B10)
        # M7 = (A01-A11)(B10+B11)
        #
        # Each M_t is a 2x2 * 2x2 product, decomposed with rank 7.
        # The 2x2 blocks are indexed by:
        # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
        #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
        # B blocks: B00 (rows 0-1, cols k_off..k_off+1), B01 (rows 0-1, cols k_off+2..k_off+3)
        #           B10 (rows 2-3, cols k_off..k_off+1), B11 (rows 2-3, cols k_off+2..k_off+3)
        
        # For block term t (0-6), and inner term s (0-6):
        # U entries: A-block elements with coefficients from U_222[s, ...]
        # V entries: B-block elements with coefficients from V_222[s, ...]
        # W entries: C-block elements with coefficients from W_222[s, ...]
        
        # Block coefficient matrices for each Strassen term:
        # For each term, we need to know which A-blocks and B-blocks are involved,
        # and with what coefficients.
        
        # Term definitions (A-block coeffs, B-block coeffs):
        block_defs = [
            # (a00, a01, a10, a11, b00, b01, b10, b11)
            ( 1,  0,  0,  1,  1,  0,  0,  1),  # M1
            ( 0,  0,  1,  1,  1,  0,  0,  0),  # M2
            ( 1,  0,  0,  0,  0,  1,  0, -1),  # M3
            ( 0,  0,  0,  1, -1,  0,  1,  0),  # M4
            ( 1,  0,  1,  0,  0,  0,  0,  1),  # M5
            ( 0,  0, -1,  1,  1,  0,  1,  0),  # M6
            ( 0,  1,  0, -1,  0,  0,  1,  1),  # M7
        ]
        
        # C-block output coefficients for each term:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        c_block_defs = [
            # (c00, c01, c10, c11)
            ( 1,  0,  0,  1),  # M1
            ( 0,  0,  1, -1),  # M2
            ( 0,  1,  0,  1),  # M3
            ( 1,  0,  1,  0),  # M4
            (-1,  1,  0,  0),  # M5
            ( 0,  0,  0,  1),  # M6
            ( 1,  0,  0,  0),  # M7
        ]
        
        for t in range(7):
            for s in range(7):
                col = t * 7 + s
                # U: A-blocks
                for (a_blk, coeff) in [(0, block_defs[t][0]), (1, block_defs[t][1]),
                                       (2, block_defs[t][2]), (3, block_defs[t][3])]:
                    if coeff == 0:
                        continue
                    # A-block a_blk: row offset, col offset
                    row_off = (a_blk // 2) * 2
                    col_off = (a_blk % 2) * 2
                    # Inner 2x2 positions within block
                    for ii in range(2):
                        for jj in range(2):
                            U[row_off + ii, col_off + jj, col] if False else None
                            # U index: i*m+j = i*4+j
                            U[(row_off + ii) * 4 + (col_off + jj), col] += coeff * U_222[ii * 2 + jj, s]
                
                # V: B-blocks
                for (b_blk, coeff) in [(0, block_defs[t][4]), (1, block_defs[t][5]),
                                       (2, block_defs[t][6]), (3, block_defs[t][7])]:
                    if coeff == 0:
                        continue
                    row_off = (b_blk // 2) * 2
                    col_off = (b_blk % 2) * 2
                    for jj in range(2):
                        for kk in range(2):
                            V[row_off + jj, k_offset + col_off + kk, col] += coeff * V_222[jj * 2 + kk, s]
                
                # W: C-blocks (output is 4x4)
                for (c_blk, coeff) in [(0, c_block_defs[t][0]), (1, c_block_defs[t][1]),
                                       (2, c_block_defs[t][2]), (3, c_block_defs[t][3])]:
                    if coeff == 0:
                        continue
                    row_off = (c_blk // 2) * 2  # i offset
                    col_off = (c_blk % 2) * 2   # k offset
                    for kk in range(2):
                        for ii in range(2):
                            W[k_offset + col_off + kk, row_off + ii, col] += coeff * W_222[kk * 2 + ii, s]
        
        return U, V, W

    # Hmm, the W indexing is getting complex. Let me redo this more carefully.
    # W[k*n+i, r] where k is column of C, i is row of C.
    # For 4x4x4: C is 4x4, so W is (16, 49) with index k*4+i.
    # For the full problem: W is (36, rank) with index k*4+i where k is 0..8.
    
    # Let me rebuild more carefully.
    
    def build_full_decomposition():
        """Build the full 4x4x9 decomposition."""
        # Split B into three parts: B[:, 0:4], B[:, 4:8], B[:, 8:9]
        # Rank: 49 + 49 + 16 = 114
        
        total_rank = 49 + 49 + 16
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        # Part 1: 4x4x4 for columns 0-3
        U1, V1, W1 = build_4x4x4_decomposition(0)
        U[:, 0:49] = U1
        V[:, 0:49] = V1
        # W1 is (16, 49) with index k*4+i for k=0..3
        # Need to map to full W index k*4+i for k=0..3 (same)
        W[0:16, 0:49] = W1
        
        # Part 2: 4x4x4 for columns 4-7
        U2, V2, W2 = build_4x4x4_decomposition(4)
        U[:, 49:98] = U2
        V[:, 49:98] = V2
        # W2 index k*4+i for k=4..7 -> full index (k)*4+i where k is 4..7
        # But W2 was built with k_offset=4, so W2 index is (k)*4+i for k=4..7
        # Actually in build_4x4x4_decomposition, W is indexed by (k_offset + kk)*4 + (row_off + ii)
        # So W2 already has the correct indices for k=4..7
        W[0:16, 49:98] = W2  # Wait, W2 has indices up to 7*4+3=31, so we need W[0:32, 49:98]
        # Let me fix: W2 is (16, 49) but indices go up to 31. Actually W2 should be (32, 49) or we handle it differently.
        
        # This is getting messy. Let me just build it directly.
        
        return None
    
    # Direct construction approach:
    
    def build_4x4x9_decomposition():
        """
        Build 4x4x9 decomposition with rank 114.
        Split B into B1 (4x4, cols 0-3), B2 (4x4, cols 4-7), B3 (4x1, col 8).
        Each 4x4x4 part uses rank-49 (Strassen squared).
        The 4x4x1 part uses rank-16 (schoolbook).
        Total rank = 114.
        """
        total_rank = 49 + 49 + 16
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        # Block Strassen definitions for 2x2x2
        # A-blocks: 00, 01, 10, 11 (row_block, col_block)
        # B-blocks: 00, 01, 10, 11
        # C-blocks: 00, 01, 10, 11
        
        # Strassen terms for 2x2 block multiplication:
        # term: (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: {blk: coeff}, B_coeffs: {blk: coeff}, C_coeffs: {blk: coeff}
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11), C00 += M1, C11 += M1
            ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
            # M2 = (A10+A11)B00, C10 += M2, C11 -= M2
            ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
            # M3 = A00(B01-B11), C01 += M3, C11 += M3
            ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
            # M4 = A11(B10-B00), C00 += M4, C10 += M4
            ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
            # M5 = (A00+A10)B11, C00 -= M5, C01 += M5
            ({'00': 1, '10': 1}, {'11': 1}, {'00': -1, '01': 1}),
            # M6 = (A11-A10)(B00+B10), C11 += M6
            ({'11': 1, '10': -1}, {'00': 1, '10': 1}, {'11': 1}),
            # M7 = (A01-A11)(B10+B11), C00 += M7
            ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
        ]
        
        # For each Strassen term and each inner 2x2x2 term, add contributions.
        
        def add_4x4x4_part(k_start, rank_offset):
            """Add a 4x4x4 part where B columns start at k_start."""
            for t in range(7):
                a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
                for s in range(7):
                    col = rank_offset + t * 7 + s
                    
                    # U contributions from A-blocks
                    for blk_name, coeff in a_coeffs.items():
                        if coeff == 0:
                            continue
                        rb = int(blk_name[0])  # row block
                        cb = int(blk_name[1])  # col block
                        row_off = rb * 2
                        col_off = cb * 2
                        for ii in range(2):
                            for jj in range(2):
                                u_idx = (row_off + ii) * m + (col_off + jj)
                                U[u_idx, col] += coeff * U_222[ii * 2 + jj, s]
                    
                    # V contributions from B-blocks
                    for blk_name, coeff in b_coeffs.items():
                        if coeff == 0:
                            continue
                        rb = int(blk_name[0])
                        cb = int(blk_name[1])
                        row_off = rb * 2
                        col_off = k_start + cb * 2
                        for jj in range(2):
                            for kk in range(2):
                                v_idx = (row_off + jj) * p + (col_off + kk)
                                V[v_idx, col] += coeff * V_222[jj * 2 + kk, s]
                    
                    # W contributions from C-blocks
                    for blk_name, coeff in c_coeffs.items():
                        if coeff == 0:
                            continue
                        rb = int(blk_name[0])  # row block of C (= i block)
                        cb = int(blk_name[1])  # col block of C (= k block)
                        row_off = rb * 2
                        col_off = k_start + cb * 2
                        for kk in range(2):
                            for ii in range(2):
                                w_idx = (col_off + kk) * n + (row_off + ii)
                                W[w_idx, col] += coeff * W_222[kk * 2 + ii, s]
        
        # Part 1: 4x4x4 for B columns 0-3
        add_4x4x4_part(0, 0)
        
        # Part 2: 4x4x4 for B columns 4-7
        add_4x4x4_part(4, 49)
        
        # Part 3: 4x4x1 for B column 8 (schoolbook)
        for i in range(n):
            for j in range(m):
                col = 98 + i * m + j
                U[i * m + j, col] = 1
                V[j * p + 8, col] = 1
                W[8 * n + i, col] = 1
        
        return U, V, W
    
    U, V, W = build_4x4x9_decomposition()
    
    # Verify
    def verify(U, V, W):
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))
    
    err = verify(U, V, W)
    if err > 1e-8:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
