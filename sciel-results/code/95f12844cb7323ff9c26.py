import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Rank 77 decomposition for 4x4x6
    # Part 1: 4x4x4 (Rank 49)
    # Part 2: 4x4x2 (Rank 28)
    
    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # U, V, W matrices for 2x2x2 rank 7
    # Indices:
    # A: 0,1,2,3 (00,01,10,11)
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    # Standard Strassen
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the vectors for U, V, W for each of the 7 terms.
    # U is 4x1, V is 4x1, W is 4x4 (flattened 16x1? No, W is output factor)
    # In the tensor decomposition T = sum u_r v_r w_r
    # u_r is in R^4 (A indices)
    # v_r is in R^4 (B indices)
    # w_r is in R^4 (C indices) -- Wait, C is 2x2, so 4 entries.
    
    # The mapping to global indices:
    # A is 4x4. Flat index i*4+j.
    # B is 4x6. Flat index j*6+k.
    # C is 4x6. Flat index k*4+i? 
    # Prompt: W[k*n+i, r]. n=4. So W index is k*4+i.
    
    # Let's define the local Strassen terms for 2x2x2.
    # Term 1:
    # u: [1, 0, 0, 1] (A00, A01, A10, A11) -> A00 + A11
    # v: [1, 0, 0, 1] (B00, B01, B10, B11) -> B00 + B11
    # w: C00 += M1, C01 += M1, C10 += M1, C11 += M1
    # So w: [1, 1, 1, 1]
    
    # Term 2:
    # u: [0, 0, 1, 1] -> A10 + A11
    # v: [1, 0, 0, 0] -> B00
    # w: C10 += M2, C11 += M2
    # So w: [0, 0, 1, 1]
    
    # Term 3:
    # u: [1, 0, 0, 0] -> A00
    # v: [0, 1, 0, -1] -> B01 - B11
    # w: C01 += M3, C11 += M3
    # So w: [0, 1, 0, 1]
    
    # Term 4:
    # u: [0, 0, 0, 1] -> A11
    # v: [0, -1, 1, 0] -> B10 - B00
    # w: C10 += M4, C11 += M4
    # So w: [0, 0, 1, 1] -- Wait, C10 and C11?
    # C = [[C00, C01], [C10, C11]]
    # M4 = A11(B10-B00).
    # C10 = A10B00 + A11B10.
    # C11 = A10B01 + A11B11.
    # M4 contributes to C10 (A11B10) and C11 (-A11B00)?
    # No, M4 is used in reconstruction.
    # C10 = M2 - M4 + M6?
    # C11 = M1 - M3 + M4 + M7?
    # Let's use the standard reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # Let's re-verify M definitions:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # So for each term r, we have u_r, v_r, and w_r (4 entries).
    
    strassen_terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2 -> C10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3 -> C01
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 1]), # M4 -> C10, C11? 
        # Wait, M4 contributes to C10 and C11?
        # C10 = M2 + M4. C11 = M1 - M3 + M6 + M7.
        # So M4 only contributes to C10?
        # Let's check M4 = A11(B10-B00).
        # C10 term from A11 is A11B10.
        # C11 term from A11 is A11B11.
        # So M4 does NOT contribute to C11 directly in the formula above?
        # Formula: C11 = M1 - M3 + M6 + M7.
        # M1 has A11B11. M3 has -A00B11. M6 has A10B01 - A00B01. M7 has A01B10 - A11B10.
        # Sum: A11B11 - A00B11 + A10B01 - A00B01 + A01B10 - A11B10.
        # Actual C11 = A10B01 + A11B11.
        # Missing: A00B11 - A00B01 - A11B10 + A01B10?
        # This is getting complicated to verify manually.
        # I will use the known valid Strassen decomposition vectors.
    ]
    
    # Correct Strassen vectors (u, v, w) for 2x2x2
    # w is flattened C (C00, C01, C10, C11)
    terms_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 1]),  # M4 -- Correction: M4 contributes to C10 and C11?
        # Let's assume the standard implementation in the seed was correct.
        # I will copy the logic from the seed but clean it up.
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7
    ]
    
    # Note: The w vectors above are guesses based on standard Strassen.
    # To be safe, I will use the exact same coefficients as the seed program which is known to work (score 1.2468).
    
    alpha = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], 
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], 
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    recon = [
        [[0, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, -1]], [[0, 1, 1], [1, 1, 1]],
        [[0, 0, 1], [1, 0, 1]], [[0, 0, -1], [0, 1, 1]], [[1, 1, 1]], [[0, 0, 1]]
    ]
    
    col = 0
    
    # Part 1: 4x4x4
    for s in range(7):
        for si in range(7):
            # U
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, br = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar2, br2 = r // 2, r % 2
                            u_idx = (ar2 * 2 + ar) * 4 + (br2 * 2 + br)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            # V
            for b in range(4):
                if beta[si][b] != 0:
                    br_row, br_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + br_row) * 6 + (bc * 2 + br_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            # W
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
            
    # Part 2: 4x4x2
    terms = [
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (2, 0, 1, 0),
        (3, 1, 1, 0)
    ]
    
    for t in terms:
        a_blk, b_blk, c_r, c_c = t
        for si in range(7):
            # U
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, br = a // 2, a % 2
                    ar_blk, ac_blk = a_blk // 2, a_blk % 2
                    u_idx = (ar_blk * 2 + ar) * 4 + (ac_blk * 2 + br)
                    U[u_idx, col] += alpha[si][a]
            # V
            br_blk = b_blk
            for b in range(4):
                if beta[si][b] != 0:
                    br, bc = b // 2, b % 2
                    v_idx = (br_blk * 2 + br) * 6 + (4 + bc)
                    V[v_idx, col] += beta[si][b]
            # W
            for za, zb, zgamma in recon[si]:
                w_idx = (4 + zb) * 4 + (c_r * 2 + za)
                W[w_idx, col] += zgamma
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
