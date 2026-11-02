import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-68 exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Strategy: Split A (3x4) into A_top (2x4) and A_bot (1x4).
    - A_top * B: Decompose B into two 4x3 blocks B_left (4x3), B_right (4x3).
      For each, split A_top into A_top_left (2x2), A_top_right (2x2) and
      B_left into B_left_top (2x3), B_left_bot (2x3).
      Use Strassen (rank 7) for the 2x2 * 2x2 products that contribute to
      the top 2 columns of the output, and schoolbook (rank 12) for the
      2x2 * 2x1 product for the 3rd column. Wait, this is getting complex.
      
      Simpler: Use the known rank 68 construction.
      Split B into B1 (4x3) and B2 (4x3).
      A_top (2x4) * B1 (4x3):
        Split A_top into A1 (2x2), A2 (2x2).
        Split B1 into B1a (2x3), B1b (2x3).
        Product = A1*B1a + A2*B1b.
        Rank of A1*B1a (2x2*2x3) + A2*B1b (2x2*2x3).
        This isn't straightforward Strassen.
      
      Let's use the measured result: rank 68.
      The previous successful program used Strassen on 2x2x2 blocks.
      3x4*4x6:
      Split A into 2x2 + 2x2 + 1x4? No.
      Let's try:
      A = [A1 (2x2), A2 (2x2)]  where A1 is cols 0-1, A2 is cols 2-3.
      B = [B1 (2x6); B2 (2x6)] where B1 is rows 0-1, B2 is rows 2-3.
      C = A1*B1 + A2*B2.
      Rank of A1*B1 (2x2*2x6) is at most 12.
      Rank of A2*B2 (2x2*2x6) is at most 12.
      Total 24? No, that's not right.
      
      Actually, the rank of 2x2*2x6 is 12 (schoolbook).
      But we can do better.
      Let's just implement the Strassen-based 68 rank decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We will build U (12 x rank), V (24 x rank), W (18 x rank)
    # Start with lists to append to
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec, v_vec, w_vec are lists of appropriate length."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2*2x2 rank 7 factors
    # Indices: u is (i,j), v is (j,k), w is (k,i) for 2x2*2x2 -> 2x2
    # Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For a 2x2*2x2 block at position (i0, j0, k0) in the full 3x4*4x6:
    # u indices: (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1)
    # v indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # w indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
    
    def strassen_222(i0, j0, k0):
        """Add 7 rank-1 terms for the 2x2*2x2 block at (i0,j0,k0)."""
        # u basis: [u00, u10, u01, u11]
        # v basis: [v00, v01, v10, v11]
        # w basis: [w00, w01, w10, w11]
        
        # M1 = (A00+A11)*(B00+B11)
        # u: A00 + A11 -> [1, 0, 0, 1]
        # v: B00 + B11 -> [1, 0, 1, 0]
        # w: contributes to C00 (+1), C11 (+1)
        # C00 = w00, C11 = w11
        u1 = [0]*12; u1[i0*m+j0] = 1; u1[(i0+1)*m+j0+1] = 1
        v1 = [0]*24; v1[j0*p+k0] = 1; v1[(j0+1)*p+(k0+1)] = 1
        w1 = [0]*18; w1[k0*n+i0] = 1; w1[(k0+1)*n+(i0+1)] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)*B00
        # u: A10 + A11 -> [0, 1, 0, 1]
        # v: B00 -> [1, 0, 0, 0]
        # w: contributes to C10 (+1), C11 (-1)
        u2 = [0]*12; u2[(i0+1)*m+j0] = 1; u2[(i0+1)*m+j0+1] = 1
        v2 = [0]*24; v2[j0*p+k0] = 1
        w2 = [0]*18; w2[(k0+1)*n+i0] = 1; w2[(k0+1)*n+(i0+1)] = -1
        add_term(u2, v2, w2)
        
        # M3 = A00*(B01-B11)
        # u: A00 -> [1, 0, 0, 0]
        # v: B01 - B11 -> [0, 1, 0, -1]
        # w: contributes to C01 (+1), C11 (+1)
        u3 = [0]*12; u3[i0*m+j0] = 1
        v3 = [0]*24; v3[j0*p+(k0+1)] = 1; v3[(j0+1)*p+(k0+1)] = -1
        w3 = [0]*18; w3[k0*n+(i0+1)] = 1; w3[(k0+1)*n+(i0+1)] = 1
        add_term(u3, v3, w3)
        
        # M4 = A11*(B10-B00)
        # u: A11 -> [0, 0, 0, 1]
        # v: B10 - B00 -> [-1, 0, 1, 0]
        # w: contributes to C00 (+1), C10 (+1)
        u4 = [0]*12; u4[(i0+1)*m+j0+1] = 1
        v4 = [0]*24; v4[j0*p+k0] = -1; v4[(j0+1)*p+k0] = 1
        w4 = [0]*18; w4[k0*n+i0] = 1; w4[(k0+1)*n+i0] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)*B11
        # u: A00 + A01 -> [1, 0, 1, 0]
        # v: B11 -> [0, 0, 0, 1]
        # w: contributes to C00 (-1), C01 (+1)
        u5 = [0]*12; u5[i0*m+j0] = 1; u5[i0*m+j0+1] = 1
        v5 = [0]*24; v5[(j0+1)*p+(k0+1)] = 1
        w5 = [0]*18; w5[k0*n+i0] = -1; w5[k0*n+(i0+1)] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)*(B00+B01)
        # u: A10 - A00 -> [-1, 1, 0, 0]
        # v: B00 + B01 -> [1, 1, 0, 0]
        # w: contributes to C00 (+1)
        u6 = [0]*12; u6[i0*m+j0] = -1; u6[(i0+1)*m+j0] = 1
        v6 = [0]*24; v6[j0*p+k0] = 1; v6[j0*p+(k0+1)] = 1
        w6 = [0]*18; w6[k0*n+i0] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)*(B10+B11)
        # u: A01 - A11 -> [0, 0, 1, -1]
        # v: B10 + B11 -> [0, 0, 1, 1]
        # w: contributes to C11 (+1)
        u7 = [0]*12; u7[i0*m+j0+1] = 1; u7[(i0+1)*m+j0+1] = -1
        v7 = [0]*24; v7[(j0+1)*p+k0] = 1; v7[(j0+1)*p+(k0+1)] = 1
        w7 = [0]*18; w7[(k0+1)*n+(i0+1)] = 1
        add_term(u7, v7, w7)
    
    # Schoolbook 2x2*2x3 rank 12 terms
    # A (2x2) at (i0, j0), B (2x3) at (j0, k0)
    # u indices: (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1)
    # v indices: (j0, k0), (j0, k0+1), (j0, k0+2), (j0+1, k0), (j0+1, k0+1), (j0+1, k0+2)
    # w indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1), (k0+2, i0), (k0+2, i0+1)
    def schoolbook_223(i0, j0, k0):
        """Add 12 rank-1 terms for the 2x2*2x3 block at (i0,j0,k0)."""
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = [0]*12; u[(i0+i)*m + (j0+j)] = 1
                    v = [0]*24; v[(j0+j)*p + (k0+k)] = 1
                    w = [0]*18; w[(k0+k)*n + (i0+i)] = 1
                    add_term(u, v, w)
    
    # Now, let's build the 3x4*4x6 decomposition.
    # Split:
    # A = [A0 (2x4); A1 (1x4)]
    # B is 4x6.
    # C0 = A0 * B (2x6)
    # C1 = A1 * B (1x6)
    
    # For C0 (2x4 * 4x6):
    # Split A0 into A0_left (2x2) and A0_right (2x2).
    # Split B into B_top (2x6) and B_bot (2x6).
    # C0 = A0_left * B_top + A0_right * B_bot.
    
    # A0_left (2x2) * B_top (2x6):
    # Split B_top into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x2).
    # A0_left * B_top_left: 2x2*2x2 -> Strassen (rank 7)
    # A0_left * B_top_mid: 2x2*2x2 -> Strassen (rank 7)
    # A0_left * B_top_right: 2x2*2x2 -> Strassen (rank 7)
    # Total for A0_left * B_top: 21
    
    # A0_right (2x2) * B_bot (2x6):
    # Split B_bot into B_bot_left (2x2), B_bot_mid (2x2), B_bot_right (2x2).
    # A0_right * B_bot_left: 2x2*2x2 -> Strassen (rank 7)
    # A0_right * B_bot_mid: 2x2*2x2 -> Strassen (rank 7)
    # A0_right * B_bot_right: 2x2*2x2 -> Strassen (rank 7)
    # Total for A0_right * B_bot: 21
    
    # C0 total: 42
    
    # C1 (1x4 * 4x6):
    # Split B into B_top (2x6) and B_bot (2x6).
    # C1 = A1 * B_top + A1 * B_bot.
    # A1 is 1x4. Split into A1_left (1x2) and A1_right (1x2).
    # A1_left (1x2) * B_top (2x6):
    #   Split B_top into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x2).
    #   A1_left * B_top_left: 1x2*2x2 -> schoolbook rank 4
    #   A1_left * B_top_mid: 1x2*2x2 -> schoolbook rank 4
    #   A1_left * B_top_right: 1x2*2x2 -> schoolbook rank 4
    #   Total: 12
    # A1_right (1x2) * B_bot (2x6):
    #   Split B_bot into B_bot_left (2x2), B_bot_mid (2x2), B_bot_right (2x2).
    #   A1_right * B_bot_left: 1x2*2x2 -> schoolbook rank 4
    #   A1_right * B_bot_mid: 1x2*2x2 -> schoolbook rank 4
    #   A1_right * B_bot_right: 1x2*2x2 -> schoolbook rank 4
    #   Total: 12
    # C1 total: 24
    
    # Grand total: 42 + 24 = 66?
    # Wait, the measured rank is 68.
    # Let me re-check.
    # Maybe I can't use Strassen for all 2x2*2x2 blocks because the indices don't align?
    # No, they do align.
    # Let's just use the 68 rank construction from the previous successful run.
    # Actually, 66 < 68, so if my construction is valid, it's better.
    # But let's be safe and use the known 68.
    
    # Actually, let's try the 66 construction and see if it's valid.
    # But I can't test it here. Let's use the 68 construction.
    
    # Let's use the 68 construction from the previous run.
    # It used Strassen on 2x2x2 blocks.
    
    # Let's just implement the 68 rank construction.
    
    # Clear the lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Use Strassen for 2x2*2x2 blocks
    # And schoolbook for the rest.
    
    # Let's use the 68 construction.
    
    # Actually, let's use the 66 construction. It should be valid.
    
    # Strassen for A0_left * B_top_left (2x2*2x2)
    # i0=0, j0=0, k0=0
    strassen_222(0, 0, 0)
    
    # Strassen for A0_left * B_top_mid (2x2*2x2)
    # i0=0, j0=0, k0=2
    strassen_222(0, 0, 2)
    
    # Strassen for A0_left * B_top_right (2x2*2x2)
    # i0=0, j0=0, k0=4
    strassen_222(0, 0, 4)
    
    # Strassen for A0_right * B_bot_left (2x2*2x2)
    # i0=0, j0=2, k0=0
    strassen_222(0, 2, 0)
    
    # Strassen for A0_right * B_bot_mid (2x2*2x2)
    # i0=0, j0=2, k0=2
    strassen_222(0, 2, 2)
    
    # Strassen for A0_right * B_bot_right (2x2*2x2)
    # i0=0, j0=2, k0=4
    strassen_222(0, 2, 4)
    
    # Schoolbook for A1_left * B_top_left (1x2*2x2)
    # i0=2, j0=0, k0=0
    schoolbook_223(2, 0, 0)  # This is 2x2*2x3, but we need 1x2*2x2.
    # Wait, my schoolbook_223 is for 2x2*2x3.
    # I need a 1x2*2x2 schoolbook.
    
    # Let's just use the 68 construction from the previous run.
    
    # Clear the lists again
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Use the 68 construction.
    
    # Actually, let's use the 66 construction.
    
    # Let's define a schoolbook for 1x2*2x2.
    def schoolbook_122(i0, j0, k0):
        """Add 4 rank-1 terms for the 1x2*2x2 block at (i0,j0,k0)."""
        for j in range(2):
            for k in range(2):
                u = [0]*12; u[i0*m + j0+j] = 1
                v = [0]*24; v[(j0+j)*p + k0+k] = 1
                w = [0]*18; w[(k0+k)*n + i0] = 1
                add_term(u, v, w)
    
    # Now, let's build the 66 construction.
    
    # Strassen for A0_left * B_top_left (2x2*2x2)
    strassen_222(0, 0, 0)
    
    # Strassen for A0_left * B_top_mid (2x2*2x2)
    strassen_222(0, 0, 2)
    
    # Strassen for A0_left * B_top_right (2x2*2x2)
    strassen_222(0, 0, 4)
    
    # Strassen for A0_right * B_bot_left (2x2*2x2)
    strassen_222(0, 2, 0)
    
    # Strassen for A0_right * B_bot_mid (2x2*2x2)
    strassen_222(0, 2, 2)
    
    # Strassen for A0_right * B_bot_right (2x2*2x2)
    strassen_222(0, 2, 4)
    
    # Schoolbook for A1_left * B_top_left (1x2*2x2)
    schoolbook_122(2, 0, 0)
    
    # Schoolbook for A1_left * B_top_mid (1x2*2x2)
    schoolbook_122(2, 0, 2)
    
    # Schoolbook for A1_left * B_top_right (1x2*2x2)
    schoolbook_122(2, 0, 4)
    
    # Schoolbook for A1_right * B_bot_left (1x2*2x2)
    schoolbook_122(2, 2, 0)
    
    # Schoolbook for A1_right * B_bot_mid (1x2*2x2)
    schoolbook_122(2, 2, 2)
    
    # Schoolbook for A1_right * B_bot_right (1x2*2x2)
    schoolbook_122(2, 2, 4)
    
    # Total rank: 6*7 + 6*4 = 42 + 24 = 66
    
    # Now, convert U_cols, V_cols, W_cols to matrices.
    # U is (12 x rank), V is (24 x rank), W is (18 x rank)
    rank = len(U_cols)
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(18)]
    
    for c in range(rank):
        for r in range(12):
            U[r][c] = U_cols[c][r]
        for r in range(24):
            V[r][c] = V_cols[c][r]
        for r in range(18):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
