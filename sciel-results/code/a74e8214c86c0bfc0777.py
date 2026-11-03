"""Tensor decomposition for 2x4 * 4x6 matrix multiplication using Strassen-based blocks."""
DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    # We will build the decomposition column by column.
    # Each column t corresponds to one multiplication term.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # We split the 4x4 middle dimension into two 2x2 blocks vertically?
    # No, A is 2x4, B is 4x6.
    # Let's view A as two 2x2 blocks side-by-side: A = [A_L, A_R] where A_L, A_R are 2x2.
    # Let's view B as two 2x6 blocks top and bottom: B = [B_T; B_B] where B_T, B_B are 2x6.
    # Then C = A_L * B_T + A_R * B_B.
    # A_L * B_T is a 2x6 product of a 2x2 and a 2x6.
    # A_R * B_B is a 2x6 product of a 2x2 and a 2x6.
    
    # Can we decompose 2x2 * 2x6 more efficiently?
    # The tensor is M(2,2,6).
    # Schoolbook rank is 2*2*6 = 24.
    # Can we do better?
    # Note that 2x2 * 2x6 is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.
    # This is equivalent to 3 separate 2x2 * 2x2 multiplications? No, the right matrix is wider.
    # Actually, 2x2 * 2x6 can be seen as 2x2 * (2x2, 2x2, 2x2).
    # So it's 3 instances of 2x2 * 2x2.
    # If we use Strassen for each 2x2*2x2, we need 7 multiplications per 2x2*2x2 block.
    # So 3 * 7 = 21 multiplications for 2x2 * 2x6.
    # Total for 2x4 * 4x6 would be 21 + 21 = 42.
    
    # Let's verify this structure.
    # A_L is 2x2, B_T is 2x6 = [B_T0, B_T1, B_T2] where each is 2x2.
    # A_L * B_T = [A_L B_T0, A_L B_T1, A_L B_T2].
    # We can compute each A_L B_Ti using Strassen with 7 products.
    # Since the left matrix A_L is the same for all three, can we share products?
    # Strassen products for A_L * B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # Here A is fixed (A_L), but B changes (B_T0, B_T1, B_T2).
    # The terms involving only A (like A11, A22, A11+A22, etc.) are constant for the 3 blocks.
    # The terms involving B change.
    # So for each of the 3 blocks, we have 7 products.
    # Total 21 products for A_L * B_T.
    # Similarly 21 for A_R * B_B.
    # Total 42.
    
    # Let's implement this.
    # We need to map the Strassen terms to U, V, W.
    
    # Strassen 2x2*2x2 decomposition:
    # Let A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]].
    # C = A*B.
    # Indices:
    # U indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (a00+a11)(b00+b11)
    # U: 0:1, 3:1
    # V: 0:1, 3:1
    # W: 0:1, 1:1, 2:1, 3:1 (Actually, M1 contributes to all C_ij? No.)
    # C00 = M1 - M2 + M3 + M6
    # C01 = M4 + M5
    # C10 = M2 + M7
    # C11 = M1 - M4 + M5 + M7
    
    # Let's list the contributions of each M to C:
    # M1: C00:+1, C11:+1
    # M2: C00:-1, C10:+1
    # M3: C00:+1
    # M4: C01:+1, C11:-1
    # M5: C01:+1, C11:+1
    # M6: C00:+1
    # M7: C10:+1, C11:+1
    
    # Let's double check:
    # C00 = M1 - M2 + M3 + M6
    # C01 = M4 + M5
    # C10 = M2 + M7
    # C11 = M1 - M4 + M5 + M7
    
    # This is the standard Strassen decomposition.
    
    # Now, for 2x2 * 2x6:
    # A is 2x2. B is 2x6 = [B0, B1, B2] where Bk is 2x2.
    # C is 2x6 = [C0, C1, C2] where Ck is 2x2.
    # Ck = A * Bk.
    # We use Strassen for each k.
    # The U part (A coefficients) is the same for all k.
    # The V part (B coefficients) depends on k.
    # The W part (C coefficients) depends on k.
    
    # Let's define the Strassen terms for a single 2x2*2x2:
    # Term t=0 (M1):
    #   U: {0:1, 3:1}
    #   V: {0:1, 3:1}
    #   W: {0:1, 3:1}  (Contributes to C00 and C11)
    # Term t=1 (M2):
    #   U: {2:1, 3:1}  (a10+a11)
    #   V: {0:1}       (b00)
    #   W: {0:-1, 2:1} (Contributes to C00:-1, C10:+1)
    # Term t=2 (M3):
    #   U: {0:1}       (a00)
    #   V: {2:1, 3:-1} (b10-b11)
    #   W: {0:1}       (Contributes to C00:+1)
    # Term t=3 (M4):
    #   U: {3:1}       (a11)
    #   V: {1:1, 0:-1} (b01-b00)
    #   W: {1:1, 3:-1} (Contributes to C01:+1, C11:-1)
    # Term t=4 (M5):
    #   U: {0:1, 1:1}  (a00+a01)
    #   V: {3:1}       (b11)
    #   W: {1:1, 3:1}  (Contributes to C01:+1, C11:+1)
    # Term t=5 (M6):
    #   U: {2:1, 0:-1} (a10-a00)
    #   V: {0:1, 1:1}  (b00+b01)
    #   W: {0:1}       (Contributes to C00:+1)
    # Term t=6 (M7):
    #   U: {1:1, 3:-1} (a01-a11)
    #   V: {2:1, 3:1}  (b10+b11)
    #   W: {2:1, 3:1}  (Contributes to C10:+1, C11:+1)
    
    # Let's re-verify M2:
    # M2 = (a10+a11)b00.
    # C00 = ... - M2 ... -> W00:-1.
    # C10 = M2 + ... -> W10:+1.
    # Correct.
    
    # Let's re-verify M3:
    # M3 = a00(b10-b11).
    # C00 = ... + M3 ... -> W00:+1.
    # Correct.
    
    # Let's re-verify M4:
    # M4 = a11(b01-b00).
    # C01 = M4 + ... -> W01:+1.
    # C11 = ... - M4 ... -> W11:-1.
    # Correct.
    
    # Let's re-verify M5:
    # M5 = (a00+a01)b11.
    # C01 = ... + M5 ... -> W01:+1.
    # C11 = ... + M5 ... -> W11:+1.
    # Correct.
    
    # Let's re-verify M6:
    # M6 = (a10-a00)(b00+b01).
    # C00 = ... + M6 ... -> W00:+1.
    # Correct.
    
    # Let's re-verify M7:
    # M7 = (a01-a11)(b10+b11).
    # C10 = ... + M7 ... -> W10:+1.
    # C11 = ... + M7 ... -> W11:+1.
    # Correct.
    
    # Now, for the full 2x4 * 4x6:
    # A = [A_L, A_R] (2x4)
    # B = [B_T; B_B] (4x6)
    # C = A_L B_T + A_R B_B
    
    # A_L is 2x2. Indices in A: 0,1,2,3.
    # A_R is 2x2. Indices in A: 4,5,6,7.
    # B_T is 2x6. Indices in B: 0..11.
    # B_B is 2x6. Indices in B: 12..23.
    # C is 2x6. Indices in C: 0..11.
    
    # We have two main parts: Part 1 (A_L B_T) and Part 2 (A_R B_B).
    # Each part involves 3 Strassen 2x2*2x2 multiplications.
    # Total 6 * 7 = 42 terms.
    
    # Let's build the matrices.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8x42
    V = [[0] * rank for _ in range(m * p)]  # 24x42
    W = [[0] * rank for _ in range(p * n)]  # 12x42
    
    # Helper to add a term
    # term_id: global column index
    # u_dict: {local_a_idx: val}
    # v_dict: {local_b_idx: val}
    # w_dict: {local_c_idx: val}
    # a_offset: offset for A indices
    # b_offset: offset for B indices
    # c_offset: offset for C indices
    
    def add_term(term_id, u_dict, v_dict, w_dict, a_off, b_off, c_off):
        for a_idx, val in u_dict.items():
            U[a_idx + a_off][term_id] = val
        for b_idx, val in v_dict.items():
            V[b_idx + b_off][term_id] = val
        for c_idx, val in w_dict.items():
            W[c_idx + c_off][term_id] = val
    
    # Strassen terms definitions
    # t=0 (M1)
    S0_U = {0:1, 3:1}
    S0_V = {0:1, 3:1}
    S0_W = {0:1, 3:1}
    # t=1 (M2)
    S1_U = {2:1, 3:1}
    S1_V = {0:1}
    S1_W = {0:-1, 2:1}
    # t=2 (M3)
    S2_U = {0:1}
    S2_V = {2:1, 3:-1}
    S2_W = {0:1}
    # t=3 (M4)
    S3_U = {3:1}
    S3_V = {1:1, 0:-1}
    S3_W = {1:1, 3:-1}
    # t=4 (M5)
    S4_U = {0:1, 1:1}
    S4_V = {3:1}
    S4_W = {1:1, 3:1}
    # t=5 (M6)
    S5_U = {2:1, 0:-1}
    S5_V = {0:1, 1:1}
    S5_W = {0:1}
    # t=6 (M7)
    S6_U = {1:1, 3:-1}
    S6_V = {2:1, 3:1}
    S6_W = {2:1, 3:1}
    
    strassen_terms = [
        (S0_U, S0_V, S0_W),
        (S1_U, S1_V, S1_W),
        (S2_U, S2_V, S2_W),
        (S3_U, S3_V, S3_W),
        (S4_U, S4_V, S4_W),
        (S5_U, S5_V, S5_W),
        (S6_U, S6_V, S6_W)
    ]
    
    term_id = 0
    # Part 1: A_L * B_T
    # A_L offset: 0
    # B_T is 2x6, split into 3 2x2 blocks: B_T0, B_T1, B_T2
    # B_T0 offset in B: 0
    # B_T1 offset in B: 2
    # B_T2 offset in B: 4
    # C offset: 0
    for k in range(3):
        b_off = k * 2
        c_off = k * 2
        for s in range(7):
            u_dict, v_dict, w_dict = strassen_terms[s]
            add_term(term_id, u_dict, v_dict, w_dict, 0, b_off, c_off)
            term_id += 1
            
    # Part 2: A_R * B_B
    # A_R offset: 4
    # B_B is 2x6, split into 3 2x2 blocks: B_B0, B_B1, B_B2
    # B_B0 offset in B: 12
    # B_B1 offset in B: 14
    # B_B2 offset in B: 16
    # C offset: 0 (Same C, so we add to the same C indices? No, C is the sum.
    # Wait, C = A_L B_T + A_R B_B.
    # The W matrix accumulates contributions.
    # So for Part 2, the C indices are the same as Part 1.
    # So c_off is 0, 2, 4 again.
    for k in range(3):
        b_off = 12 + k * 2
        c_off = k * 2
        for s in range(7):
            u_dict, v_dict, w_dict = strassen_terms[s]
            add_term(term_id, u_dict, v_dict, w_dict, 4, b_off, c_off)
            term_id += 1
            
    return {"U": U, "V": V, "W": W}
