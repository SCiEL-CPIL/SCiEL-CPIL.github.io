import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank-26 decomposition for 3x3x3, duplicated for 3x3x6.
    # Total rank = 52.
    rank = 52
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # term: (U_idx, V_idx, W_idx, coeff)
    # U_idx is (i, j), V_idx is (j, k), W_idx is (k, i)
    # We will generate the 26 terms for the first half (k in 0..2)
    # and duplicate them for the second half (k in 3..5)
    
    terms = []
    
    # 1. A22 * B22 (Strassen, Rank 7)
    # A22 indices: (0,0), (0,1), (1,0), (1,1) -> U[0], U[1], U[3], U[4]
    # B22 indices: (0,0), (0,1), (1,0), (1,1) -> V[0], V[1], V[3], V[4]
    # C22 indices: (0,0), (0,1), (1,0), (1,1) -> W[0], W[1], W[3], W[4] (for k=0,1)
    
    # Strassen 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global U, V, W.
    # For the first half (k=0,1,2), the k indices in V and W are 0,1,2.
    # B22 corresponds to B[0,0], B[0,1], B[1,0], B[1,1].
    # So in V, these are indices (0,0), (0,1), (1,0), (1,1).
    # In W, for C22, the k indices are 0,1.
    # C11 (global 0,0) -> W[0*3+0] = W[0]
    # C12 (global 0,1) -> W[1*3+0] = W[3]
    # C21 (global 1,0) -> W[0*3+1] = W[1]
    # C22 (global 1,1) -> W[1*3+1] = W[4]
    
    # This is getting complicated to hardcode manually.
    # I will write a helper function to add a term.
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        # u_idx: (i, j)
        # v_idx: (j, k)
        # w_idx: (k, i)
        col = len(terms)
        i, j = u_idx
        k, i_w = w_idx
        # Check consistency i == i_w
        U[i * m + j][col] = coeff
        V[j * p + k][col] = coeff
        W[k * n + i][col] = coeff
        terms.append((u_idx, v_idx, w_idx, coeff))

    # We will generate terms for the first half (k in 0..2)
    # Then duplicate for second half (k in 3..5)
    
    # Strassen 2x2 for A22 B22
    # A22: (0,0), (0,1), (1,0), (1,1)
    # B22: (0,0), (0,1), (1,0), (1,1)
    # C22: (0,0), (0,1), (1,0), (1,1)
    
    # M1
    add_term((0,0), (0,0), (0,0), 1) # A11 B11
    add_term((0,0), (0,1), (1,0), 1) # A11 B12
    add_term((0,1), (0,0), (0,1), 1) # A12 B11
    add_term((0,1), (0,1), (1,1), 1) # A12 B12
    add_term((1,0), (0,0), (0,0), 1) # A21 B11
    add_term((1,0), (0,1), (1,0), 1) # A21 B12
    add_term((1,1), (0,0), (0,1), 1) # A22 B11
    add_term((1,1), (0,1), (1,1), 1) # A22 B12
    # This is schoolbook for 2x2, rank 4. I wanted Strassen rank 7.
    # Let's just use schoolbook for 2x2 to be safe and simple?
    # Rank 4 for 2x2.
    # Total rank for 3x3x3 with schoolbook 2x2:
    # 4 (A22B22) + 4 (abT) + 4 (A22b) + 2 (ab11) + 4 (aTB22) + 2 (a11bT) + 2 (aTb) + 1 (a11b11)
    # 4+4+4+2+4+2+2+1 = 23.
    # Wait, schoolbook for 2x2 is rank 4.
    # So total rank is 23?
    # No, Strassen is 7. Schoolbook is 4.
    # If I use schoolbook for 2x2, the rank is LOWER?
    # No, Strassen is 7, which is HIGHER than 4?
    # No, Strassen is 7 multiplications. Schoolbook is 4?
    # 2x2 schoolbook: 4 multiplications.
    # Strassen: 7 multiplications.
    # Strassen is SLOWER for 2x2?
    # No, Strassen is 7. Schoolbook is 4.
    # Wait, Strassen is for 2x2. It uses 7 multiplications.
    # Schoolbook uses 4.
    # Why is Strassen better?
    # Because for large matrices, the exponent is lower.
    # For 2x2, schoolbook is better (4 < 7).
    # So I should use schoolbook for the 2x2 block!
    # Rank of 2x2 schoolbook is 4.
    
    # So:
    # 1. A22 B22: Rank 4.
    # 2. a bT: Rank 4.
    # 3. A22 b: Rank 4.
    # 4. a b11: Rank 2.
    # 5. aT B22: Rank 4.
    # 6. a11 bT: Rank 2.
    # 7. aT b: Rank 2.
    # 8. a11 b11: Rank 1.
    # Total: 4+4+4+2+4+2+2+1 = 23.
    
    # So I can get rank 23 for 3x3x3 using schoolbook for all sub-blocks!
    # This is the same as the optimal rank 23?
    # Yes, 23 is the known optimal rank.
    # So I just need to implement the schoolbook decomposition for the partitioned 3x3x3.
    
    # Let's re-verify the rank of 2x2 schoolbook.
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # 4 terms. Correct.
    
    # So I will implement the 23-term decomposition for 3x3x3.
    # And duplicate for 3x3x6.
    # Total rank 46.
    
    # I will clear the previous terms and restart.
    terms.clear()
    rank = 46
    
    # 1. A22 B22 (4 terms)
    # (0,0)*(0,0) -> (0,0)
    add_term((0,0), (0,0), (0,0), 1)
    # (0,0)*(0,1) -> (0,1)
    add_term((0,0), (0,1), (1,0), 1)
    # (0,1)*(0,0) -> (0,0)
    add_term((0,1), (0,0), (0,0), 1)
    # (0,1)*(0,1) -> (0,1)
    add_term((0,1), (0,1), (1,1), 1)
    # (1,0)*(0,0) -> (1,0)
    add_term((1,0), (0,0), (0,1), 1)
    # (1,0)*(0,1) -> (1,1)
    add_term((1,0), (0,1), (1,1), 1)
    # (1,1)*(0,0) -> (1,0)
    add_term((1,1), (0,0), (0,1), 1)
    # (1,1)*(0,1) -> (1,1)
    add_term((1,1), (0,1), (1,1), 1)
    
    # 2. a bT (4 terms)
    # a: (0,2), (1,2)
    # bT: (2,0), (2,1)
    # a0 bT0 -> C00
    add_term((0,2), (2,0), (0,0), 1)
    # a0 bT1 -> C01
    add_term((0,2), (2,1), (1,0), 1)
    # a1 bT0 -> C10
    add_term((1,2), (2,0), (0,1), 1)
    # a1 bT1 -> C11
    add_term((1,2), (2,1), (1,1), 1)
    
    # 3. A22 b (4 terms)
    # A22: (0,0), (0,1), (1,0), (1,1)
    # b: (0,2), (1,2)
    # A00 b0 -> C02
    add_term((0,0), (0,2), (2,0), 1)
    # A00 b1 -> C02? No, A00 * B02?
    # B indices: j,k. b is column 2.
    # B02 is (0,2). B12 is (1,2).
    # C02 = A00 B02 + A01 B12
    add_term((0,0), (0,2), (2,0), 1) # Already added? No, (0,0) is A00.
    # Wait, I added (0,0) in step 1.
    # Step 3:
    # A00 B02 -> C02
    add_term((0,0), (0,2), (2,0), 1)
    # A01 B12 -> C02
    add_term((0,1), (1,2), (2,0), 1)
    # A10 B02 -> C12
    add_term((1,0), (0,2), (2,1), 1)
    # A11 B12 -> C12
    add_term((1,1), (1,2), (2,1), 1)
    
    # 4. a b11 (2 terms)
    # a0 b11 -> C02
    add_term((0,2), (2,2), (2,0), 1)
    # a1 b11 -> C12
    add_term((1,2), (2,2), (2,1), 1)
    
    # 5. aT B22 (4 terms)
    # aT: (2,0), (2,1)
    # B22: (0,0), (0,1), (1,0), (1,1)
    # aT0 B00 -> C20
    add_term((2,0), (0,0), (0,2), 1)
    # aT0 B10 -> C20? No.
    # C20 = A20 B00 + A21 B10
    add_term((2,0), (0,0), (0,2), 1)
    add_term((2,1), (1,0), (0,2), 1)
    # C21 = A20 B01 + A21 B11
    add_term((2,0), (0,1), (1,2), 1)
    add_term((2,1), (1,1), (1,2), 1)
    
    # 6. a11 bT (2 terms)
    # a11 bT0 -> C20
    add_term((2,2), (2,0), (0,2), 1)
    # a11 bT1 -> C21
    add_term((2,2), (2,1), (1,2), 1)
    
    # 7. aT b (2 terms)
    # aT0 b0 -> C22
    add_term((2,0), (0,2), (2,2), 1)
    # aT1 b1 -> C22
    add_term((2,1), (1,2), (2,2), 1)
    
    # 8. a11 b11 (1 term)
    add_term((2,2), (2,2), (2,2), 1)
    
    # Now duplicate for second half (k=3,4,5)
    # The terms above used k in 0,1,2.
    # We need to add terms with k in 3,4,5.
    # The structure is identical.
    # For each term (u, v, w, coeff) in terms:
    # v_idx is (j, k). New k = k + 3.
    # w_idx is (k, i). New k = k + 3.
    
    # I will just copy the terms and adjust indices.
    # But I need to add them to U, V, W.
    # U is unchanged.
    # V index changes.
    # W index changes.
    
    for u_idx, v_idx, w_idx, coeff in terms:
        i, j = u_idx
        j_v, k_v = v_idx
        k_w, i_w = w_idx
        
        new_k_v = k_v + 3
        new_k_w = k_w + 3
        
        col = len(terms)
        U[i * m + j][col] = coeff
        V[j_v * p + new_k_v][col] = coeff
        W[new_k_w * n + i_w][col] = coeff
        terms.append((u_idx, (j_v, new_k_v), (new_k_w, i_w), coeff))
        
    return {"U": U, "V": V, "W": W}
