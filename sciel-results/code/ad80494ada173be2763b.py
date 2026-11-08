"""Matrix multiplication 2x3x6 with a block-strassen style rank-16 decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build a rank-16 decomposition by viewing C = A (2x3) times B (3x6)
    # B is split into three 3x2 blocks: B = [B1 B2 B3]
    # C = [A B1, A B2, A B3] where each A Bk is 2x2.
    # We use a rank-5 decomposition for 2x2x2 (Strassen) for each block.
    # Total rank = 3 * 5 = 15. But we need to check if we can do better.
    # Actually, the known optimal rank for 2x3x6 is not known to be 15.
    # Let's try to implement the Strassen-based rank-15 decomposition.
    #
    # Strassen 2x2x2 decomposition (rank 7):
    # M1 = (A11 - A21)(B21 + B22)
    # M2 = (A21 + A22)(B11 - B21)
    # M3 = (A11 + A22)(B11 + B22)
    # M4 = A11(B12 - B22)
    # M5 = A22(B21 - B11)
    # M6 = (A11 + A12)(B22 - M2's B part?) 
    # Actually, standard Strassen:
    # M1 = (A00 - A10)(B01 + B11)
    # M2 = (A10 + A11)(B00 - B10)
    # M3 = (A00 + A11)(B00 + B11)
    # M4 = A00(B01 - B11)
    # M5 = A11(B10 - B00)
    # M6 = (A00 + A01)(B11 - M2's?) 
    # Let's use a verified Strassen formula:
    # M1 = (A00 - A10)(B01 + B11)
    # M2 = (A10 + A11)(B00 - B10)
    # M3 = (A00 + A11)(B00 + B11)
    # M4 = A00(B01 - B11)
    # M5 = A11(B10 - B00)
    # M6 = (A00 + A01)(B10 - B00)
    # M7 = (A10 + A11)(B01 - B11)
    # C00 = M3 + M1 - M5 + M7
    # C01 = M4 + M3
    # C10 = M2 + M5
    # C11 = M3 - M1 + M2 + M4
    #
    # For 2x3x2, we can use a rank-5 or rank-6 decomposition.
    # Let's use a known rank-6 decomposition for 2x3x2 (which is just two 2x2 multiplications sharing A).
    # Actually, 2x3x2 tensor has rank 6 (trivial) but maybe 5?
    # Let's just use the trivial rank-6 for each 2x2x2 block to get rank 18, which is worse than 36? No, 3 blocks * 6 = 18 < 36.
    # So rank 18 is better than 36.
    #
    # Let's implement rank 18: three independent 2x2x2 trivial decompositions.
    # Each 2x2x2 trivial has rank 4. 3 * 4 = 12.
    # Wait, 2x2x2 trivial rank is 4. So 3 blocks * 4 = 12.
    # That's much better than 36.
    #
    # Let's implement rank 12.
    rank = 12
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # A is 2x3: rows i=0,1; cols j=0,1,2
    # B is 3x6: rows j=0,1,2; cols k=0..5
    # C is 2x6: rows i=0,1; cols k=0..5
    # B = [B1 B2 B3] where Bt is 3x2, t=0,1,2. Bt columns are 2t, 2t+1.
    # Ct = A * Bt, which is 2x2.
    # For each t, we do a 2x2x2 multiplication: A(2x3) * Bt(3x2) -> Ct(2x2).
    # Wait, A is 2x3, Bt is 3x2. The multiplication is sum over j of A[i,j]*Bt[j,k'].
    # This is a 2x3x2 tensor, not 2x2x2.
    # The trivial decomposition for 2x3x2 has rank 2*3*2 = 12.
    # So if I do this independently for each t, I get rank 12 * 3 = 36? No.
    # The tensor is C[i, k] = sum_j A[i,j] B[j, k].
    # k = 2t + k', where k' is 0 or 1.
    # C[i, 2t+k'] = sum_{j=0}^{2} A[i,j] B[j, 2t+k'].
    # This is a sum of 3 terms for each (i, 2t+k').
    # Total terms = 2 * 6 * 3 = 36.
    #
    # To reduce rank, we need to share terms across t or k'.
    # If we use Strassen for 2x2x2, we assume A is 2x2. But A is 2x3.
    # We can split A into A = [A_left, A_right] where A_left is 2x2 (j=0,1) and A_right is 2x1 (j=2).
    # Bt = [Bt_left; Bt_right] where Bt_left is 2x2 (j=0,1) and Bt_right is 1x2 (j=2).
    # Ct = A_left * Bt_left + A_right * Bt_right.
    # A_right * Bt_right is 2x1 times 1x2 = 2x2, rank 2 (trivial).
    # A_left * Bt_left is 2x2 times 2x2 = 2x2, rank 7 (Strassen) or 4 (trivial).
    # If we use trivial for both: rank 2 + 4 = 6 per block. 3 blocks * 6 = 18.
    # If we use Strassen for the 2x2 part: rank 2 + 7 = 9 per block. 3 blocks * 9 = 27. Worse.
    # So trivial is better: rank 18.
    #
    # Can we share across blocks?
    # A_right is the same for all t. Bt_right depends on t.
    # A_right * Bt_right = sum_{i,k'} A_right[i] Bt_right[k'] * (e_i e_k'^T).
    # This is a rank-2 term for each t. Since A_right is shared, maybe we can combine?
    # But Bt_right is different for each t, so the terms are linearly independent.
    # So rank 18 seems hard to beat with this simple split.
    #
    # Let's implement rank 18.
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(3):  # t = 0, 1, 2 corresponds to the 3 blocks of B (cols 0-1, 2-3, 4-5)
        # Part 1: A_right (row 2 of A, which is index j=2) * Bt_right (row 2 of B, which is j=2)
        # A_right[i] = A[i, 2] for i=0,1
        # Bt_right[k'] = B[2, 2t+k'] for k'=0,1
        # Term: A[i,2] * B[2, 2t+k']
        for i in range(2):
            for kp in range(2):
                U[i * 3 + 2][col] = 1.0
                V[2 * 6 + (2 * t + kp)][col] = 1.0
                W[(2 * t + kp) * 2 + i][col] = 1.0
                col += 1
        
        # Part 2: A_left (j=0,1) * Bt_left (j=0,1)
        # Trivial 2x2x2: sum over j=0,1 of A[i,j] B[j, 2t+k']
        for i in range(2):
            for j in range(2):
                for kp in range(2):
                    U[i * 3 + j][col] = 1.0
                    V[j * 6 + (2 * t + kp)][col] = 1.0
                    W[(2 * t + kp) * 2 + i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
