import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition step by step.
    # The goal is rank 128.
    
    # Helper to create a term in the decomposition
    # U, V, W are lists of lists (matrices)
    # We will accumulate terms
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx is index in U (size n*m = 16)
        # v_idx is index in V (size m*p = 36)
        # w_idx is index in W (size p*n = 36)
        # Note: The problem statement says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # Indices:
        # A is n x m. Index in A: i*m + j.
        # B is m x p. Index in B: j*p + k.
        # C is p x n. Index in C: k*n + i. (Trace convention W[k*n+i, r])
        
        U_terms.append((u_idx, coeff))
        V_terms.append((v_idx, coeff))
        W_terms.append((w_idx, coeff))

    # Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices for 2x2 block:
    # A: (i, j) -> i*2 + j
    # B: (j, k) -> j*2 + k
    # C: (k, i) -> k*2 + i
    
    def strassen_2x2(A_off, B_off, C_off):
        # A_off, B_off, C_off are offsets in the global indices
        # A indices: A_off + i*2 + j
        # B indices: B_off + j*2 + k
        # C indices: C_off + k*2 + i
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: (0,0) -> 0. A11: (1,1) -> 3.
        # B00: (0,0) -> 0. B11: (1,1) -> 3.
        # C00: (0,0) -> 0.
        
        # Term 1: A00 B00 C00
        add_term(A_off + 0, B_off + 0, C_off + 0)
        # Term 2: A00 B11 C00
        add_term(A_off + 0, B_off + 3, C_off + 0)
        # Term 3: A11 B00 C00
        add_term(A_off + 3, B_off + 0, C_off + 0)
        # Term 4: A11 B11 C00
        add_term(A_off + 3, B_off + 3, C_off + 0)
        
        # M2 = (A10 + A11) B00
        # A10: (1,0) -> 2. A11: (1,1) -> 3.
        # B00: (0,0) -> 0.
        # C10: (0,1) -> 2.
        add_term(A_off + 2, B_off + 0, C_off + 2)
        add_term(A_off + 3, B_off + 0, C_off + 2)
        
        # M3 = A00 (B10 - B11)
        # A00: (0,0) -> 0.
        # B10: (1,0) -> 2. B11: (1,1) -> 3.
        # C01: (1,0) -> 1.
        add_term(A_off + 0, B_off + 2, C_off + 1)
        add_term(A_off + 0, B_off + 3, C_off + 1, -1)
        
        # M4 = A11 (B01 - B00)
        # A11: (1,1) -> 3.
        # B01: (0,1) -> 1. B00: (0,0) -> 0.
        # C11: (1,1) -> 3.
        add_term(A_off + 3, B_off + 1, C_off + 3)
        add_term(A_off + 3, B_off + 0, C_off + 3, -1)
        
        # M5 = (A00 + A01) B11
        # A00: (0,0) -> 0. A01: (0,1) -> 1.
        # B11: (1,1) -> 3.
        # C00: (0,0) -> 0.
        add_term(A_off + 0, B_off + 3, C_off + 0)
        add_term(A_off + 1, B_off + 3, C_off + 0)
        
        # M6 = (A10 - A00) (B00 + B01)
        # A10: (1,0) -> 2. A00: (0,0) -> 0.
        # B00: (0,0) -> 0. B01: (0,1) -> 1.
        # C11: (1,1) -> 3.
        add_term(A_off + 2, B_off + 0, C_off + 3)
        add_term(A_off + 2, B_off + 1, C_off + 3)
        add_term(A_off + 0, B_off + 0, C_off + 3, -1)
        add_term(A_off + 0, B_off + 1, C_off + 3, -1)
        
        # M7 = (A01 - A11) (B10 + B11)
        # A01: (0,1) -> 1. A11: (1,1) -> 3.
        # B10: (1,0) -> 2. B11: (1,1) -> 3.
        # C00: (0,0) -> 0.
        add_term(A_off + 1, B_off + 2, C_off + 0)
        add_term(A_off + 1, B_off + 3, C_off + 0)
        add_term(A_off + 3, B_off + 2, C_off + 0, -1)
        add_term(A_off + 3, B_off + 3, C_off + 0, -1)

    # Standard 2x2x1 decomposition
    def std_2x2_1(A_off, B_off, C_off):
        # A is 2x2, B is 2x1, C is 2x1.
        # B indices: B_off + j*1 + 0 -> B_off + j.
        # C indices: C_off + k*1 + 0? No, C is 2x1.
        # In the 2x2x2 context, C was 2x2.
        # Here C is a column.
        # Let's map it to the 2x2x2 indices where k=0.
        # C indices: C_off + k*2 + i. For k=0, C_off + i.
        
        # C0 = A00 B0 + A01 B1
        add_term(A_off + 0, B_off + 0, C_off + 0)
        add_term(A_off + 1, B_off + 1, C_off + 0)
        
        # C1 = A10 B0 + A11 B1
        add_term(A_off + 2, B_off + 0, C_off + 1)
        add_term(A_off + 3, B_off + 1, C_off + 1)

    # Construct 2x2x9 decomposition
    # A is 2x2, B is 2x9, C is 2x9.
    def mul_2x2_9(A_off, B_off, C_off):
        # B is 2x9. Split into four 2x2 and one 2x1.
        # B0: cols 0-1. Offset 0.
        # B1: cols 2-3. Offset 2.
        # B2: cols 4-5. Offset 4.
        # B3: cols 6-7. Offset 6.
        # B4: col 8. Offset 8.
        
        # C is 2x9. Split similarly.
        # C0: cols 0-1. Offset 0.
        # C1: cols 2-3. Offset 2.
        # C2: cols 4-5. Offset 4.
        # C3: cols 6-7. Offset 6.
        # C4: col 8. Offset 8.
        
        strassen_2x2(A_off, B_off + 0, C_off + 0)
        strassen_2x2(A_off, B_off + 2, C_off + 2)
        strassen_2x2(A_off, B_off + 4, C_off + 4)
        strassen_2x2(A_off, B_off + 6, C_off + 6)
        std_2x2_1(A_off, B_off + 8, C_off + 8)

    # Construct 4x4x9 decomposition
    # A is 4x4. Split into 2x2 blocks.
    # A00: rows 0-1, cols 0-1. Offset 0.
    # A01: rows 0-1, cols 2-3. Offset 2.
    # A10: rows 2-3, cols 0-1. Offset 4.
    # A11: rows 2-3, cols 2-3. Offset 6.
    
    # B is 4x9. Split into top (2x9) and bot (2x9).
    # B_top: rows 0-1. Offset 0.
    # B_bot: rows 2-3. Offset 9. (Since each row has 9 cols, row 2 starts at 2*9=18? No.)
    # B index is j*p + k. j is row (0-3), k is col (0-8).
    # B_top rows 0,1. Indices 0..8, 9..17.
    # B_bot rows 2,3. Indices 18..26, 27..35.
    
    # C is 4x9. Wait, C is p x n = 9 x 4.
    # C index is k*n + i. k is row (0-8), i is col (0-3).
    # C_top corresponds to A00, A01 multiplying B_top, B_bot?
    # C = A B.
    # C is 4x4 * 4x9 = 4x9.
    # But the tensor output is p x n = 9 x 4.
    # So C is 9x4.
    # C[k, i] = sum_j A[i, j] B[j, k].
    # i is 0-3, k is 0-8.
    # C_top (rows 0-3 of C? No, C is 9x4).
    # Let's check the indices.
    # A is 4x4. i=0..3, j=0..3.
    # B is 4x9. j=0..3, k=0..8.
    # C is 9x4. k=0..8, i=0..3.
    
    # C[i, k] = sum_j A[i, j] B[j, k].
    # We split A into 2x2 blocks.
    # A00: i=0,1; j=0,1.
    # A01: i=0,1; j=2,3.
    # A10: i=2,3; j=0,1.
    # A11: i=2,3; j=2,3.
    
    # We split B into top (j=0,1) and bot (j=2,3).
    # B_top: j=0,1. k=0..8.
    # B_bot: j=2,3. k=0..8.
    
    # C[i, k] for i=0,1:
    # C[i, k] = sum_{j=0,1} A[i, j] B[j, k] + sum_{j=2,3} A[i, j] B[j, k]
    # = (A00 B_top)[i, k] + (A01 B_bot)[i, k]
    
    # C[i, k] for i=2,3:
    # C[i, k] = (A10 B_top)[i, k] + (A11 B_bot)[i, k]
    
    # So we need to compute:
    # X1 = A00 B_top (2x2 * 2x9 -> 2x9)
    # X2 = A01 B_bot (2x2 * 2x9 -> 2x9)
    # X3 = A10 B_top (2x2 * 2x9 -> 2x9)
    # X4 = A11 B_bot (2x2 * 2x9 -> 2x9)
    
    # The output C is 9x4.
    # C[0:2, 0:2] = X1[0:2, 0:2] + X2[0:2, 0:2] ?
    # No, X1 is 2x9. Its rows correspond to i=0,1. Cols to k=0..8.
    # X2 is 2x9. Its rows correspond to i=0,1. Cols to k=0..8.
    # So C[i, k] for i=0,1 is X1[i, k] + X2[i, k].
    # In the 9x4 matrix C, C[k, i].
    # So for i=0,1, we add X1 and X2 to C[:, 0:2].
    
    # Similarly for i=2,3, we add X3 and X4 to C[:, 2:4].
    
    # Offsets:
    # A00: 0. A01: 2. A10: 4. A11: 6.
    # B_top: 0. B_bot: 18. (Row 2 starts at 2*9=18).
    
    # C offsets:
    # C is 9x4.
    # C[:, 0] is k*4 + 0.
    # C[:, 1] is k*4 + 1.
    # C[:, 2] is k*4 + 2.
    # C[:, 3] is k*4 + 3.
    
    # For X1 (A00 B_top), output is 2x9.
    # Rows i=0,1. Cols k=0..8.
    # In C (9x4), this corresponds to C[k, 0] and C[k, 1].
    # So C_offset for X1 is 0.
    # But wait, mul_2x2_9 expects C to be 2x9 contiguous?
    # My mul_2x2_9 uses C_off + k*2 + i.
    # This assumes C is stored as 2x9 with i as col, k as row?
    # No, in mul_2x2_9, I used C_off + k*2 + i.
    # This means C is stored with i as the fastest varying index?
    # If C is 2x9 (rows i, cols k), then C[i, k] = C_off + i*9 + k.
    # But I used C_off + k*2 + i. This is C[k, i].
    # So my mul_2x2_9 produces a 9x2 matrix?
    # Let's re-check strassen_2x2.
    # C indices: C_off + k*2 + i.
    # k is 0,1. i is 0,1.
    # So it produces a 2x2 matrix stored as C[k, i].
    # This matches the global C storage C[k, i].
    
    # So for X1 (A00 B_top), the output is 2x9 in terms of i,k.
    # Stored as C[k, i] for i=0,1.
    # So the offset in the global 9x4 C matrix is 0.
    # The "width" is 2 (for i=0,1).
    # My mul_2x2_9 assumes the C block is 2x9 (i=0,1; k=0..8).
    # And it writes to C[k, i] relative to C_off.
    # So C_off should be 0.
    
    # For X2 (A01 B_bot), output is 2x9 (i=0,1; k=0..8).
    # Stored as C[k, i] for i=0,1.
    # So C_off should be 0.
    
    # For X3 (A10 B_top), output is 2x9 (i=2,3; k=0..8).
    # Stored as C[k, i] for i=2,3.
    # Global C index is k*4 + i.
    # So C_off should be 2.
    
    # For X4 (A11 B_bot), output is 2x9 (i=2,3; k=0..8).
    # Stored as C[k, i] for i=2,3.
    # So C_off should be 2.
    
    mul_2x2_9(0, 0, 0)   # A00 B_top -> C[:, 0:2]
    mul_2x2_9(2, 18, 0)  # A01 B_bot -> C[:, 0:2]
    mul_2x2_9(4, 0, 2)   # A10 B_top -> C[:, 2:4]
    mul_2x2_9(6, 18, 2)  # A11 B_bot -> C[:, 2:4]
    
    # Now construct U, V, W matrices
    rank = len(U_terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r in range(rank):
        u_idx, coeff = U_terms[r]
        v_idx, coeff_v = V_terms[r]
        w_idx, coeff_w = W_terms[r]
        
        # Check consistency
        assert coeff == coeff_v == coeff_w
        
        U[u_idx][r] = coeff
        V[v_idx][r] = coeff
        W[w_idx][r] = coeff
        
    return {"U": U, "V": V, "W": W}
