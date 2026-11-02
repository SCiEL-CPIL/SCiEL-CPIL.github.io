def construct():
    import numpy as np
    import time

    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 5

    # Target Tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Strategy: Block decomposition
    # 4x4x5 = 2x(2x4x5)
    # 2x4x5 = 2x(2x2x5) + 2x(2x2x5)
    # 2x2x5 = 2x(2x2x2) + 2x(2x2x1)
    # 2x2x2 rank 7 (Strassen)
    # 2x2x1 rank 4 (naive)
    # 2x2x5 rank 7+7+4 = 18
    # 2x4x5 rank 18+18 = 36
    # 4x4x5 rank 36+36 = 72

    # Wait, let's re-verify the block structure.
    # A (4x4), B (4x5), C (5x4)
    # Split A into A00, A01 (top/bottom 2x4? No, 4x4 split into 2x2 blocks)
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B (4x5) split into [[B00, B01], [B10, B11]]?
    # B is 4 rows, 5 cols.
    # Rows split into 2+2. Cols split into 2+2+1.
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # B00, B10 are 2x2.
    # B01, B11 are 2x2.
    # B02, B12 are 2x1.
    
    # C (5x4) is the output.
    # C = [[C00, C01], [C10, C11], [C20, C21]]?
    # C rows correspond to B cols? No.
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i).
    # i in 0..3, j in 0..3, k in 0..4.
    
    # Let's stick to the 72 rank construction which is known to work.
    # Rank 72 = 4 * 18.
    # 4 products of 2x2 * 2x5.
    # Each 2x2 * 2x5 has rank 18.
    # 2x2 * 2x5 = (2x2 * 2x2) + (2x2 * 2x2) + (2x2 * 2x1)
    # Rank = 7 + 7 + 4 = 18.
    
    # We need to map the indices correctly.
    
    # A (4x4) flattened to 16.
    # B (4x5) flattened to 20.
    # C (5x4) flattened to 20.
    
    # A indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B indices (4x5):
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Row 3: 15,16,17,18,19
    
    # B blocks (2x2, 2x2, 2x1) for top 2 rows:
    # B00: 0,1,5,6
    # B01: 2,3,7,8
    # B02: 4,9
    
    # B blocks for bottom 2 rows:
    # B10: 10,11,15,16
    # B11: 12,13,17,18
    # B12: 14,19
    
    # C indices (5x4):
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15
    # Row 4: 16,17,18,19
    
    # C blocks (2x2, 2x2) for top 2 rows (corresponding to B cols 0-1):
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    
    # C blocks for next 2 rows (corresponding to B cols 2-3):
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # C block for last row (corresponding to B col 4):
    # C20: 16,17,18,19 (this is 1x4, but we treat it as 2x2 with zeros? No, 2x1 * 2x2? No.)
    # Wait, the 2x2 * 2x1 product gives a 2x2 matrix.
    # The 2x1 part of B is 2x1. The 2x2 part of A is 2x2.
    # The result is a 2x2 matrix.
    # This result contributes to C rows corresponding to B col 4.
    # B col 4 is the 5th column.
    # In C (5x4), the rows corresponding to B col 4 are... wait.
    # C[k, i] = sum_j A[i, j] B[j, k].
    # If k=4, C[4, i] = sum_j A[i, j] B[j, 4].
    # So C row 4 is the result of multiplying A (4x4) by B[:, 4] (4x1).
    # This is a 4x1 vector.
    # My block decomposition assumed C is 5x4.
    # C row 4 is 1x4.
    # So the 2x2 * 2x1 product (which is 2x2) cannot directly map to C row 4 (1x4).
    
    # Let's re-evaluate the block structure.
    # A (4x4), B (4x5), C (5x4).
    # Split A into 2x2 blocks: A00, A01, A10, A11.
    # Split B into 2x5 blocks: B0 (2x5), B1 (2x5).
    # C = A * B.
    # C is 5x4.
    # C can be split into 5x2 blocks? Or 2x4, 2x4, 1x4?
    # C = [[C00, C01], [C10, C11], [C20, C21]]?
    # C00, C10, C20 are 2x2, 2x2, 1x2.
    # C01, C11, C21 are 2x2, 2x2, 1x2.
    
    # C00 = A00*B00 + A01*B10 (where B00, B10 are 2x2 parts of B0, B1)
    # C01 = A00*B01 + A01*B11 (where B01, B11 are 2x2 parts of B0, B1)
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C20 = A00*B02 + A01*B12 (where B02, B12 are 2x1 parts of B0, B1)
    # C21 = A10*B02 + A11*B12
    
    # Wait, C20 and C21 are 1x2 blocks.
    # But A00*B02 is 2x2.
    # So C20 is not just A00*B02.
    # C[k, i] = sum_j A[i, j] B[j, k].
    # If k=4 (5th col of B), C[4, i] = sum_j A[i, j] B[j, 4].
    # This is a row vector.
    # So C row 4 is 1x4.
    # My block C20, C21 should be 1x2 each.
    # But A00*B02 is 2x2.
    # This means the block decomposition doesn't align simply.
    
    # Let's use the known valid Rank 72 construction.
    # It uses 4 products of 2x2 * 2x5.
    # Product 1: A00 * B0 -> C0 (2x5 part of C?)
    # No, A00 is 2x2. B0 is 2x5. Product is 2x5.
    # C is 5x4.
    # This doesn't match.
    
    # Let's use the Rank 65 construction (49 + 16).
    # 49 for 4x4x4.
    # 16 for 4x4x1.
    # This is reliable.
    
    # I will implement the 49 term Strassen decomposition for 4x4x4.
    # And the 16 term naive decomposition for 4x4x1.
    
    def strassen_2x2():
        terms = []
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        return terms

    s2_terms = strassen_2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    # 4x4x4 decomposition (49 terms)
    # A, B, C are 4x4.
    # Indices 0..15.
    
    # Block offsets for A, B, C (4x4)
    offsets_4x4 = {
        0: [0, 1, 4, 5],   # 00
        1: [2, 3, 6, 7],   # 01
        2: [8, 9, 12, 13], # 10
        3: [10, 11, 14, 15]# 11
    }
    
    # High-level Strassen terms for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix product.
    # We decompose each Mi into 7 rank-1 terms.
    # Total 49 terms.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # For M1:
    # u1 = A00 + A11.
    # v1 = B00 + B11.
    # w1 = C00 + C11.
    
    # We will generate the 49 terms by iterating over the 7 high-level terms
    # and the 7 low-level Strassen terms.
    
    # This is complex to code correctly without verification.
    # I will use the schoolbook 80 for safety, or try to find a valid 65.
    # Given the time, I will return the schoolbook 80.
    # But I want to beat 80.
    
    # I will use the Rank 72 construction if I can verify it.
    # But I can't verify it easily.
    
    # I will return the schoolbook 80.
    
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col_fb] = 1
                V_fb[j * p + k][col_fb] = 1
                W_fb[k * n + i][col_fb] = 1
                col_fb += 1

    return {"U": U_fb, "V": V_fb, "W": W_fb}
