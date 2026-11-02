import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to construct rank 38 using Laderman's algorithm for 3x3x3 twice
    # Laderman's 19 terms for 3x3x3
    # A indices: 0..8 (00, 01, 02, 10, 11, 12, 20, 21, 22)
    # B indices: 0..8 (00, 01, 02, 10, 11, 12, 20, 21, 22)
    # C indices: 0..8 (00, 01, 02, 10, 11, 12, 20, 21, 22)
    
    # Coefficients for the 19 terms
    # Each term r has u_r (9x1), v_r (9x1), w_r (9x1)
    # u_r[i] is coeff of A_i
    # v_r[j] is coeff of B_j
    # w_r[k] is coeff of C_k
    
    terms = []
    
    # Term 1: a j
    terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 2: b n
    terms.append(([0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # Term 3: c r
    terms.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # Term 4: d j
    terms.append(([0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    # Term 5: e n
    terms.append(([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # Term 6: f r
    terms.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # Term 7: g j
    terms.append(([0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    # Term 8: h n
    terms.append(([0,0,0,0,0,0,0,1,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,1,0]))
    # Term 9: i r
    terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    
    # Term 10: (a+e)(n+r)
    terms.append(([1,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,1], [0,0,0,0,0,0,0,0,0])) # W is not simple sum, need to check reconstruction
    # Actually, W coefficients are determined by the expansion.
    # C = sum w_r M_r
    # M_10 = (a+e)(n+r) = an + ar + en + er
    # This contributes to C_01 (an), C_02 (ar), C_11 (en), C_12 (er)
    # So w_10 should have 1 at 01, 02, 11, 12?
    # But we also have M_2=bn, M_5=en, etc.
    # The W matrix entries are the coefficients of C in the linear combination.
    # C_ik = sum_r w_ik^r M_r
    # For M_10 = (a+e)(n+r):
    # a n -> C_01
    # a r -> C_02
    # e n -> C_11
    # e r -> C_12
    # So w_10 has 1 at indices 1 (C_01), 2 (C_02), 4 (C_11), 5 (C_12).
    terms[-1] = ([1,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,1], [0,1,1,0,1,1,0,0,0])
    
    # Term 11: (b+e)(k+n)
    # b k -> C_00
    # b n -> C_01
    # e k -> C_10
    # e n -> C_11
    terms.append(([0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0], [1,1,0,1,1,0,0,0,0]))
    
    # Term 12: (c+f)(l+o)
    # c l -> C_02
    # c o -> C_05? No, B indices: 00,01,02,10,11,12,20,21,22.
    # l is B_02 (index 2). o is B_12 (index 5).
    # c is A_02 (index 2). f is A_12 (index 5).
    # c l -> C_02
    # c o -> C_05 (C index 5 is C_12? No. C indices: 00,01,02,10,11,12,20,21,22. Index 5 is C_12).
    # Wait, C_ik. i=0, k=2 -> C_02 (index 2). i=0, k=5? No, k is 0..2 for 3x3.
    # B is 3x3. k=0,1,2.
    # B_02 is index 2. B_12 is index 5.
    # c (A_02) * l (B_02) -> C_02 (index 2).
    # c (A_02) * o (B_12) -> C_02? No. A_02 * B_12. Sum over j.
    # C_0k = sum_j A_0j B_jk.
    # A_02 B_2k. So c * B_2k.
    # My term 12 is (c+f)(l+o).
    # c = A_02. f = A_12.
    # l = B_02. o = B_12.
    # c l = A_02 B_02. This is not a valid term in C_0k unless j=0 and k=2?
    # C_02 = A_00 B_02 + A_01 B_12 + A_02 B_22.
    # So A_02 B_02 is not a direct term.
    # Laderman's terms are products of linear combinations.
    # The expansion must match C.
    
    # This manual derivation is error-prone.
    # I will use a known correct set of coefficients if I can recall it, or use a simpler fallback.
    # Given the risk, I will implement the schoolbook decomposition (rank 54) as a safe fallback.
    # However, the prompt asks for a "stronger algorithm".
    # I will try to use the rank 38 construction but verify it.
    
    # If verification fails, return rank 54.
    
    # For now, I will return the schoolbook decomposition to ensure validity, 
    # as I cannot guarantee the Laderman coefficients are correct from memory.
    # But wait, the score 1.0 corresponds to rank 54.
    # I need to beat 1.0.
    # I will try to implement Laderman carefully.
    
    # Let's use the standard 19-term Laderman algorithm coefficients.
    # I will hardcode the 19x9 matrices for U, V, W for one 3x3x3 block.
    
    # U (9x19), V (9x19), W (9x19)
    # Rows of U: 0..8 correspond to A_00..A_22
    # Rows of V: 0..8 correspond to B_00..B_22
    # Rows of W: 0..8 correspond to C_00..C_22
    
    U_lad = np.zeros((9, 19), dtype=int)
    V_lad = np.zeros((9, 19), dtype=int)
    W_lad = np.zeros((9, 19), dtype=int)
    
    # Term 1: a j
    U_lad[0, 0] = 1; V_lad[0, 0] = 1; W_lad[0, 0] = 1
    # Term 2: b n
    U_lad[1, 1] = 1; V_lad[4, 1] = 1; W_lad[4, 1] = 1
    # Term 3: c r
    U_lad[2, 2] = 1; V_lad[8, 2] = 1; W_lad[8, 2] = 1
    # Term 4: d j
    U_lad[3, 3] = 1; V_lad[0, 3] = 1; W_lad[3, 3] = 1
    # Term 5: e n
    U_lad[4, 4] = 1; V_lad[4, 4] = 1; W_lad[4, 4] = 1
    # Term 6: f r
    U_lad[5, 5] = 1; V_lad[8, 5] = 1; W_lad[8, 5] = 1
    # Term 7: g j
    U_lad[6, 6] = 1; V_lad[0, 6] = 1; W_lad[6, 6] = 1
    # Term 8: h n
    U_lad[7, 7] = 1; V_lad[4, 7] = 1; W_lad[7, 7] = 1
    # Term 9: i r
    U_lad[8, 8] = 1; V_lad[8, 8] = 1; W_lad[8, 8] = 1
    
    # Term 10: (a+e)(n+r)
    U_lad[0, 9] = 1; U_lad[4, 9] = 1
    V_lad[4, 9] = 1; V_lad[8, 9] = 1
    W_lad[1, 9] = 1; W_lad[2, 9] = 1; W_lad[4, 9] = 1; W_lad[5, 9] = 1
    # a n -> C_01 (idx 1)
    # a r -> C_02 (idx 2)
    # e n -> C_11 (idx 4)
    # e r -> C_12 (idx 5)
    
    # Term 11: (b+e)(k+n)
    U_lad[1, 10] = 1; U_lad[4, 10] = 1
    V_lad[1, 10] = 1; V_lad[4, 10] = 1
    W_lad[0, 10] = 1; W_lad[1, 10] = 1; W_lad[3, 10] = 1; W_lad[4, 10] = 1
    # b k -> C_00 (idx 0)
    # b n -> C_01 (idx 1)
    # e k -> C_10 (idx 3)
    # e n -> C_11 (idx 4)
    
    # Term 12: (c+f)(l+o)
    U_lad[2, 11] = 1; U_lad[5, 11] = 1
    V_lad[2, 11] = 1; V_lad[5, 11] = 1
    W_lad[2, 11] = 1; W_lad[5, 11] = 1; W_lad[5, 11] = 1; W_lad[8, 11] = 1
    # c l -> C_02 (idx 2)
    # c o -> C_05? No. c=A_02, o=B_12. A_02 B_12 is not a term in C_0k.
    # C_0k = A_00 B_0k + A_01 B_1k + A_02 B_2k.
    # So c (A_02) multiplies B_2k.
    # l = B_02. o = B_12.
    # c l = A_02 B_02. This is not in the sum for C_0k unless j=0? No, j is the middle index.
    # A_02 B_02 implies j=2 for A, j=0 for B. Mismatch.
    # So (c+f)(l+o) is not a valid term for standard matrix multiplication?
    # Laderman's algorithm uses specific linear combinations that DO work.
    # My assumption that l=B_02 and o=B_12 might be wrong for the specific term structure.
    
    # I will stop here and return the schoolbook decomposition to avoid invalid output.
    # The score will be 1.0.
    
    rank = rank_schoolbook
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
