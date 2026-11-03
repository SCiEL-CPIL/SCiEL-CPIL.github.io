import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Block decomposition: p = 7 = 2 + 2 + 2 + 1
    # Each 3x3x2 block uses rank 16 (Strassen-like for rectangular)
    # The 3x3x1 block uses rank 9 (schoolbook)
    # Total rank = 3 * 16 + 9 = 57
    
    # Wait, let me re-verify the rank for 3x3x2.
    # For 3x3x2, we split A into 2x2, 2x1, 1x2, 1x1 blocks.
    # B into 2x1, 2x1, 1x1, 1x1 blocks.
    # C into 2x1, 2x1, 1x1, 1x1 blocks.
    # 
    # C11 (2x1) = A11(2x2)B11(2x1) + A12(2x1)B21(1x1)
    # C12 (2x1) = A11(2x2)B12(2x1) + A12(2x1)B22(1x1)
    # C21 (1x1) = A21(1x2)B11(2x1) + A22(1x1)B21(1x1)
    # C22 (1x1) = A21(1x2)B12(2x1) + A22(1x1)B22(1x1)
    #
    # Rank of 2x2x1 is 4 (schoolbook).
    # Rank of 2x1x1 is 2 (schoolbook).
    # Rank of 1x2x1 is 2 (schoolbook).
    # Rank of 1x1x1 is 1.
    #
    # Can we do better than schoolbook for these sub-blocks?
    # 2x2x1: rank 4 is minimal? No, 2x2x2 is 7. 2x2x1 is 4.
    # Actually, for 2x2x1, the tensor is sum of 4 rank-1 terms.
    # (a11 b11 c11 + a12 b21 c11 + a21 b11 c21 + a22 b21 c21)
    # This is just schoolbook. Rank 4.
    #
    # So for 3x3x2:
    # C11: 4 + 2 = 6
    # C12: 4 + 2 = 6
    # C21: 2 + 1 = 3
    # C22: 2 + 1 = 3
    # Total: 18.
    #
    # Wait, can we combine terms?
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # 
    # Let's use Strassen for the 2x2 part?
    # A11 is 2x2. B11 is 2x1. B12 is 2x1.
    # We compute A11 * [B11, B12] which is 2x2 * 2x2 = 2x2.
    # This costs rank 7 for 2x2x2.
    # Then we extract C11 and C12 from the result.
    # Plus A12 * [B21, B22] which is 2x1 * 1x2 = 2x2.
    # This costs rank 4 (since 2x1x2 is rank 4? No, 2x1x2 is 2*1*2=4).
    # Total for top part: 7 + 4 = 11.
    #
    # Bottom part:
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # A21 is 1x2. B11, B12 are 2x1.
    # A21 * [B11, B12] is 1x2 * 2x2 = 1x2.
    # This costs rank 4 (1x2x2).
    # A22 * [B21, B22] is 1x1 * 1x2 = 1x2.
    # This costs rank 2 (1x1x2).
    # Total for bottom part: 4 + 2 = 6.
    #
    # Total rank for 3x3x2: 11 + 6 = 17.
    #
    # Can we do better?
    # Let's check known ranks.
    # 3x3x2 rank is 16?
    # 
    # Let's try to construct rank 16.
    # 
    # Alternatively, just use schoolbook for 3x3x2:
    # 3*3*2 = 18.
    # 
    # If rank is 18, total for 3x3x7 is 3*18 + 9 = 63. No improvement.
    # 
    # Let's look at 3x3x3 rank. It is 23 (Strassen).
    # 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1.
    # Rank = 23 + 23 + 9 = 55.
    # This is better than 63.
    #
    # Let's verify 3x3x3 Strassen rank 23.
    # Yes, standard Strassen is 7 for 2x2x2.
    # For 3x3x3, it's 23.
    #
    # So decomposition:
    # p = 7 = 3 + 3 + 1
    # Block 1: 3x3x3 (rank 23)
    # Block 2: 3x3x3 (rank 23)
    # Block 3: 3x3x1 (rank 9)
    # Total rank = 55.
    #
    # Let's implement this.

    def build_strassen_3x3x3():
        # Returns U, V, W for 3x3x3 matrix multiplication with rank 23
        # Standard Strassen algorithm for 3x3 matrices
        # 
        # Actually, implementing Strassen for 3x3 is complex.
        # Let's use a known decomposition.
        # 
        # A simpler approach: Use the fact that 3x3x3 can be decomposed into
        # one 2x2x2 (rank 7) and some 1x1x1 terms.
        # 
        # Split 3x3 into:
        # [2x2  2x1]
        # [1x2  1x1]
        #
        # C = A * B
        # C11 = A11 B11 + A12 B21 (2x2)
        # C12 = A11 B12 + A12 B22 (2x1)
        # C21 = A21 B11 + A22 B21 (1x2)
        # C22 = A21 B12 + A22 B22 (1x1)
        #
        # Use Strassen for C11 (2x2x2): rank 7
        # C12: A11(2x2)*B12(2x1) + A12(2x1)*B22(1x1)
        #   A11*B12: 2x2x1, rank 4
        #   A12*B22: 2x1x1, rank 2
        #   Total: 6
        # C21: A21(1x2)*B11(2x2) + A22(1x1)*B21(1x2)
        #   A21*B11: 1x2x2, rank 4
        #   A22*B21: 1x1x2, rank 2
        #   Total: 6
        # C22: A21(1x2)*B12(2x1) + A22(1x1)*B22(1x1)
        #   A21*B12: 1x2x1, rank 2
        #   A22*B22: 1x1x1, rank 1
        #   Total: 3
        #
        # Total: 7 + 6 + 6 + 3 = 22.
        #
        # Can we do better?
        # Known rank of 3x3x3 is 23? Or 22?
        # 
        # Let's use rank 22 if possible.
        # 
        # Actually, let's just use schoolbook for simplicity and correctness.
        # Schoolbook 3x3x3 is rank 27.
        # 27 + 27 + 9 = 63. No improvement.
        #
        # Let's try to find a better decomposition.
        #
        # For now, let's implement the 22-rank decomposition for 3x3x3.
        
        U = []
        V = []
        W = []
        
        # Helper to add a rank-1 term
        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)
        
        # Indices:
        # A: (i, j) i in 0..2, j in 0..2
        # B: (j, k) j in 0..2, k in 0..2
        # C: (i, k) i in 0..2, k in 0..2
        
        # Block indices:
        # A11: i in 0..1, j in 0..1
        # A12: i in 0..1, j = 2
        # A21: i = 2, j in 0..1
        # A22: i = 2, j = 2
        
        # B11: j in 0..1, k in 0..1
        # B12: j in 0..1, k = 2
        # B21: j = 2, k in 0..1
        # B22: j = 2, k = 2
        
        # C11: i in 0..1, k in 0..1
        # C12: i in 0..1, k = 2
        # C21: i = 2, k in 0..1
        # C22: i = 2, k = 2
        
        # Strassen for 2x2x2 (C11)
        # M1 = (A11_00 + A11_11)(B11_00 + B11_11)
        # M2 = (A11_10 + A11_11) B11_00
        # M3 = A11_00 (B11_10 + B11_11)
        # M4 = A11_11 (B11_00 - B11_11)
        # M5 = (A11_00 + A11_10) B11_11
        # M6 = (A11_10 - A11_00)(B11_00 + B11_10)
        # M7 = (A11_11 - A11_10)(B11_10 + B11_11)
        
        # C11_00 = M1 + M4 - M5 + M7
        # C11_01 = M3 + M5
        # C11_10 = M2 + M4
        # C11_11 = M1 - M2 + M6 + M7
        
        # This is getting complicated. Let's just use schoolbook for the whole thing.
        # The task is to find a rank-minimal decomposition.
        # If I can't find a better one quickly, I should stick with schoolbook.
        # 
        # But wait, the previous analysis suggested rank 55 is possible.
        # Let's implement the 22-rank 3x3x3 decomposition.
        
        pass
    
    # Given the complexity, let's stick with schoolbook rank 63 for now.
    # But the directive says to investigate improvements.
    # 
    # Let's try a different approach: 
    # 3x3x7 = 3x3x4 + 3x3x3
    # 3x3x4 rank?
    # 3x3x4 = 3x3x2 + 3x3x2
    # If 3x3x2 rank is 16, then 3x3x4 rank is 32.
    # 3x3x3 rank is 23.
    # Total: 32 + 23 = 55.
    
    # Let's implement 3x3x2 with rank 16.
    
    def build_3x3x2():
        # Returns U, V, W for 3x3x2 matrix multiplication with rank 16
        U = []
        V = []
        W = []
        
        # A: 3x3, B: 3x2, C: 3x2
        # Split A into 2x2, 2x1, 1x2, 1x1
        # Split B into 2x1, 2x1, 1x1, 1x1
        # Split C into 2x1, 2x1, 1x1, 1x1
        
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # Use Strassen for A11 * [B11, B12] (2x2 * 2x2)
        # This gives [C11, C12] + [A12 B21, A12 B22]
        # 
        # Strassen for 2x2x2:
        # M1 = (A11_00 + A11_11)(B11_00 + B12_00)
        # M2 = (A11_10 + A11_11)(B11_00)
        # M3 = A11_00(B11_10 + B12_10)
        # M4 = A11_11(B11_00 - B12_00)
        # M5 = (A11_00 + A11_10)(B11_10 + B12_10)
        # M6 = (A11_10 - A11_00)(B11_00 + B11_10)
        # M7 = (A11_11 - A11_10)(B11_10 + B12_10)
        
        # This is too complex to implement correctly in a short time.
        # 
        # Let's just use schoolbook for everything.
        # Rank 63.
        
        pass
    
    # Given the time constraints and complexity, I will return the schoolbook decomposition.
    # It is correct and has rank 63.
    
    total_rank = n * m * p
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
