import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # We will implement a rank-32 decomposition.
    # Strategy: Split the 4x7 matrix B into two blocks B1 (4x4) and B2 (4x3).
    # C = A1 * B1 + A2 * B2.
    # A1 is rows 0-1 of A (indices 0,1,4,5 in row-major 2x4? No, A is 2x4.
    # Indices in U (size 8):
    # Row 0 of A: indices 0, 1, 2, 3
    # Row 1 of A: indices 4, 5, 6, 7
    #
    # B is 4x7. Indices in V (size 28):
    # Col 0 of B: 0, 4, 8, 12 (rows 0-3) -> indices j*7+k for j in 0..3, k in 0..6?
    # Wait, B is m x p = 4 x 7.
    # V index for B[j,k] is j*p + k.
    # B1 is 4x4 (cols 0-3): indices 0..3, 7..10, 14..17, 21..24.
    # B2 is 4x3 (cols 4-6): indices 4..6, 11..13, 18..20, 25..27.
    #
    # C is 2x7. Indices in W (size 14):
    # C[i,k] index is k*n + i = k*2 + i.
    # C1 (cols 0-3): indices 0,1, 2,3, 4,5, 6,7.
    # C2 (cols 4-6): indices 8,9, 10,11, 12,13.
    
    # We use Strassen for A1*B1 and A2*B1?
    # A1 is 2x4? No, A is 2x4.
    # To use Strassen M_2,2,2, we need 2x2 blocks.
    # A = [A11 A12; A21 A22] where A11 is 1x2, A12 is 1x2, etc.
    # This doesn't fit M_2,2,2 directly which requires 2x2 inputs.
    #
    # Let's try a different split.
    # Treat A as 2x4. B as 4x7.
    # We can view A as two 2x2 blocks? No, it's 2x4.
    # We can view B as two 4x2 blocks and one 4x3?
    #
    # Actually, standard Strassen applies to square matrices.
    # For rectangular, we can pad or use specific algorithms.
    #
    # Let's try a simpler construction:
    # Rank of M_2,4,7.
    # Schoolbook: 56.
    # We can use the fact that R(M_{2,4,7}) <= R(M_{2,4,8}) - something?
    # Or R(M_{2,4,7}) <= R(M_{2,4,4}) + R(M_{2,4,3}).
    # R(M_{2,4,4}) = R(M_{2,4,4}).
    # R(M_{2,4,3}) = R(M_{2,4,3}).
    #
    # Let's calculate R(M_{2,4,4}) and R(M_{2,4,3}) using Strassen-like decomposition.
    # M_{2,4,4}: A(2x4) * B(4x4).
    # Split B into B_left (4x2) and B_right (4x2).
    # C_left = A * B_left (2x2). C_right = A * B_right (2x2).
    # So M_{2,4,4} is equivalent to two M_{2,4,2} multiplications.
    # R(M_{2,4,2})?
    # A(2x4) * B(4x2).
    # Split A into A_left (2x2) and A_right (2x2).
    # C = A_left * B_left + A_right * B_right.
    # This is not quite right. B is 4x2.
    # Let B = [B11; B21] where B11 is 2x2, B21 is 2x2.
    # C = A1 * B11 + A2 * B21? No, A is 2x4.
    # A = [A1 A2] where A1 is 2x2, A2 is 2x2.
    # C = A1 * [B11; B12] + A2 * [B21; B22]? No.
    # B is 4x2. Rows 0-3, Cols 0-1.
    # B = [B_top; B_bottom] where B_top is 2x2, B_bottom is 2x2.
    # C = A1 * B_top + A2 * B_bottom.
    # So M_{2,4,2} is equivalent to two M_{2,2,2} multiplications.
    # R(M_{2,2,2}) = 7 (Strassen).
    # So R(M_{2,4,2}) <= 2 * 7 = 14.
    #
    # Now M_{2,4,4}:
    # C = A * B. B is 4x4.
    # Split B into B_left (4x2) and B_right (4x2).
    # C_left = A * B_left (2x2).
    # C_right = A * B_right (2x2).
    # R(M_{2,4,4}) <= R(M_{2,4,2}) + R(M_{2,4,2}) = 14 + 14 = 28.
    #
    # Now M_{2,4,3}:
    # C = A * B. B is 4x3.
    # Split B into B_left (4x2) and B_right (4x1).
    # C_left = A * B_left (2x2).
    # C_right = A * B_right (2x1).
    # R(M_{2,4,2}) <= 14.
    # R(M_{2,4,1})?
    # A(2x4) * B(4x1).
    # B is a column vector.
    # C = A1 * B_top + A2 * B_bottom.
    # This is sum of two dot products?
    # A1 (2x2) * B_top (2x1) -> 2x1 vector.
    # A2 (2x2) * B_bottom (2x1) -> 2x1 vector.
    # Each M_{2,2,1} has rank 4 (schoolbook) or can it be lower?
    # M_{2,2,1} is just two dot products of length 2.
    # Rank is 4.
    # So R(M_{2,4,1}) <= 4 + 4 = 8.
    #
    # Total Rank for M_{2,4,7}:
    # R(M_{2,4,7}) <= R(M_{2,4,4}) + R(M_{2,4,3})
    #             <= 28 + (14 + 8)
    #             <= 28 + 22
    #             <= 50.
    #
    # This is better than 56.
    # Can we do better for M_{2,4,1}?
    # M_{2,2,1}:
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    # This is 4 multiplications.
    # Can we use Strassen? No, Strassen is for 2x2 output.
    # For 2x1 output, rank 4 is likely minimal.
    #
    # Can we improve M_{2,4,2}?
    # R(M_{2,4,2}) <= 14.
    # Is there a rank 12 or 13 algorithm?
    # Maybe. But 14 is safe with Strassen.
    #
    # Let's implement the rank 50 algorithm.
    #
    # Structure:
    # 1. Block for B1 (4x4): Rank 28.
    #    - Two M_{2,4,2} blocks.
    #    - Each M_{2,4,2} uses two M_{2,2,2} Strassen blocks.
    # 2. Block for B2 (4x3): Rank 22.
    #    - One M_{2,4,2} block (Rank 14).
    #    - One M_{2,4,1} block (Rank 8).
    #      - Two M_{2,2,1} blocks (Rank 4 each).
    
    U = np.zeros((n*m, 50))
    V = np.zeros((m*p, 50))
    W = np.zeros((p*n, 50))
    
    col = 0
    
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: 4 indices for A block (2x2 flattened row-major)
        # b_idx: 4 indices for B block (2x2 flattened row-major)
        # c_idx: 4 indices for C block (2x2 flattened row-major)
        # Strassen M1-M7
        # M1 = (A11 + A22) * (B11 + B22)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2 = A22 * (B12 - B11)
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M3 = (A11 - A22) * B22
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[3]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M4 = A11 * (B21 - B22)
        U[a_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1

    def add_m221(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: 4 indices for A (2x2)
        # b_idx: 2 indices for B (2x1) -> rows 0,1
        # c_idx: 2 indices for C (2x1) -> rows 0,1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # This is 4 terms.
        for i in range(2):
            for j in range(2):
                U[a_idx[i*2+j]][col] += 1
                V[b_idx[j]][col] += 1
                W[c_idx[i]][col] += 1
                col += 1

    # Indices helper
    # A is 2x4. Indices in U:
    # A[i,j] -> i*4 + j
    # B is 4x7. Indices in V:
    # B[j,k] -> j*7 + k
    # C is 2x7. Indices in W:
    # C[i,k] -> k*2 + i
    
    # Block 1: B1 (cols 0-3)
    # B1 is 4x4.
    # Split B1 into B1_left (cols 0-1) and B1_right (cols 2-3).
    # B1_left is 4x2.
    # B1_right is 4x2.
    
    # M_{2,4,2} for B1_left
    # A is full 2x4.
    # B is 4x2 (rows 0-3, cols 0-1).
    # C is 2x2 (rows 0-1, cols 0-1).
    # Split A into A_left (cols 0-1) and A_right (cols 2-3).
    # A_left is 2x2. A_right is 2x2.
    # Split B into B_top (rows 0-1) and B_bottom (rows 2-3).
    # B_top is 2x2. B_bottom is 2x2.
    # C = A_left * B_top + A_right * B_bottom.
    # So two M_{2,2,2} blocks.
    
    # Term 1: A_left * B_top
    # a_idx: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 4, 5
    # b_idx: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 7, 8
    # c_idx: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 2, 3
    add_strassen([0, 1, 4, 5], [0, 1, 7, 8], [0, 1, 2, 3])
    
    # Term 2: A_right * B_bottom
    # a_idx: A[0,2], A[0,3], A[1,2], A[1,3] -> 2, 3, 6, 7
    # b_idx: B[2,0], B[2,1], B[3,0], B[3,1] -> 14, 15, 21, 22
    # c_idx: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 2, 3
    add_strassen([2, 3, 6, 7], [14, 15, 21, 22], [0, 1, 2, 3])
    
    # M_{2,4,2} for B1_right
    # B1_right is 4x2 (rows 0-3, cols 2-3).
    # C is 2x2 (rows 0-1, cols 2-3).
    # Term 3: A_left * B_top_right
    # a_idx: 0, 1, 4, 5
    # b_idx: B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 9, 10
    # c_idx: C[0,2], C[0,3], C[1,2], C[1,3] -> 4, 5, 6, 7
    add_strassen([0, 1, 4, 5], [2, 3, 9, 10], [4, 5, 6, 7])
    
    # Term 4: A_right * B_bottom_right
    # a_idx: 2, 3, 6, 7
    # b_idx: B[2,2], B[2,3], B[3,2], B[3,3] -> 16, 17, 23, 24
    # c_idx: 4, 5, 6, 7
    add_strassen([2, 3, 6, 7], [16, 17, 23, 24], [4, 5, 6, 7])
    
    # Block 2: B2 (cols 4-6)
    # B2 is 4x3.
    # Split into B2_left (cols 4-5) and B2_right (col 6).
    # B2_left is 4x2.
    # B2_right is 4x1.
    
    # M_{2,4,2} for B2_left
    # C is 2x2 (rows 0-1, cols 4-5).
    # Term 5: A_left * B_top_2
    # a_idx: 0, 1, 4, 5
    # b_idx: B[0,4], B[0,5], B[1,4], B[1,5] -> 4, 5, 11, 12
    # c_idx: C[0,4], C[0,5], C[1,4], C[1,5] -> 8, 9, 10, 11
    add_strassen([0, 1, 4, 5], [4, 5, 11, 12], [8, 9, 10, 11])
    
    # Term 6: A_right * B_bottom_2
    # a_idx: 2, 3, 6, 7
    # b_idx: B[2,4], B[2,5], B[3,4], B[3,5] -> 18, 19, 25, 26
    # c_idx: 8, 9, 10, 11
    add_strassen([2, 3, 6, 7], [18, 19, 25, 26], [8, 9, 10, 11])
    
    # M_{2,4,1} for B2_right
    # C is 2x1 (rows 0-1, col 6).
    # Term 7: A_left * B_top_3
    # a_idx: 0, 1, 4, 5
    # b_idx: B[0,6], B[1,6] -> 6, 13
    # c_idx: C[0,6], C[1,6] -> 12, 13
    add_m221([0, 1, 4, 5], [6, 13], [12, 13])
    
    # Term 8: A_right * B_bottom_3
    # a_idx: 2, 3, 6, 7
    # b_idx: B[2,6], B[3,6] -> 20, 27
    # c_idx: 12, 13
    add_m221([2, 3, 6, 7], [20, 27], [12, 13])
    
    # Total columns used: 7 + 7 + 7 + 7 + 7 + 7 + 4 + 4 = 50.
    
    return {"U": U, "V": V, "W": W}
