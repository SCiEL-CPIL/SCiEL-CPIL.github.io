import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # View 2x4 * 4x8 as:
    # A = [[A11, A12], [A21, A22]] where each block is 1x4 (row vector)
    # B = [[B11, B12, B13, B14],
    #      [B21, B22, B23, B24],
    #      [B31, B32, B33, B34],
    #      [B41, B42, B43, B44]]
    # 
    # Actually, let's think of it as 2x2 matrix multiplication over the ring of 1x4 * 4x8 matrices?
    # No. Let's think of A as 2x4, B as 4x8.
    # Split A into columns: A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    # Split B into rows: B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8
    # Then C = A_left * B_top + A_right * B_bottom
    # A_left is 2x2, B_top is 2x8. A_right is 2x2, B_bottom is 2x8.
    # 
    # So we have two 2x2 * 2x8 multiplications.
    # For 2x2 * 2x8: 
    #   A is 2x2, B is 2x8.
    #   C = A * B, C is 2x8.
    #   We can split B into columns: B = [B1, B2] where B1 is 2x4, B2 is 2x4.
    #   C = A * [B1, B2] = [A*B1, A*B2].
    #   So we need 2x2 * 2x4 twice.
    #   For 2x2 * 2x4:
    #     A is 2x2, B is 2x4.
    #     C = A * B, C is 2x4.
    #     We can use Strassen for the 2x2 part? No, Strassen is for 2x2 * 2x2.
    #     But we can view 2x2 * 2x4 as two 2x2 * 2x2 multiplications? No, that's not right.
    #     2x2 * 2x4: A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    #     C = [[ae+bi, af+bj, ag+bl, ah+bl], [ce+di, cf+dj, cg+dl, ch+dl]]
    #     This is 8 multiplications if done naively for each column.
    #     But we can use a faster method.
    #
    # Let's use the Strassen algorithm for 2x2 * 2x8.
    # Actually, let's just use the schoolbook method but see if we can do better.
    # The schoolbook rank is n*m*p = 2*4*8 = 64.
    # 
    # The current program has a bug. Let's see what rank it produces.
    # It uses a Kronecker product of Strassen (rank 7) and schoolbook (rank 8).
    # So rank_total = 7 * 8 = 56.
    # 
    # Can we do better?
    # Let's try to find a lower rank.
    # 
    # Let's use the fact that 2x4 * 4x8 can be viewed as 2x2 * 2x8 over a ring?
    # No.
    #
    # Let's try a different decomposition.
    # A is 2x4, B is 4x8.
    # Let's split A into 2 blocks of 2 columns each: A = [A1, A2]
    # Let's split B into 2 blocks of 4 rows each: B = [B1; B2]
    # C = A1 * B1 + A2 * B2
    # A1 is 2x2, B1 is 2x8. A2 is 2x2, B2 is 2x8.
    # We need to compute 2x2 * 2x8 twice.
    # For 2x2 * 2x8:
    #   A is 2x2, B is 2x8.
    #   Let's split B into 4 blocks of 2 columns each: B = [B11, B12, B13, B14]
    #   C = A * B = [A*B11, A*B12, A*B13, A*B14]
    #   Each A*Bij is a 2x2 * 2x2 multiplication.
    #   So we need 4 Strassen multiplications, each of rank 7.
    #   Total rank for one 2x2 * 2x8 is 4 * 7 = 28.
    #   For two such multiplications, total rank is 2 * 28 = 56.
    #   This is the same as the current program.
    #
    # Can we do better?
    # Let's try to use a Strassen-like algorithm for 2x2 * 2x8 directly.
    # A = [[a, b], [c, d]], B = [[e, f, g, h, i, j, k, l], [m, n, o, p, q, r, s, t]]
    # C = [[a*e + b*m, a*f + b*n, ..., a*l + b*t], [c*e + d*m, ..., c*l + d*t]]
    # 
    # We can use the fact that 2x2 * 2x8 is equivalent to 2x2 * 2x2 done 4 times, but maybe we can share some computations.
    # Actually, 2x2 * 2x8 can be done in rank 28 (4 * 7).
    # Is there a better way?
    # The rank of 2x2 * 2x8 is at least 14 (by a theorem).
    # But we don't have a known algorithm for rank < 28.
    #
    # Let's stick with rank 56 and fix the current program.
    # The current program has a bug in the Kronecker product permutation.
    # Let's rewrite it correctly.
    #
    # A is 2x4, B is 4x8.
    # U has shape (2*4, rank), V has shape (4*8, rank), W has shape (8*2, rank).
    #
    # Let's use the decomposition:
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x8.
    # C = A1 * B1 + A2 * B2.
    #
    # For A1 * B1:
    #   A1 is 2x2, B1 is 2x8.
    #   We can view this as 4 Strassen multiplications of 2x2 * 2x2.
    #   B1 = [B11, B12, B13, B14] where Bij are 2x2.
    #   C1 = [A1*B11, A1*B12, A1*B13, A1*B14].
    #   Each A1*Bij is a 2x2 * 2x2 multiplication.
    #
    # So we have 8 Strassen multiplications in total (4 for A1*B1, 4 for A2*B2).
    # Total rank = 8 * 7 = 56.
    #
    # Let's construct this.
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen basis for 2x2 * 2x2
    # U_s has shape (4, 7), V_s has shape (4, 7), W_s has shape (4, 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # P1: (a+d)(e+h)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # P2: (c+d)e
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1
    
    # P3: a(f-h)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1
    
    # P4: d(g-e)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    # P5: (a+b)h
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    # P6: (c-a)(e+f)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # P7: (b-d)(g+h)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[3, 6] = -1
    
    # Now, we have 8 Strassen multiplications.
    # For each multiplication, we need to map the 2x2 * 2x2 indices to the 2x4 * 4x8 indices.
    #
    # Let's index the 8 multiplications by (block_a, block_b) where block_a in {0, 1} and block_b in {0, 1, 2, 3}.
    # block_a = 0 corresponds to A1, block_a = 1 corresponds to A2.
    # block_b = 0, 1, 2, 3 corresponds to the 4 blocks of B1 or B2.
    #
    # For a given (block_a, block_b), the 2x2 * 2x2 multiplication is:
    # A_sub = A[block_a * 2 : block_a * 2 + 2, block_b * 2 : block_b * 2 + 2]
    # B_sub = B[block_b * 2 : block_b * 2 + 2, block_b * 2 : block_b * 2 + 2]  # Wait, this is not right.
    #
    # Let's be more careful.
    # A is 2x4. A1 = A[:, 0:2], A2 = A[:, 2:4].
    # B is 4x8. B1 = B[0:2, :], B2 = B[2:4, :].
    #
    # For A1 * B1:
    #   A1 is 2x2, B1 is 2x8.
    #   B1 = [B11, B12, B13, B14] where B1j = B1[:, 2*j : 2*j+2].
    #   So B1j is 2x2.
    #   The product C1 = A1 * B1 = [A1*B11, A1*B12, A1*B13, A1*B14].
    #
    # For a given (block_a, block_b):
    #   block_a = 0: A_sub = A1 = A[:, 0:2]
    #   block_a = 1: A_sub = A2 = A[:, 2:4]
    #   block_b = 0: B_sub = B11 = B[0:2, 0:2]
    #   block_b = 1: B_sub = B12 = B[0:2, 2:4]
    #   block_b = 2: B_sub = B13 = B[0:2, 4:6]
    #   block_b = 3: B_sub = B14 = B[0:2, 6:8]
    #   If block_a = 1, then B_sub = B2j = B[2:4, 2*j : 2*j+2].
    #
    # So for (block_a, block_b):
    #   A_sub = A[:, 2*block_a : 2*block_a+2]
    #   B_sub = B[2*block_a : 2*block_a+2, 2*block_b : 2*block_b+2]
    #   Wait, this is not right.
    #   If block_a = 0, A_sub = A[:, 0:2], B_sub = B[0:2, 2*block_b : 2*block_b+2].
    #   If block_a = 1, A_sub = A[:, 2:4], B_sub = B[2:4, 2*block_b : 2*block_b+2].
    #
    # So the mapping is:
    #   A index i (0 or 1), j (0 or 1) -> A index i, 2*block_a + j
    #   B index i (0 or 1), j (0 or 1) -> B index 2*block_a + i, 2*block_b + j
    #   C index i (0 or 1), j (0 or 1) -> C index i, 2*block_b + j
    #
    # Let's construct U, V, W for each (block_a, block_b) and each Strassen step.
    
    for block_a in range(2):
        for block_b in range(4):
            for s in range(7):
                col = (block_a * 4 + block_b) * 7 + s
                # U
                for i in range(2):
                    for j in range(2):
                        u_idx = i * 4 + (2 * block_a + j)
                        U[u_idx, col] = U_s[i * 2 + j, s]
                # V
                for i in range(2):
                    for j in range(2):
                        v_idx = (2 * block_a + i) * 8 + (2 * block_b + j)
                        V[v_idx, col] = V_s[i * 2 + j, s]
                # W
                for i in range(2):
                    for j in range(2):
                        w_idx = (2 * block_b + j) * 2 + i
                        W[w_idx, col] = W_s[i * 2 + j, s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
