import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    n, m, p = 2, 4, 8

    # Hierarchical ring-based Strassen for 2x4 * 4x8
    # Model as 8x8 multiplication over ring R = M_{2x2}
    # A (2x4) viewed as 2x2 block matrix with 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # B (4x8) viewed as 2x4 block matrix with 2x2 blocks: B = [[B11, B12, B13, B14], [B21, B22, B23, B24]]
    # C (2x8) = AB, with 2x2 blocks: C = [[C11, C12, C13, C14], [C21, C22, C23, C24]]
    #
    # We use Strassen on the "outer" structure. The 2x4 multiplication of A (2x2 blocks) by B (2x4 blocks)
    # can be seen as: for each column-group of B (2 blocks wide), we do a 2x2 * 2x2 block multiplication.
    # There are 4 such column-groups.
    #
    # For each 2x2 * 2x2 block multiplication, Strassen gives 7 ring multiplications.
    # Each ring multiplication is a 2x2 * 2x2 scalar multiplication = rank 7.
    # Total: 4 * 7 * 7 = 196? No, that's wrong.
    #
    # Actually: A is 2x4, split into 2x2 blocks: 2 rows of blocks, 2 cols of blocks.
    # B is 4x8, split into 2x2 blocks: 2 rows of blocks, 4 cols of blocks.
    # C is 2x8, split into 2x2 blocks: 2 rows of blocks, 4 cols of blocks.
    #
    # C[i,j] = sum_k A[i,k] * B[k,j] where i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # So for fixed j: C[:,j] = A * B[:,j] where A is 2x2 (block), B[:,j] is 2x2 (block).
    # This is a 2x2 * 2x2 multiplication for each j, giving 4 independent 2x2*2x2 products.
    # Each 2x2*2x2 has rank 7, total rank = 4*7 = 28? But that's not right either.
    #
    # Wait: A is 2x4 = [A_left | A_right] where each is 2x2.
    # B is 4x8 = [[B_top_left, B_top_right], [B_bot_left, B_bot_right]] where each is 2x2.
    # C = A*B: C[i, :] = A[i, 0:2] * B[0:2, :] + A[i, 2:4] * B[2:4, :]
    # So C is sum of two 2x2 * 2x4 products.
    #
    # Let me just use the direct block approach that gives rank 56.
    # Split A into 2 blocks of 2x2: A0, A1 (each 2x2)
    # Split B into 4 blocks of 2x2: B0, B1, B2, B3 (each 2x2)
    # C = A0*B0 + A1*B1 ... no, that's not right.
    #
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x8: rows 0,1,2,3; cols 0,1,2,3,4,5,6,7
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    #
    # Group k into pairs: (k=0,1) and (k=2,3)
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j] + A[i,3]*B[3,j]
    #
    # For fixed j, define:
    #   A_left = [A[i,0], A[i,1]] (2x1)
    #   A_right = [A[i,2], A[i,3]] (2x1)
    #   B_top = [B[0,j], B[1,j]] (2x1)
    #   B_bot = [B[2,j], B[3,j]] (2x1)
    #   C[i,j] = A_left . B_top + A_right . B_bot
    #
    # This is a sum of two dot products of 2-vectors. Not directly Strassen.
    #
    # Better: treat A as 2x2 matrix [A[0,0] A[0,1]; A[1,0] A[1,1]] and [A[0,2] A[0,3]; A[1,2] A[1,3]]
    # and B as 2x4 matrix with columns being 2-vectors.
    #
    # The standard approach: 8 independent 2x2*2x2 multiplications.
    # Split A into 2 blocks of 2 columns each: A_L (2x2), A_R (2x2)
    # Split B into 4 blocks of 2 columns each: B_0, B_1, B_2, B_3 (each 2x2)
    # But B has 4 rows, so B_0 is 4x2, not 2x2.
    #
    # Correct splitting:
    # A: 2x4 -> split columns into 2 groups of 2: A_0 (2x2), A_1 (2x2)
    # B: 4x8 -> split rows into 2 groups of 2, and columns into 4 groups of 2:
    #   B_{0,0}, B_{0,1}, B_{0,2}, B_{0,3} (each 2x2, from rows 0-1)
    #   B_{1,0}, B_{1,1}, B_{1,2}, B_{1,3} (each 2x2, from rows 2-3)
    # C: 2x8 -> split columns into 4 groups of 2: C_0, C_1, C_2, C_3 (each 2x2)
    #
    # C_j = A_0 * B_{0,j} + A_1 * B_{1,j} for j = 0,1,2,3
    #
    # So we have 4 independent computations of the form: C_j = A_0 * B_{0,j} + A_1 * B_{1,j}
    # where A_0, A_1, B_{0,j}, B_{1,j} are all 2x2 matrices.
    #
    # This is a 2x2 + 2x2 multiplication: we're computing A_0*B_{0,j} + A_1*B_{1,j}
    # which is equivalent to multiplying [A_0 | A_1] (2x4) by [B_{0,j}; B_{1,j}] (4x2).
    #
    # Each such computation has rank at most 7+7=14 (schoolbook) or can be done with Strassen.
    # Actually, [A_0 | A_1] * [B_{0,j}; B_{1,j}] is a 2x4 * 4x2 multiplication.
    # The rank of M_{<2,4,2>} is known to be 7 (same as 2x2x2 by symmetry).
    # So total rank = 4 * 7 = 28? Let me verify.
    #
    # M_{<2,4,2>}: multiply 2x4 by 4x2 to get 2x2.
    # This is equivalent to: given 2x4 matrix X and 4x2 matrix Y, compute XY (2x2).
    # We can write X = [X1 | X2] where X1, X2 are 2x2, Y = [Y1; Y2] where Y1, Y2 are 2x2.
    # XY = X1*Y1 + X2*Y2.
    # This is the sum of two 2x2*2x2 products.
    # Rank of this tensor: we need to compute X1*Y1 + X2*Y2.
    # Using Strassen on each: 7+7=14, but there might be a better way.
    # Actually, this is exactly the "doubling" trick. The rank of M_{<2,4,2>} is 7.
    # Proof: it's isomorphic to M_{<2,2,2>} by a change of variables.
    # Let X' = [X1; X2] (4x2), Y' = [Y1 | Y2] (2x4). Then X'Y' = X1*Y1 + X2*Y2.
    # So M_{<2,4,2>} = M_{<4,2,4>}^T... hmm, not quite.
    #
    # Actually, the tensor for 2x4 * 4x2 -> 2x2 has 2*4*4*2 = 64 input-output pairs.
    # The schoolbook rank is 4 (one for each k). But we can do better.
    # Let me just use the known result: rank(M_{<2,4,2>}) = 7.
    #
    # So total rank = 4 * 7 = 28.
    #
    # Let me construct this. For each j in 0..3:
    #   Compute C_j = A_0 * B_{0,j} + A_1 * B_{1,j} using a rank-7 decomposition.
    #
    # The rank-7 decomposition of 2x4 * 4x2:
    # Let X = [a b; c d] (2x2) = X1, Y = [e f; g h] (2x2) = Y1
    # Let X' = [i j; k l] (2x2) = X2, Y' = [m n; o p] (2x2) = Y2
    # We want XY + X'Y'.
    #
    # Using Strassen-like:
    # P1 = (X+X') * (Y+Y') = XY + XY' + X'Y + X'Y'
    # P2 = X * Y
    # P3 = X' * Y'
    # P4 = (X-X') * (Y-Y') = XY - XY' - X'Y + X'Y'
    # XY + X'Y' = (P1 + P4)/2
    # But P1, P2, P3, P4 each require 7 multiplications, so 28 total. Not good.
    #
    # Better: use the fact that this is a 2x2 matrix over the ring of 2x2 matrices.
    # Or use a direct rank-7 construction.
    #
    # Actually, I recall that rank(M_{<2,4,2>}) = 7 can be achieved by:
    # Identifying 2x4 matrices with 2x2 matrices over a 2-dimensional algebra.
    #
    # Let me just use the 8 independent 2x2*2x2 approach which gives rank 56,
    # and construct it via the hierarchical ring method.

    # Strassen coefficients for 2x2 multiplication
    # U coefficients (for A), V coefficients (for B), W coefficients (for C)
    S = 7
    # Each Strassen step s has:
    # A_combined = sum over (i,j) of U_s[i,j] * A[i,j]
    # B_combined = sum over (k,l) of V_s[k,l] * B[k,l]
    # C += W_s[i,j] * A_combined * B_combined

    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    U_strassen = [
        [1, 0, 0, 1],  # M1: A11+A22
        [0, 0, 1, 1],  # M2: A21+A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A11
        [0, 1, 0, -1], # M7: A12-A22
    ]
    V_strassen = [
        [1, 0, 0, 1],  # M1: B11+B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B21-B22
        [-1, 0, 1, 0], # M4: B12-B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11+B12
        [0, 0, 1, 1],  # M7: B21+B22
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]

    # 8 independent 2x2*2x2 multiplications:
    # A split into 2 column-blocks of width 2: A_0 (cols 0-1), A_1 (cols 2-3)
    # B split into 2 row-blocks of height 2 and 4 col-blocks of width 2:
    #   B_{0,0} (rows 0-1, cols 0-1), B_{0,1} (rows 0-1, cols 2-3), etc.
    #   B_{1,0} (rows 2-3, cols 0-1), B_{1,1} (rows 2-3, cols 2-3), etc.
    # C split into 4 col-blocks of width 2: C_0, C_1, C_2, C_3
    #
    # C_j = A_0 * B_{0,j} + A_1 * B_{1,j}
    #
    # For each j, this is a 2x2*2x2 + 2x2*2x2 computation.
    # We decompose each 2x2*2x2 with Strassen (rank 7).
    # Total: 4 j-values * 2 A-blocks * 7 Strassen = 56.

    num_j = 4      # column groups of C
    num_ablock = 2 # A column blocks
    rank = num_j * num_ablock * S

    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 16 x rank

    r = 0
    for j in range(num_j):
        for ab in range(num_ablock):
            for s in range(S):
                # U: A indices. A is 2x4.
                # A block ab covers columns ab*2 and ab*2+1.
                # A[i, ab*2 + jp] for i in {0,1}, jp in {0,1}
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + ab * 2 + jp
                        U[a_idx][r] = U_strassen[s][i * 2 + jp]

                # V: B indices. B is 4x8.
                # B block: rows ab*2 to ab*2+1, cols j*2 to j*2+1
                # B[ab*2 + jp, j*2 + kp] for jp in {0,1}, kp in {0,1}
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (ab * 2 + jp) * p + (j * 2 + kp)
                        V[b_idx][r] = V_strassen[s][jp * 2 + kp]

                # W: C indices. C is 2x8.
                # C[i, j*2 + kp] for i in {0,1}, kp in {0,1}
                for i in range(n):
                    for kp in range(2):
                        c_idx = (j * 2 + kp) * n + i
                        W[c_idx][r] = W_strassen[i * 2 + kp][s]

                r += 1

    return {"U": U, "V": V, "W": W}


if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    n, m, p = 2, 4, 8
    assert len(U) == n*m
    assert len(U[0]) == len(V[0]) == len(W[0])
    assert len(V) == m*p
    assert len(W) == p*n
    print(f"Rank: {len(U[0])}")
    print("Constructed successfully.")
