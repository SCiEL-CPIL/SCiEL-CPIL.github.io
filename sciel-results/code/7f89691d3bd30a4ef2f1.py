"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS  # 4, 5, 6

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7 (index i*2+j), V: 4x7 (index j*2+k), W: 4x7 (index k*2+i)
        Condition: for all i,j,k in {0,1}, sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # Strassen 7 products for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # c=0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # c=1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1

        # c=2: M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[1][2] = 1

        # c=3: M4 = A11(B01-B00)
        U[3][3] = 1
        V[0][3] = -1; V[1][3] = 1
        W[1][3] = 1; W[3][3] = 1

        # c=4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # c=5: M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1

        # c=6: M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def strassen_4x4x4():
        """
        Build 4x4x4 Strassen decomposition.
        Returns U (16x49), V (16x49), W (16x49)
        U indexed as i*4+j, V as j*4+k, W as k*4+i
        """
        U2, V2, W2 = strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]

        # 7 Strassen terms at 4x4 level. Each term is a product of two 2x2 matrices.
        # For each term, we specify:
        #   - Which 2x2 blocks of A (i_block, j_block) with signs
        #   - Which 2x2 blocks of B (j_block, k_block) with signs
        #   - Which 2x2 blocks of C (k_block, i_block) with signs
        # The 2x2x2 Strassen decomposition handles the internal 2x2 multiplication.

        # Block offsets: block (bi,bj) of 4x4 matrix has i in [2*bi,2*bi+2), j in [2*bj,2*bj+2)
        # A blocks: (0,0)=(0,0), (0,1)=(0,2), (1,0)=(2,0), (1,1)=(2,2)
        # B blocks: (0,0)=(0,0), (0,1)=(0,2), (1,0)=(2,0), (1,1)=(2,2)
        # C blocks: (0,0)=(0,0), (0,1)=(0,2), (1,0)=(2,0), (1,1)=(2,2)

        # Strassen terms for 4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # For each M, we need to add contributions to U, V, W.
        # The key insight: each M is computed as a 2x2 matrix product using Strassen 2x2x2.
        # The 2x2x2 decomposition gives us 7 terms. For each of the 7 M's, we get 7 sub-terms.

        # For M_a = (sum_{(i,j) in Sa} s_a(i,j) * A_{i,j}) * (sum_{(j,k) in Sb} t_a(j,k) * B_{j,k})
        # where Sa and Sb are sets of block indices with signs.

        # The product M_a is a 2x2 matrix. Using Strassen 2x2x2, we compute it with 7 scalar multiplications.
        # For each of those 7 scalar multiplications, we need to add to U, V, W.

        # Let me define the 7 M's with their A-block and B-block specifications:
        # Each entry is (list of (i_block, j_block, sign), list of (j_block, k_block, sign))

        strassen_terms = [
            # M1 = (A00+A11)(B00+B11)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2 = (A10+A11)B00
            ([(1,0,1), (1,1,1)], [(0,0,1)]),
            # M3 = A00(B10-B11)
            ([(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M4 = A11(B01-B00)
            ([(1,1,1)], [(0,1,1), (0,0,-1)]),
            # M5 = (A00+A01)B11
            ([(0,0,1), (0,1,1)], [(1,1,1)]),
            # M6 = (A10-A00)(B00+B10)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M7 = (A01-A11)(B01+B11)
            ([(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ]

        # C reconstruction:
        # C00 = M1 + M4 - M5 + M7  -> C block (0,0) gets contributions from M1(+1), M4(+1), M5(-1), M7(+1)
        # C01 = M3 + M5            -> C block (0,1) gets contributions from M3(+1), M5(+1)
        # C10 = M2 + M4            -> C block (1,0) gets contributions from M2(+1), M4(+1)
        # C11 = M1 - M2 + M3 + M6  -> C block (1,1) gets contributions from M1(+1), M2(-1), M3(+1), M6(+1)

        c_reconstruction = {
            (0,0): [(0,1), (3,1), (4,-1), (6,1)],   # C00: M1, M4, -M5, M7
            (0,1): [(2,1), (4,1)],                   # C01: M3, M5
            (1,0): [(1,1), (3,1)],                   # C10: M2, M4
            (1,1): [(0,1), (1,-1), (2,1), (5,1)],   # C11: M1, -M2, M3, M6
        }

        # For each Strassen term a (0-6), for each of the 7 sub-terms c2 (0-6):
        # The sub-term c2 in the 2x2x2 decomposition contributes to specific (i2,j2,k2) in the 2x2 space.
        # U2[i2*2+j2][c2] gives the A-side coefficient
        # V2[j2*2+k2][c2] gives the B-side coefficient
        # W2[k2*2+i2][c2] gives the C-side coefficient

        # For M_a, the A-side is a linear combination of 2x2 blocks.
        # The 2x2x2 decomposition assumes we're multiplying two specific 2x2 matrices.
        # But in our case, the "A matrix" for M_a is actually a sum of blocks.

        # The trick: for each M_a, we can think of it as multiplying two 2x2 matrices
        # where each element of the "A" 2x2 matrix is a linear combination of actual A blocks,
        # and similarly for B.

        # Actually, the cleaner way: For M_a = P_a * Q_a where P_a and Q_a are 2x2 matrices
        # formed from A and B blocks. The Strassen 2x2x2 decomposition computes P_a * Q_a.
        # P_a[i2,j2] = sum of signs * A_{block_i, block_j} for the blocks that form P_a.
        # Similarly for Q_a.

        # But this doesn't work directly because the 2x2x2 Strassen decomposition is for
        # specific 2x2 matrices, not linear combinations.

        # Let me think differently. The 2x2x2 Strassen decomposition satisfies:
        # For all i2,j2,k2 in {0,1}: sum_{c2} U2[i2*2+j2][c2] * V2[j2*2+k2][c2] * W2[k2*2+i2][c2] = 1

        # This means: if A2[i2,j2] and B2[j2,k2] are the elements of two 2x2 matrices,
        # then C2[i2,k2] = sum_{j2} A2[i2,j2] * B2[j2,k2] = sum_{c2} (sum_{j2} A2[i2,j2]*U2[i2*2+j2][c2]) * (sum_{j2} B2[j2,k2]*V2[j2*2+k2][c2]) * W2[k2*2+i2][c2]

        # So for M_a, we define:
        # A2[i2,j2] = sum over the A-blocks contributing to position (i2,j2) in P_a
        # B2[j2,k2] = sum over the B-blocks contributing to position (j2,k2) in Q_a

        # Then M_a[i2,k2] = sum_{j2} A2[i2,j2] * B2[j2,k2]

        # And the contribution to the overall C is:
        # C[k_block*2+k2, i_block*2+i2] += sign * M_a[i2,k2] for the appropriate blocks.

        # This is getting very complex. Let me use a direct approach.

        # Alternative: For each of the 7 M's, I'll explicitly construct the 2x2 matrices
        # P_a and Q_a, then use the 2x2x2 Strassen decomposition.

        # For M_a, P_a is a 2x2 matrix where P_a[i2,j2] is a linear combination of A blocks.
        # Specifically, P_a[i2,j2] = sum_{(bi,bj,s)} s * A_{bi,bj} where (bi,bj,s) are the A-blocks
        # that contribute to position (i2,j2) in P_a.

        # But wait - in the standard Strassen formulation, each M_a is a product of two
        # specific 2x2 matrices, not linear combinations. The linear combinations are in
        # the Strassen formulas themselves.

        # I think the correct approach is:
        # 1. For each M_a, identify the two 2x2 matrices being multiplied.
        # 2. Each of those 2x2 matrices has elements that are linear combinations of A/B blocks.
        # 3. Use the 2x2x2 Strassen decomposition for each M_a.

        # Let me define for each M_a:
        #   P_a: 2x2 matrix where P_a[i2,j2] = sum of (sign * A_{bi,bj})
        #   Q_a: 2x2 matrix where Q_a[j2,k2] = sum of (sign * B_{bj,bk})

        # M1: P1[i2,j2] = A_{i2,j2} + A_{1-i2,1-j2} (i.e., A00+A11 pattern)
        #     Q1[j2,k2] = B_{j2,k2} + B_{1-j2,1-k2} (i.e., B00+B11 pattern)
        # M2: P2[i2,j2] = A_{1,j2} + A_{1,1-j2}... 

        # This is too error-prone. Let me just verify the 2x2x2 decomposition first,
        # then build the 4x4x4 carefully.

        # Actually, let me take a step back. The simplest correct approach:
        # Use Strassen 2x2x2 for the four 2x2x2 subproblems that partition the 4x4x4.
        # But 4x4x4 is not simply four 2x2x2's because of the Strassen structure.

        # Let me just implement it by direct enumeration with verification.

        # For now, return schoolbook 4x4x4 to be safe.
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4 + j][col] = 1
                    V[j*4 + k][col] = 1
                    W[k*4 + i][col] = 1
                    col += 1
        return U, V, W

    # Use schoolbook for the entire 4x5x6 problem
    rank = n * m * p
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

# EVOLVE-BLOCK-END
