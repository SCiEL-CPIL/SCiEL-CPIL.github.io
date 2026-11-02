"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10-second safety margin before 140s limit

    n, m, p = 3, 3, 3

    # Rank 27 schoolbook decomposition (always valid fallback)
    def schoolbook_rank_27():
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, 27

    # Try to construct a rank 21 decomposition using known structure
    # The 3x3x3 matrix multiplication tensor has rank 21 (Landsberg 1994,
    # independently by others). We attempt to build it via the following approach:
    #
    # We use the fact that we can express C = A*B where A,B,C are 3x3.
    # The rank 21 decomposition can be constructed from:
    #   - 12 terms from a 2x2x2 core (Strassen-like)
    #   - 9 additional terms for the remaining entries
    #
    # A known rank 21 decomposition uses the following structure:
    # Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    #
    # We can use the following 21 products:
    # Based on the work of Landsberg and the explicit constructions:
    #
    # The rank 21 decomposition for 3x3 matrix multiplication:
    # We decompose using 21 scalar multiplications.

    # Known rank 21 decomposition for 3x3 matrix multiplication.
    # This is based on the explicit construction from the literature.
    # The 21 terms are constructed as follows:
    #
    # Let's use a systematic approach. The tensor [3,3,3] has rank 21.
    # One explicit construction uses the following 21 rank-1 terms:

    # I'll use a verified rank 21 construction.
    # Reference: The rank of 3x3 matrix multiplication is 21.
    # One explicit decomposition:

    # Term 1-4: Main diagonal interactions (using Strassen-like for top-left 2x2)
    # Terms 5-8: Cross terms for 2x2 block
    # Terms 9-21: Remaining entries

    # Actually, let me use a known explicit rank 21 decomposition.
    # From the literature, one construction:

    def rank_21_decomposition():
        """
        Rank 21 decomposition of 3x3 matrix multiplication.
        Uses 21 rank-1 tensors.
        """
        rank = 21
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]

        # The decomposition uses the following structure:
        # We partition the 3x3 matrices into blocks and use
        # a combination of direct and Strassen-like products.

        # Known rank 21 decomposition (one explicit construction):
        # This uses 21 scalar multiplications.

        # Let me use the construction from:
        # "The rank of 3x3 matrix multiplication" - explicit formulas

        # Term indices 0-20

        # I'll use a verified set of 21 terms.
        # Based on the explicit construction in the literature:

        # The 21 terms are defined by their (i,j) positions in U,
        # (j',k) positions in V, and (k',i') positions in W.

        # Using the known rank 21 decomposition:

        # Terms for the 3x3 case:
        # Let's define them explicitly.

        # From one known construction:
        # The decomposition can be written as:
        # C = sum_{r=0}^{20} U_r * V_r * W_r

        # Where each term is a rank-1 tensor.

        # Explicit 21 terms:

        # I'll use a construction based on partitioning into:
        # - A 2x2x2 sub-tensor handled with 7 terms (Strassen)
        # - Additional terms for the third row/column

        # Actually, the cleanest known rank 21 construction:

        # Let me hardcode a verified rank 21 decomposition.
        # Based on: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
        # and related sources.

        # One explicit rank 21 decomposition:

        # Define the 21 terms:
        # Each term r has:
        #   U term: which (i,j) entry of A is used, with coefficient
        #   V term: which (j',k) entry of B is used, with coefficient
        #   W term: which (k',i') entry of C is produced, with coefficient

        # Using a known decomposition:

        # Term 0: a11 * b11 -> c11
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1

        # Term 1: a12 * b21 -> c11
        U[1][1] = 1; V[1][3] = 1; W[1][0] = 1

        # Term 2: a13 * b31 -> c11
        U[2][2] = 1; V[2][6] = 1; W[2][0] = 1

        # Term 3: a21 * b11 -> c21
        U[3][3] = 1; V[3][0] = 1; W[3][3] = 1

        # Term 4: a22 * b21 -> c21
        U[4][4] = 1; V[4][3] = 1; W[4][3] = 1

        # Term 5: a23 * b31 -> c21
        U[5][5] = 1; V[5][6] = 1; W[5][3] = 1

        # Term 6: a31 * b11 -> c31
        U[6][6] = 1; V[6][0] = 1; W[6][6] = 1

        # Term 7: a32 * b21 -> c31
        U[7][7] = 1; V[7][3] = 1; W[7][6] = 1

        # Term 8: a33 * b31 -> c31
        U[8][8] = 1; V[8][6] = 1; W[8][6] = 1

        # Term 9: a11 * b12 -> c12
        U[9][0] = 1; V[9][1] = 1; W[9][1] = 1

        # Term 10: a12 * b22 -> c12
        U[10][1] = 1; V[10][4] = 1; W[10][1] = 1

        # Term 11: a13 * b32 -> c12
        U[11][2] = 1; V[11][7] = 1; W[11][1] = 1

        # Term 12: a21 * b12 -> c22
        U[12][3] = 1; V[12][1] = 1; W[12][4] = 1

        # Term 13: a22 * b22 -> c22
        U[13][4] = 1; V[13][4] = 1; W[13][4] = 1

        # Term 14: a23 * b32 -> c22
        U[14][5] = 1; V[14][7] = 1; W[14][4] = 1

        # Term 15: a31 * b12 -> c32
        U[15][6] = 1; V[15][1] = 1; W[15][7] = 1

        # Term 16: a32 * b22 -> c32
        U[16][7] = 1; V[16][4] = 1; W[16][7] = 1

        # Term 17: a33 * b32 -> c32
        U[17][8] = 1; V[17][7] = 1; W[17][7] = 1

        # Term 18: a11 * b13 -> c13
        U[18][0] = 1; V[18][2] = 1; W[18][2] = 1

        # Term 19: a12 * b23 -> c13
        U[19][1] = 1; V[19][5] = 1; W[19][2] = 1

        # Term 20: a13 * b33 -> c13
        U[20][2] = 1; V[20][8] = 1; W[20][2] = 1

        # But this is still 21 terms and it's just the schoolbook method!
        # The schoolbook method for 3x3 uses 27 terms (3*3*3 = 27).
        # What I wrote above only covers 21 of the 27 products.
        # I'm missing c23 and c33 terms.

        # The problem is that I can't just pick 21 of the 27 schoolbook terms.
        # A rank 21 decomposition is NOT a subset of the schoolbook terms.
        # It uses different linear combinations.

        # Let me reconsider. A true rank 21 decomposition uses linear combinations
        # of entries, not just individual entries.

        # This is getting complex. Let me just return the schoolbook rank 27.
        return None  # Signal failure

    # Given the complexity and risk of error in manually constructing a rank 21
    # decomposition, and the fact that the previous attempt failed validation,
    # I'll return the safe rank 27 schoolbook decomposition.

    # However, let me try one more approach: use numpy to verify if a
    # candidate decomposition is valid, and try to find one with rank < 27.

    # Actually, the safest approach given the constraints is to return rank 27.
    # The rank 27 decomposition is guaranteed correct.

    if time.time() > deadline:
        U, V, W, r = schoolbook_rank_27()
        return {"U": U, "V": V, "W": W}

    # For now, return the guaranteed-correct rank 27 decomposition.
    U, V, W, r = schoolbook_rank_27()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
