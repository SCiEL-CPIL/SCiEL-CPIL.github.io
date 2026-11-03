import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Strassen 2x2 factors
    s2 = [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, 0]])),
        (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, -1]]), np.array([[0, 1], [0, 0]])),
        (np.array([[0, 0], [0, 1]]), np.array([[0, 1], [0, 0]]), np.array([[-1, 1], [0, 0]])),
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 1]])),
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[-1, 0], [0, 0]])),
        (np.array([[0, 1], [0, -1]]), np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]])),
    ]

    # Build rank-1 terms for 4x4 x 4x4 using double Strassen (49 terms)
    def build_4x4x4_terms():
        terms = []
        for Uo, Vo, Wo in s2:
            for Ui, Vi, Wi in s2:
                # Effective coefficients: 2x2 matrices
                Ueff = Ui @ Uo  # 2x2
                Veff = Vi @ Vo  # 2x2
                Weff = Wi @ Wo  # 2x2

                # Ueff[r, l] is coefficient for A block [l, ?]
                # The left operand is a 2x2 matrix of 2x2 blocks:
                # L[c, i, j] = sum_l Ueff[?, l] * A[2*l+i, 2*c+j]
                # Wait, let me be more careful with indexing.

                # A is 4x4, viewed as 2x2 blocks of 2x2:
                # A_block[a, b] = A[2*a:2*a+2, 2*b:2*b+2]

                # Ueff is 2x2, where Ueff[r, l] means:
                # The left operand's block row r gets contribution from A block row l
                # Specifically: X[r, c] = sum_l Ueff[r, l] * A_block[l, c]
                # X[r, c] is a 2x2 block: X[r, c][i, j] = sum_l Ueff[r, l] * A[2*l+i, 2*c+j]

                # So the linear combination for the left side is:
                # sum_{r, c, i, j} Ueff[r, l] * A[2*l+i, 2*c+j]
                # But this is not a single scalar; it's a 2x2 matrix of 2x2 blocks.

                # For the rank-1 decomposition, each scalar multiplication in the innermost level
                # corresponds to one rank-1 term. The innermost level involves multiplying
                # two 2x2 matrices (the "block" products from Strassen).

                # Actually, I realize the correct interpretation:
                # The double Strassen gives 49 terms, where each term is:
                # prod = (sum of some A entries) * (sum of some B entries)
                # and contributes to C with some linear combination.

                # Let me construct the u, v, w vectors directly.

                u = np.zeros(16)
                v = np.zeros(16)
                w = np.zeros(16)

                # For each entry (r, c, i, j) in the left operand (2x2 blocks of 2x2):
                # The coefficient of A[2*l+i, 2*c+j] is Ueff[r, l]
                # But we need a single linear combination, not a 2x2 matrix of 2x2 blocks.

                # I think the issue is that I'm conflating the block structure with scalar structure.
                # Let me just directly enumerate the 49 terms by considering that
                # each (outer, inner) pair gives one "block multiplication" which is
                # a product of two 2x2 matrices, and each such product has 4 scalar multiplications.
                # But 7*7*4 = 196, not 49.

                # I'll just use schoolbook for the 4x4x4 part.
                pass

        return terms

    # For now, use schoolbook for the entire 4x4 x 4x5 multiplication
    U_list = []
    V_list = []
    W_list = []

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)

    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
