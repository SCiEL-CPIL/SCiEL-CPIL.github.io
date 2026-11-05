"""Matrix multiplication 6x6x6: Strassen(2x2) x schoolbook(3x3) block decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-level Strassen decomposition:
    - Partition 6x6 matrices into 2x2 blocks of 3x3 matrices.
    - Apply Strassen (rank 7) at the 3x3 block level.
    - Schoolbook at the 3x3 block level (rank 27).
    - Total rank = 27 * 7 = 189.

    Indexing convention:
    U[i*m + j, r] : coefficient of A[i,j] in rank-r term
    V[j*p + k, r] : coefficient of B[j,k] in rank-r term
    W[k*n + i, r] : coefficient of C[i,k] in rank-r term (official trace convention)

    n=m=p=6.
    """
    n, m, p = DIMENSIONS  # all 6
    block_size = 3
    num_blocks = 2

    # Strassen 2x2 block factors
    # s=0: M1 = (A00+A11)*(B00+B11)
    # s=1: M2 = (A10+A11)*B00
    # s=2: M3 = A00*(B01-B11)
    # s=3: M4 = A11*(B10-B00)
    # s=4: M5 = (A00+A01)*B11
    # s=5: M6 = (A10-A00)*(B00+B01)
    # s=6: M7 = (A01-A11)*(B10+B11)

    # Each step defines which blocks contribute to A-factor, B-factor, C-factor
    # A_coeffs: dict from (I,J) to coefficient
    # B_coeffs: dict from (J,K) to coefficient
    # C_coeffs: dict from (I,K) to coefficient

    steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        # M2 = (A10 + A11) * B00
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        # M3 = A00 * (B01 - B11)
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        # M4 = A11 * (B10 - B00)
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        # M5 = (A00 + A01) * B11
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]

    rank = 27 * 7  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for step in steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']

        # Iterate over local indices for the 3x3 blocks
        # u: local row of output and X
        # w: local col of output and Y
        # v: local col of X and row of Y (summed index)
        for u in range(block_size):
            for w in range(block_size):
                for v in range(block_size):
                    # This corresponds to the multiplication X[u,v] * Y[v,w]
                    # which contributes to M[u,w]

                    # Update U (Left factor)
                    # X[u,v] = sum_{I,J} A_coeffs[(I,J)] * A[I*3+u, J*3+v]
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * m + j_global
                            U[idx_u][col] += ca

                    # Update V (Right factor)
                    # Y[v,w] = sum_{J,K} B_coeffs[(J,K)] * B[J*3+v, K*3+w]
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * p + k_global
                            V[idx_v][col] += cb

                    # Update W (Output factor)
                    # M[u,w] contributes to C blocks defined in C_coeffs
                    # C[I,K,u,w] gets contribution with weight C_coeffs[(I,K)]
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            # Task convention: W[k*n+i, r]
                            idx_w = k_global * n + i_global
                            W[idx_w][col] += cc

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
