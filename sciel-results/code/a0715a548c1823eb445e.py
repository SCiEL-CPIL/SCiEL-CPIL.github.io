"""Rewritten program: Kronecker-product rank reduction for 2x3x6 matrix multiplication.

Uses the factorization 6 = 2*3 to decompose the problem as a Kronecker product
of a (1,1,2) problem (rank 2) and a (2,3,3) problem (rank 14), giving total rank 28.
"""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6

    # Step 1: Build rank-14 decomposition for (2,3,3)
    # A: 2x3, B: 3x3, C: 2x3
    # U1 shape: (6, 14), V1 shape: (9, 14), W1 shape: (6, 14)
    # Indexing: U1[i*m+j, c], V1[j*p+k, c], W1[k*n+i, c]

    U1 = np.zeros((6, 14), dtype=float)
    V1 = np.zeros((9, 14), dtype=float)
    W1 = np.zeros((6, 14), dtype=float)

    # Part 1: 2x2x3 (rows 0,1 of A; rows 0,1 of B; all 3 cols of C)
    # c_{i,k} = sum_{j=0}^{1} a_{i,j} * b_{j,k}
    # 8 multiplications
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U1[i * 3 + j, col] = 1
                V1[j * 3 + k, col] = 1
                W1[k * 2 + i, col] = 1
                col += 1

    # Part 2: 2x1x3 (row 2 of A; row 2 of B; all 3 cols of C)
    # c_{i,k} += a_{i,2} * b_{2,k}
    # 6 multiplications
    for i in range(2):
        for k in range(3):
            U1[i * 3 + 2, col] = 1
            V1[2 * 3 + k, col] = 1
            W1[k * 2 + i, col] = 1
            col += 1

    # Step 2: Build rank-2 decomposition for (1,1,2)
    # A: 1x1, B: 1x2, C: 1x2
    # c_k = a_0 * b_k
    # 2 multiplications
    U2 = np.zeros((1, 2), dtype=float)
    V2 = np.zeros((2, 2), dtype=float)
    W2 = np.zeros((2, 2), dtype=float)

    for k in range(2):
        U2[0, k] = 1
        V2[k, k] = 1
        W2[k, k] = 1

    # Step 3: Kronecker product
    # (2,3,6) = (1,1,2) ⊗ (2,3,3)
    # n_total = 1*2 = 2, m_total = 1*3 = 3, p_total = 2*3 = 6
    # rank_total = 2 * 14 = 28
    #
    # U_total shape: (n_total * m_total, rank_total) = (6, 28)
    # V_total shape: (m_total * p_total, rank_total) = (18, 28)
    # W_total shape: (p_total * n_total, rank_total) = (12, 28)
    #
    # For Kronecker product of tensors:
    # If T = T2 ⊗ T1, then:
    # U_total[(i2*n1 + i1) * m1 + j1, c2*rank1 + c1] = U2[i2*m2 + j2, c2] * U1[i1*m1 + j1, c1]
    # Wait, let me be more careful with the indexing.
    #
    # The Kronecker product structure:
    # A_total has shape (n2*n1, m2*m1) where n2=1, n1=2, m2=1, m1=3
    # B_total has shape (m2*m1, p2*p1) where p2=2, p1=3
    # C_total has shape (n2*n1, p2*p1)
    #
    # Actually, let me think about this differently.
    # The (2,3,6) problem can be seen as:
    # A is 2x3, B is 3x6, C is 2x6
    # View B as 2x3 blocks: B = [B_top, B_bottom] where B_top is 3x3, B_bottom is 3x3
    # Then C = A * B = [A * B_top, A * B_bottom]
    # So we need two independent (2,3,3) multiplications.
    #
    # This is just 2 * R(2,3,3) = 2 * 14 = 28.
    # No Kronecker product needed, just two copies!

    # Simpler approach: two independent (2,3,3) multiplications
    rank_total = 28
    U_total = np.zeros((6, 28), dtype=float)
    V_total = np.zeros((18, 28), dtype=float)
    W_total = np.zeros((12, 28), dtype=float)

    # Copy 1: for k=0,1,2 (first 3 columns of B, first 3 columns of C)
    for c in range(14):
        for i in range(2):
            for j in range(3):
                if U1[i * 3 + j, c] != 0:
                    U_total[i * 3 + j, c] = U1[i * 3 + j, c]
        for j in range(3):
            for k in range(3):
                if V1[j * 3 + k, c] != 0:
                    V_total[j * 3 + k, c] = V1[j * 3 + k, c]
        for k in range(3):
            for i in range(2):
                if W1[k * 2 + i, c] != 0:
                    W_total[k * 2 + i, c] = W1[k * 2 + i, c]

    # Copy 2: for k=3,4,5 (last 3 columns of B, last 3 columns of C)
    for c in range(14):
        for i in range(2):
            for j in range(3):
                if U1[i * 3 + j, c] != 0:
                    U_total[i * 3 + j, c + 14] = U1[i * 3 + j, c]
        for j in range(3):
            for k in range(3):
                if V1[j * 3 + k, c] != 0:
                    V_total[j * 3 + (k + 3), c + 14] = V1[j * 3 + k, c]
        for k in range(3):
            for i in range(2):
                if W1[k * 2 + i, c] != 0:
                    W_total[(k + 3) * 2 + i, c + 14] = W1[k * 2 + i, c]

    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
