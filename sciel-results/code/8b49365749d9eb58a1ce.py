"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of factor matrices
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # We construct a rank-28 decomposition.
    # Structure:
    # 1. M_{2,3,2} for k=0,1 (Rank 11)
    #    - M_{2,2,2} for j=0,1 (Rank 7)
    #    - M_{2,1,2} for j=2 (Rank 4)
    # 2. M_{2,3,1} for k=2 (Rank 6)
    # 3. M_{2,3,2} for k=3,4 (Rank 11)
    #    - M_{2,2,2} for j=0,1 (Rank 7)
    #    - M_{2,1,2} for j=2 (Rank 4)
    # Total Rank = 11 + 6 + 11 = 28.

    rank = 28
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][col] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][col] += val
        col += 1

    def add_strassen(i_offset, j_offset, k_offset):
        # i_offset, j_offset, k_offset are 0 for now, but we use local indices 0,1
        # Global i = i_offset + local_i
        # Global j = j_offset + local_j
        # Global k = k_offset + local_k
        
        # P1 = (A00+A11)(B00+B11)
        u1 = {(i_offset+0, j_offset+0): 1, (i_offset+1, j_offset+1): 1}
        v1 = {(j_offset+0, k_offset+0): 1, (j_offset+1, k_offset+1): 1}
        w1 = {(k_offset+0, i_offset+0): 1, (k_offset+1, i_offset+1): 1}
        add_term(u1, v1, w1)

        # P2 = (A10+A11)B00
        u2 = {(i_offset+1, j_offset+0): 1, (i_offset+1, j_offset+1): 1}
        v2 = {(j_offset+0, k_offset+0): 1}
        w2 = {(k_offset+0, i_offset+1): 1}
        add_term(u2, v2, w2)

        # P3 = A00(B10-B11)
        u3 = {(i_offset+0, j_offset+0): 1}
        v3 = {(j_offset+1, k_offset+0): 1, (j_offset+1, k_offset+1): -1}
        w3 = {(k_offset+1, i_offset+0): 1}
        add_term(u3, v3, w3)

        # P4 = A11(B01+B11)
        u4 = {(i_offset+1, j_offset+1): 1}
        v4 = {(j_offset+0, k_offset+1): 1, (j_offset+1, k_offset+1): 1}
        w4 = {(k_offset+0, i_offset+0): -1, (k_offset+1, i_offset+0): 1}
        add_term(u4, v4, w4)

        # P5 = (A00+A01)B11
        u5 = {(i_offset+0, j_offset+0): 1, (i_offset+0, j_offset+1): 1}
        v5 = {(j_offset+1, k_offset+1): 1}
        w5 = {(k_offset+0, i_offset+1): 1, (k_offset+1, i_offset+1): 1}
        add_term(u5, v5, w5)

        # P6 = (A10-A00)(B00+B01)
        u6 = {(i_offset+1, j_offset+0): 1, (i_offset+0, j_offset+0): -1}
        v6 = {(j_offset+0, k_offset+0): 1, (j_offset+0, k_offset+1): 1}
        w6 = {(k_offset+0, i_offset+0): 1}
        add_term(u6, v6, w6)

        # P7 = (A01-A11)(B10+B11)
        u7 = {(i_offset+0, j_offset+1): 1, (i_offset+1, j_offset+1): -1}
        v7 = {(j_offset+1, k_offset+0): 1, (j_offset+1, k_offset+1): 1}
        w7 = {(k_offset+0, i_offset+0): -1}
        add_term(u7, v7, w7)

    # Part 1: k=0,1 (M_{2,3,2})
    # 1a: j=0,1 (M_{2,2,2})
    add_strassen(0, 0, 0)
    # 1b: j=2 (M_{2,1,2})
    # C_{i,k} = A_{i,2} B_{2,k}
    for i in range(2):
        for k in range(2):
            u = {(i, 2): 1}
            v = {(2, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)

    # Part 2: k=2 (M_{2,3,1})
    # C_{i,2} = sum_j A_{i,j} B_{j,2}
    for i in range(2):
        for j in range(3):
            u = {(i, j): 1}
            v = {(j, 2): 1}
            w = {(2, i): 1}
            add_term(u, v, w)

    # Part 3: k=3,4 (M_{2,3,2})
    # 3a: j=0,1 (M_{2,2,2})
    add_strassen(0, 0, 3)
    # 3b: j=2 (M_{2,1,2})
    for i in range(2):
        for k in range(3, 5):
            u = {(i, 2): 1}
            v = {(2, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
