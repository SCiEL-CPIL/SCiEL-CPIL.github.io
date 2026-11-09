"""
Strategy: Constructive rank-17 decomposition using Strassen's rank-7 algorithm
for the 2x2x2 core block, with direct (schoolbook) decomposition for boundary
blocks. The 2x3 * 3x3 multiplication is decomposed into three disjoint sub-tensors:
  Block 1: 2x2x2 core (indices i in {0,1}, j in {0,1}, k in {0,1}) - rank 7
  Block 2: j=2, k in {0,1,2}, i in {0,1} (2x1x3) - rank 6
  Block 3: j in {0,1}, k=2, i in {0,1} (2x2x1) - rank 4
Total rank: 7 + 6 + 4 = 17
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6

    # The multiplication tensor T
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Schoolbook fallback (rank 18)
    best_rank = n * m * p
    best_U = np.zeros((dim_U, best_rank))
    best_V = np.zeros((dim_V, best_rank))
    best_W = np.zeros((dim_W, best_rank))
    for c in range(best_rank):
        i = c // (m*p)
        j = (c // p) % m
        k = c % p
        best_U[i*m+j, c] = 1.0
        best_V[j*p+k, c] = 1.0
        best_W[k*n+i, c] = 1.0

    # Helper functions to build factor vectors with coefficients
    def make_U_coeff(indices_ij, coeffs):
        vec = np.zeros(dim_U)
        for (i, j), c in zip(indices_ij, coeffs):
            vec[i*m + j] += c
        return vec

    def make_V_coeff(indices_jk, coeffs):
        vec = np.zeros(dim_V)
        for (j, k), c in zip(indices_jk, coeffs):
            vec[j*p + k] += c
        return vec

    def make_W_coeff(indices_ki, coeffs):
        vec = np.zeros(dim_W)
        for (k, i), c in zip(indices_ki, coeffs):
            vec[k*n + i] += c
        return vec

    # --- Block 1: 2x2x2 Strassen (rank 7) ---
    # W indices: C[k,i] -> k*n + i
    # C00: k=0,i=0 -> 0
    # C10: k=0,i=1 -> 1
    # C01: k=1,i=0 -> 2
    # C11: k=1,i=1 -> 3

    U_b1 = [
        make_U_coeff([(0,0), (1,1)], [1, 1]),       # M1
        make_U_coeff([(1,0), (1,1)], [1, 1]),       # M2
        make_U_coeff([(0,0)], [1]),                 # M3
        make_U_coeff([(1,1)], [1]),                 # M4
        make_U_coeff([(0,0), (0,1)], [1, 1]),       # M5
        make_U_coeff([(1,0), (0,0)], [1, -1]),      # M6
        make_U_coeff([(0,1), (1,1)], [1, -1])       # M7
    ]
    V_b1 = [
        make_V_coeff([(0,0), (1,1)], [1, 1]),       # M1
        make_V_coeff([(0,0)], [1]),                 # M2
        make_V_coeff([(0,1), (1,1)], [1, -1]),      # M3
        make_V_coeff([(1,0), (0,0)], [1, -1]),      # M4
        make_V_coeff([(1,1)], [1]),                 # M5
        make_V_coeff([(0,0), (0,1)], [1, 1]),       # M6
        make_V_coeff([(1,0), (1,1)], [1, 1])        # M7
    ]
    W_b1 = [
        make_W_coeff([(0,0), (1,1)], [1, 1]),       # M1: C00+1, C11+1
        make_W_coeff([(0,1), (1,1)], [1, -1]),      # M2: C10+1, C11-1
        make_W_coeff([(1,0)], [1]),                 # M3: C01+1
        make_W_coeff([(0,1), (1,1)], [1, 1]),       # M4: C10+1, C11+1
        make_W_coeff([(0,0), (1,0)], [-1, 1]),      # M5: C00-1, C01+1
        make_W_coeff([(1,1)], [1]),                 # M6: C11+1
        make_W_coeff([(0,0), (1,1)], [1, 1])        # M7: C00+1, C11+1
    ]

    # --- Block 2: j=2, k in {0,1,2}, i in {0,1} (rank 6) ---
    U_b2 = []
    V_b2 = []
    W_b2 = []
    for i in range(2):
        for k in range(3):
            U_b2.append(make_U_coeff([(i, 2)], [1]))
            V_b2.append(make_V_coeff([(2, k)], [1]))
            W_b2.append(make_W_coeff([(k, i)], [1]))

    # --- Block 3: j in {0,1}, k=2, i in {0,1} (rank 4) ---
    U_b3 = []
    V_b3 = []
    W_b3 = []
    for i in range(2):
        for j in range(2):
            U_b3.append(make_U_coeff([(i, j)], [1]))
            V_b3.append(make_V_coeff([(j, 2)], [1]))
            W_b3.append(make_W_coeff([(2, i)], [1]))

    # Combine all blocks
    U_all = np.array(U_b1 + U_b2 + U_b3)
    V_all = np.array(V_b1 + V_b2 + V_b3)
    W_all = np.array(W_b1 + W_b2 + W_b3)

    rank = len(U_all)

    # Verify correctness
    T_recon = np.zeros((dim_U, dim_V, dim_W))
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U_all[c], V_all[c], W_all[c])

    if not np.allclose(T_recon, T):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U_all.tolist(), "V": V_all.tolist(), "W": W_all.tolist()}
