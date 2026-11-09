"""
Rewritten program for 2x3x8 matrix multiplication rank minimization.
Uses Strassen decomposition for 2x2x2 blocks and outer products for 2x1x2 blocks.
Total rank 44.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if the tensor decomposition is exact.
    U: (nm, R)
    V: (mp, R)
    W: (pn, R)
    T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0.
    """
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    T_approx = np.zeros_like(T)
    for r in range(rank):
        T_approx += U_np[:, r, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
                    
    return np.allclose(T_approx, T, atol=1e-5)

def construct():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            if c != 0:
                U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            if c != 0:
                V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            if c != 0:
                W[idx][col] += c

    current_col = 0
    
    # Split p=8 into two blocks of 4.
    # Each block is 2x3x4.
    # Split m=3 into 2x2 part (j=0,1) and 1x2 part? No, 2x2 part is j=0,1 with k=0,1 (2x2x2) and j=0,1 with k=2,3?
    # Wait, the previous logic split k into pairs.
    # Let's stick to the 4 blocks of 2x2x2 + 2x1x2?
    # No, the previous logic was:
    # for k_block in range(4): # k=0,1; 2,3; 4,5; 6,7
    #   Part 1: j in 0,1 (2x2x2) using Strassen
    #   Part 2: j = 2 (2x1x2) using Outer Product
    # This covers j=0,1 for all k, and j=2 for all k.
    # This is equivalent to splitting m into 2+1 and p into 4 pairs.
    # Rank: 4 * (7 + 4) = 44.
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # Part 1: Strassen 2x2x2 for j=0,1
        # A indices: a=0, b=1, c=3, d=4
        # B indices: e=0*8+k0, f=0*8+k1, g=1*8+k0, h=1*8+k1
        # W indices: w00=k0*2+0, w01=k1*2+0, w10=k0*2+1, w11=k1*2+1
        
        a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
        e_idx, f_idx = 0 * p + k0, 0 * p + k1
        g_idx, h_idx = 1 * p + k0, 1 * p + k1
        w_00, w_01 = k0 * n, k1 * n
        w_10, w_11 = k0 * n + 1, k1 * n + 1
        
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: Outer Product 2x1x2 for j=2
        # A indices: a2=2, c2=5
        # B indices: e2=2*8+k0, f2=2*8+k1
        # W indices: w00, w01, w10, w11
        
        a2_idx, c2_idx = 2, 5
        e2_idx, f2_idx = 2 * p + k0, 2 * p + k1
        
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The logic above is already in the global scope or I can wrap it.
# The prompt requires keeping function names.
# I will place the construct function as is.
# EVOLVE-BLOCK-END

BET: {mechanism: "No change made, maintaining the current optimal decomposition within the allowed design family", q: 0.0, delta: [0.0, 0.0]}
