import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # M_{3,4,8} = M_{1,4,8} (rank 32) + M_{2,4,8} (rank 56)
    # M_{2,4,8} = 4 * M_{2,2,4} (rank 14 each)
    # M_{2,2,4} = 2 * M_{2,2,2} (rank 7 each)
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Strassen factors for 2x2x2
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # M_{1,4,8}: row i=2, all j in 0..3, all k in 0..7
    # Rank 32: each (j,k) is a separate term
    for j in range(4):
        for k in range(8):
            if time.time() > deadline - 5:
                break
            u_idx = 2 * 4 + j  # i=2, j
            v_idx = j * 8 + k  # j, k
            w_idx = k * 3 + 2  # k, i=2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # M_{2,4,8}: rows i=0,1; all j in 0..3; all k in 0..7
    # Split into 4 blocks of M_{2,2,4}
    for j_block in range(2):  # j offset: 0 or 2
        for k_block in range(2):  # k offset: 0 or 4
            # M_{2,2,4}: split k into two M_{2,2,2}
            for k_sub in range(2):  # k sub-offset: 0 or 2
                # M_{2,2,2}
                for factor in strassen_factors:
                    if time.time() > deadline - 5:
                        break
                    a_coeffs, b_coeffs, c_coeffs = factor
                    for (ai, aj), au in a_coeffs.items():
                        u_idx = (ai) * 4 + (j_block * 2 + aj)
                        U[u_idx][col] += au
                    for (bi, bj), bv in b_coeffs.items():
                        v_idx = (j_block * 2 + bi) * 8 + (k_block * 4 + k_sub * 2 + bj)
                        V[v_idx][col] += bv
                    for (ci, ck), cw in c_coeffs.items():
                        w_idx = (k_block * 4 + k_sub * 2 + ck) * 3 + ci
                        W[w_idx][col] += cw
                    col += 1
    
    # Verify correctness
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # Build the tensor from the decomposition
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    # The matrix multiplication tensor should be all 1s
    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col_s] = 1
                    V_s[j*p+k][col_s] = 1
                    W_s[k*n+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
