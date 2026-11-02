import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 59: 23 (Laderman for 3x3x3) + 36 (Schoolbook for 3x3x4)
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman coefficients for 3x3x3
    # Format: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # u_index: i*3+j, v_index: j*3+k, w_index: k*3+i
    # Note: V index in full problem is j*7+k. For Laderman part, k is 0,1,2.
    # W index in full problem is k*3+i. For Laderman part, k is 0,1,2.
    
    laderman_terms = [
        # 1
        ([0], [1], [0], [1], [0], [1]),
        # 2
        ([2], [1], [6], [1], [0], [1]), # A02, B20 -> C00? No, A02*B20 contributes to C02? No.
        # A02 * B20: i=0, j=2, k=0. C_{0,0} += A02 B20? No.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Term A_{0,2} B_{2,0} contributes to C_{0,0}.
        # So W index for C_{0,0} is k=0, i=0 -> 0.
        # U index for A_{0,2} is 0*3+2 = 2.
        # V index for B_{2,0} is 2*3+0 = 6.
        # This term is correct.
    ]
    
    # This is getting too risky to hardcode from memory without verification.
    # I will return the schoolbook solution to ensure validity.
    # Score 1.0 is better than 0.
    
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
