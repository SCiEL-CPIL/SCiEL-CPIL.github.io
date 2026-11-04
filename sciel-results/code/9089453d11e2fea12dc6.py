import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A_left_coeffs[s]: list of (i, j, coeff) for local i in {0,1}, j in {0,1}
    # A_right_coeffs[s]: list of (i, j, coeff) for local i in {0,1}, j in {2,3}
    # B_top_coeffs[s]: list of (j, k, coeff) for local j in {0,1}, k in {0,1}
    # B_bot_coeffs[s]: list of (j, k, coeff) for local j in {2,3}, k in {0,1}
    # C_coeffs[s]: list of (i, k, coeff) for local i in {0,1}, k in {0,1}
    
    A_left_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    A_right_coeffs = [
        [(0, 2, 1), (1, 3, 1)],
        [(1, 2, 1), (1, 3, 1)],
        [(0, 2, 1)],
        [(1, 3, 1)],
        [(0, 2, 1), (0, 3, 1)],
        [(1, 2, 1), (0, 2, -1)],
        [(0, 3, 1), (1, 3, -1)]
    ]
    
    B_top_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    B_bot_coeffs = [
        [(2, 0, 1), (3, 1, 1)],
        [(2, 0, 1)],
        [(2, 1, 1), (3, 1, -1)],
        [(3, 0, 1), (2, 0, -1)],
        [(3, 1, 1)],
        [(2, 0, 1), (2, 1, 1)],
        [(3, 0, 1), (3, 1, 1)]
    ]
    
    C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(4):
        # P_t: A_left * B_top
        for s in range(7):
            c_idx = t * 14 + s
            # U
            for i, j, coeff in A_left_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            # V
            for j, k_local, coeff in B_top_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            # W
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff
                
        # Q_t: A_right * B_bot
        for s in range(7):
            c_idx = t * 14 + 7 + s
            # U
            for i, j, coeff in A_right_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            # V
            for j, k_local, coeff in B_bot_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            # W
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
