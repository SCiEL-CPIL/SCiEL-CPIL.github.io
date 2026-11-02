import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block size 3
    b = 3
    # Number of blocks 2x2
    nb = 2
    
    # Strassen coefficients for 2x2 blocks
    # X_r is list of (I, J, coeff)
    # Y_r is list of (J, K, coeff)
    # W_r is dict (I, K) -> coeff
    
    strassen = [
        # r=1
        {
            'X': [(0, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1), (1, 1, 1)],
            'W': {(0, 0): 1, (0, 1): -1, (1, 0): -1, (1, 1): 1}
        },
        # r=2
        {
            'X': [(1, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1)],
            'W': {(0, 0): -1, (1, 0): 1}
        },
        # r=3
        {
            'X': [(0, 0, 1)],
            'Y': [(0, 1, 1), (1, 1, -1)],
            'W': {(0, 1): 1, (1, 1): -1}
        },
        # r=4
        {
            'X': [(1, 1, 1)],
            'Y': [(1, 0, 1), (0, 0, -1)],
            'W': {(1, 0): 1, (0, 0): -1}
        },
        # r=5
        {
            'X': [(0, 0, 1), (0, 1, 1)],
            'Y': [(1, 1, 1)],
            'W': {(0, 1): 1, (1, 1): 1}
        },
        # r=6
        {
            'X': [(1, 0, 1), (0, 0, -1)],
            'Y': [(0, 0, 1), (0, 1, 1)],
            'W': {(0, 0): 1, (0, 1): 1}
        },
        # r=7
        {
            'X': [(0, 1, 1), (1, 1, -1)],
            'Y': [(1, 0, 1), (1, 1, 1)],
            'W': {(1, 0): 1, (1, 1): 1}
        }
    ]
    
    rank = 7 * b * b * b
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for r_data in strassen:
        X_blocks = r_data['X']
        Y_blocks = r_data['Y']
        W_coeffs = r_data['W']
        
        for ip in range(b):
            for jp in range(b):
                for kp in range(b):
                    # Update U
                    for I, J, alpha in X_blocks:
                        u_idx = (I * b + ip) * m + (J * b + jp)
                        U[u_idx][c] += alpha
                    # Update V
                    for J, K, beta in Y_blocks:
                        v_idx = (J * b + jp) * p + (K * b + kp)
                        V[v_idx][c] += beta
                    # Update W
                    for (I, K), gamma in W_coeffs.items():
                        w_idx = (K * b + kp) * n + (I * b + ip)
                        W[w_idx][c] += gamma
                    c += 1
                    
    return {"U": U, "V": V, "W": W}
