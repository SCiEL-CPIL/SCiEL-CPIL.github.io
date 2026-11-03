import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # alpha: 4 coeffs for A (00, 01, 10, 11)
    # beta: 4 coeffs for B (00, 01, 10, 11)
    # gamma: 4 coeffs for C (00, 01, 10, 11)
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for b in range(8):
        g_J = b // 4
        g_K = b % 4
        j_start = 2 * g_J
        k_start = 2 * g_K
        
        for s in range(7):
            c = 7 * b + s
            alpha, beta, gamma = steps[s]
            
            # Fill U
            for i_prime in range(2):
                for j_prime in range(2):
                    j_global = j_start + j_prime
                    U[i_prime * m + j_global][c] = alpha[i_prime * 2 + j_prime]
            
            # Fill V
            for j_prime in range(2):
                for k_prime in range(2):
                    j_global = j_start + j_prime
                    k_global = k_start + k_prime
                    V[j_global * p + k_global][c] = beta[j_prime * 2 + k_prime]
            
            # Fill W
            for i_prime in range(2):
                for k_prime in range(2):
                    k_global = k_start + k_prime
                    W[k_global * n + i_prime][c] = gamma[i_prime * 2 + k_prime]
                    
    return {"U": U, "V": V, "W": W}
