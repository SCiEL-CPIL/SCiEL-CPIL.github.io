import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # rank = 5 * 8 = 40 (Schoolbook for 2x2x2 blocks)
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook coefficients for 2x2x2
    # Each term is (alpha_dict, beta_dict, gamma_dict)
    # alpha_dict keys: (i, j)
    # beta_dict keys: (j, k_prime)
    # gamma_dict keys: (k_prime, i)
    
    schoolbook_terms = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1}, {(0,1):1}),
        ({(0,1):1}, {(1,1):1}, {(0,1):1}),
        ({(1,0):1}, {(0,0):1}, {(1,0):1}),
        ({(1,1):1}, {(1,0):1}, {(1,0):1}),
        ({(1,0):1}, {(0,1):1}, {(1,1):1}),
        ({(1,1):1}, {(1,1):1}, {(1,1):1})
    ]
    
    for b in range(5):
        for s_idx in range(8):
            r = b * 8 + s_idx
            alpha, beta, gamma = schoolbook_terms[s_idx]
            
            for (i, j), val in alpha.items():
                u_idx = i * m + j
                U[u_idx][r] = val
                
            for (j, k_prime), val in beta.items():
                k_global = 2 * b + k_prime
                v_idx = j * p + k_global
                V[v_idx][r] = val
                
            for (k_prime, i), val in gamma.items():
                k_global = 2 * b + k_prime
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
    return {"U": U, "V": V, "W": W}
