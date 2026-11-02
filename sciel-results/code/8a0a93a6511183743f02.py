import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # rank = 5 * 7 = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # alpha: (i, j) -> coeff
    # beta: (j, k') -> coeff
    # gamma: (k', i) -> coeff
    # Note: W index is k*n + i, so we store gamma[k', i]
    
    # s=1: P1
    # A: (0,0):1, (1,1):1
    # B: (0,0):1, (1,1):1
    # C: (0,0):1, (1,1):1
    # s=2: P2
    # A: (1,0):1, (1,1):1
    # B: (0,0):1
    # C: (1,0):1, (1,1):-1
    # s=3: P3
    # A: (0,0):1
    # B: (0,1):1, (1,1):-1
    # C: (0,1):1, (1,1):1
    # s=4: P4
    # A: (1,1):1
    # B: (1,0):1, (0,0):-1
    # C: (0,0):1, (1,0):1
    # s=5: P5
    # A: (0,0):1, (0,1):1
    # B: (1,1):1
    # C: (0,0):-1, (0,1):1
    # s=6: P6
    # A: (1,0):1, (0,0):-1
    # B: (0,0):1, (0,1):1
    # C: (1,1):1
    # s=7: P7
    # A: (0,1):1, (1,1):-1
    # B: (1,0):1, (1,1):1
    # C: (0,0):1
    
    # List of tuples: (alpha_dict, beta_dict, gamma_dict)
    # alpha_dict keys: (i, j)
    # beta_dict keys: (j, k_prime)
    # gamma_dict keys: (k_prime, i)
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    for b in range(5):
        for s_idx in range(7):
            r = b * 7 + s_idx
            alpha, beta, gamma = strassen_terms[s_idx]
            
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
