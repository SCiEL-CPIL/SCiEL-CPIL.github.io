import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0 # seconds
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for a in range(2):
        for b in range(4):
            for t in range(7):
                if time.time() > deadline - 10:
                    break
                x_c, y_c, z_c = strassen_terms[t]
                
                for (i_prime, j_prime), coeff in x_c.items():
                    u_idx = i_prime * m + (2*a + j_prime)
                    U[u_idx][col] += coeff
                
                for (j_prime, k_prime), coeff in y_c.items():
                    v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                    V[v_idx][col] += coeff
                
                for (i_prime, k_prime), coeff in z_c.items():
                    w_idx = (2*b + k_prime) * n + i_prime
                    W[w_idx][col] += coeff
                
                col += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    return {"U": U, "V": V, "W": W}
