import time

def construct():
    n, m, p = 2, 4, 8
    rank = 16
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # 2 blocks of 4 k-values each
    for b in range(2):
        k_base = b * 4
        for k_local in range(4):
            k_global = k_base + k_local
            
            # S terms (4 of them)
            for j in range(4):
                t = 8 * b + 4 * k_local + j
                # U: A[0,j] + A[1,j]
                U[0 * m + j][t] = 1
                U[1 * m + j][t] = 1
                # V: B[j, k_global]
                V[j * p + k_global][t] = 1
                # W: C[0, k_global] + C[1, k_global]
                W[k_global * n + 0][t] = 0.5
                W[k_global * n + 1][t] = 0.5
                
            # D terms (4 of them)
            for j in range(4):
                t = 8 * b + 4 * k_local + 4 + j
                # U: A[0,j] - A[1,j]
                U[0 * m + j][t] = 1
                U[1 * m + j][t] = -1
                # V: B[j, k_global]
                V[j * p + k_global][t] = 1
                # W: C[0, k_global] - C[1, k_global]
                W[k_global * n + 0][t] = 0.5
                W[k_global * n + 1][t] = -0.5
                
    return {"U": U, "V": V, "W": W}
