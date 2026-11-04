import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    def get_laderman_333():
        """
        Laderman's rank-23 algorithm for 3x3x3 matrix multiplication.
        Returns U, V, W as (9, 23) arrays.
        """
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        terms = [
            ([(0,1)], [(0,1)], [(0,1)]),
            ([(4,1)], [(4,1)], [(4,1)]),
            ([(8,1)], [(8,1)], [(8,1)]),
            ([(0,1),(1,1)], [(0,1),(3,-1)], [(0,1),(1,1)]),
            ([(0,1),(2,1)], [(0,1),(6,-1)], [(0,1),(2,1)]),
            ([(4,1),(5,1)], [(4,1),(7,-1)], [(4,1),(5,1)]),
            ([(6,1),(7,1)], [(6,1),(0,-1)], [(6,1),(7,1)]),
            ([(6,1),(8,1)], [(6,1),(3,-1)], [(6,1),(8,1)]),
            ([(0,1),(4,1)], [(1,1),(3,-1)], [(1,1),(4,1)]),
            ([(0,1),(8,1)], [(2,1),(6,-1)], [(2,1),(8,1)]),
            ([(3,1),(8,1)], [(7,1),(3,-1)], [(7,1),(5,1)]),
            ([(4,1),(6,1)], [(5,1),(1,-1)], [(5,1),(7,1)]),
            ([(0,1),(3,1),(6,1)], [(1,1),(4,1),(7,1)], [(1,1),(4,1),(7,1)]),
            ([(0,1),(1,1),(4,1)], [(2,1),(5,1),(8,1)], [(2,1),(5,1),(8,1)]),
            ([(1,1),(4,1),(5,1)], [(0,1),(3,1),(6,1)], [(0,1),(3,1),(6,1)]),
            ([(4,1),(5,1),(8,1)], [(0,1),(1,1),(2,1)], [(0,1),(1,1),(2,1)]),
            ([(0,1),(2,1),(6,1),(8,1)], [(1,1),(5,1),(6,1)], [(1,1),(5,1),(6,1),(7,1)]),
            ([(0,1),(1,1),(4,1),(5,1)], [(2,1),(7,1),(8,1)], [(2,1),(5,1),(7,1),(8,1)]),
            ([(1,1),(2,1),(5,1),(8,1)], [(0,1),(3,1),(4,1)], [(0,1),(3,1),(4,1),(7,1)]),
            ([(0,1),(1,1),(3,1),(4,1),(6,1),(7,1)], [(2,1),(5,1),(8,1)], [(2,1),(5,1),(8,1)]),
            ([(0,1),(2,1),(3,1),(5,1),(6,1),(8,1)], [(1,1),(4,1),(7,1)], [(1,1),(4,1),(7,1)]),
            ([(1,1),(2,1),(4,1),(5,1),(7,1),(8,1)], [(0,1),(3,1),(6,1)], [(0,1),(3,1),(6,1)]),
            ([(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)]),
        ]
        
        for t, (a_terms, b_terms, c_terms) in enumerate(terms):
            for idx, coeff in a_terms:
                U[idx, t] += coeff
            for idx, coeff in b_terms:
                V[idx, t] += coeff
            for idx, coeff in c_terms:
                W[idx, t] += coeff
        
        return U, V, W
    
    def verify_laderman(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = sum(U[i*3+j, t] * V[j*3+k, t] * W[i*3+k, t] for t in range(23))
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    U_l, V_l, W_l = get_laderman_333()
    
    if not verify_laderman(U_l, V_l, W_l):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[:, 0:23] = U_l
    for j in range(3):
        for k in range(3):
            V[j*7+k, 0:23] = V_l[j*3+k]
    for i in range(3):
        for k in range(3):
            W[k*3+i, 0:23] = W_l[i*3+k]
    
    U[:, 23:46] = U_l
    for j in range(3):
        for k in range(3):
            V[j*7+(k+3), 23:46] = V_l[j*3+k]
    for i in range(3):
        for k in range(3):
            W[(k+3)*3+i, 23:46] = W_l[i*3+k]
    
    for i in range(3):
        for j in range(3):
            c = 46 + i*3 + j
            U[i*3+j, c] = 1.0
            V[j*7+6, c] = 1.0
            W[6*3+i, c] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
