import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    def strassen_2x2x2(k0, k1):
        add_term(
            {0*m+0: 1, 1*m+1: 1},
            {0*p+k0: 1, 1*p+k1: 1},
            {k0*n+0: 1, k1*n+1: 1}
        )
        add_term(
            {1*m+0: 1, 1*m+1: 1},
            {0*p+k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        add_term(
            {0*m+0: 1},
            {0*p+k1: 1, 1*p+k1: -1},
            {k1*n+0: 1, k1*n+1: 1}
        )
        add_term(
            {1*m+1: 1},
            {1*p+k0: 1, 0*p+k0: -1},
            {k0*n+0: 1, k0*n+1: 1}
        )
        add_term(
            {0*m+0: 1, 0*m+1: 1},
            {1*p+k1: 1},
            {k0*n+0: -1, k1*n+0: 1}
        )
        add_term(
            {1*m+0: 1, 0*m+0: -1},
            {0*p+k0: 1, 0*p+k1: 1},
            {k1*n+1: 1}
        )
        add_term(
            {0*m+1: 1, 1*m+1: -1},
            {1*p+k0: 1, 1*p+k1: 1},
            {k0*n+0: 1}
        )
    
    for block_start in range(0, 8, 2):
        strassen_2x2x2(block_start, block_start + 1)
    
    k = 8
    add_term({0*m+0: 1}, {0*p+k: 1}, {k*n+0: 1})
    add_term({0*m+1: 1}, {1*p+k: 1}, {k*n+0: 1})
    add_term({1*m+0: 1}, {0*p+k: 1}, {k*n+1: 1})
    add_term({1*m+1: 1}, {1*p+k: 1}, {k*n+1: 1})
    
    for i in range(n):
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
    
    rank = len(U_cols)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    U_list = [[U[i, r] for r in range(rank)] for i in range(n*m)]
    V_list = [[V[j, r] for r in range(rank)] for j in range(m*p)]
    W_list = [[W[k, r] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
