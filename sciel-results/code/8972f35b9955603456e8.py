import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Hybrid Strassen/Schoolbook: Rank 39.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen for 2x2x2 at B columns (k0, k1)
    def strassen_2x2x2(k0, k1):
        add_term({u_idx(0,0):1, u_idx(1,1):1}, {v_idx(0,k0):1, v_idx(1,k1):1}, {w_idx(k0,0):1, w_idx(k1,1):1})
        add_term({u_idx(1,0):1, u_idx(1,1):1}, {v_idx(0,k0):1}, {w_idx(k0,1):1, w_idx(k1,1):-1})
        add_term({u_idx(0,0):1}, {v_idx(0,k1):1, v_idx(1,k1):-1}, {w_idx(k1,0):1, w_idx(k1,1):1})
        add_term({u_idx(1,1):1}, {v_idx(1,k0):1, v_idx(0,k0):-1}, {w_idx(k0,0):1, w_idx(k0,1):1})
        add_term({u_idx(0,0):1, u_idx(0,1):1}, {v_idx(1,k1):1}, {w_idx(k0,0):-1, w_idx(k1,0):1})
        add_term({u_idx(1,0):1, u_idx(0,0):-1}, {v_idx(0,k0):1, v_idx(0,k1):1}, {w_idx(k1,1):1})
        add_term({u_idx(0,1):1, u_idx(1,1):-1}, {v_idx(1,k0):1, v_idx(1,k1):1}, {w_idx(k0,0):1})
    
    # Schoolbook for 2x2x1 at B column k
    def schoolbook_2x2x1(k):
        add_term({u_idx(0,0):1}, {v_idx(0,k):1}, {w_idx(k,0):1})
        add_term({u_idx(0,1):1}, {v_idx(1,k):1}, {w_idx(k,0):1})
        add_term({u_idx(1,0):1}, {v_idx(0,k):1}, {w_idx(k,1):1})
        add_term({u_idx(1,1):1}, {v_idx(1,k):1}, {w_idx(k,1):1})
    
    # Schoolbook for 2x1x1 at B column k (A row i, B row 2)
    def schoolbook_2x1x1(k):
        add_term({u_idx(0,2):1}, {v_idx(2,k):1}, {w_idx(k,0):1})
        add_term({u_idx(1,2):1}, {v_idx(2,k):1}, {w_idx(k,1):1})
    
    strassen_2x2x2(0, 1)
    schoolbook_2x2x1(2)
    strassen_2x2x2(3, 4)
    strassen_2x2x2(5, 6)
    for k in range(p):
        schoolbook_2x1x1(k)
    
    rank = len(terms)
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    for c, (u,v,w) in enumerate(terms):
        for i in range(n*m): U[i][c] = u[i]
        for i in range(m*p): V[i][c] = v[i]
        for i in range(p*n): W[i][c] = w[i]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
