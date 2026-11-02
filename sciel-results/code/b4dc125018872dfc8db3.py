import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Construct a rank-39 exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    - Split B into columns [0:2], [2:4], [4:6], [6]
    - Apply Strassen (rank 7) to each 2x2 block of A[0:2,0:2] * B[0:2, k:k+2] for k in {0,2,4}
    - Apply schoolbook (rank 4) for A[0:2,0:2] * B[0:2,6] (single column)
    - Apply schoolbook (rank 14) for A[0:2,2] * B[2,0:7] (single row of A)
    
    Total rank: 7*3 + 4 + 14 = 21 + 4 + 14 = 39
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    _start_time = time.time()
    _deadline = _start_time + 100.0  # safety margin before 120s limit
    
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
    
    # Helper: add Strassen terms for A[0:2, 0:2] * B[0:2, c0:c0+2]
    def add_strassen_2x2(c0):
        # M1
        add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,c0): 1, v_idx(1,c0+1): 1}, {w_idx(c0,0): 1, w_idx(c0+1,1): 1})
        # M2
        add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,c0): 1}, {w_idx(c0,1): 1, w_idx(c0+1,1): -1})
        # M3
        add_term({u_idx(0,0): 1}, {v_idx(0,c0+1): 1, v_idx(1,c0+1): -1}, {w_idx(c0+1,0): 1, w_idx(c0+1,1): 1})
        # M4
        add_term({u_idx(1,1): 1}, {v_idx(1,c0): 1, v_idx(0,c0): -1}, {w_idx(c0,0): 1, w_idx(c0,1): 1})
        # M5
        add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,c0+1): 1}, {w_idx(c0,0): -1, w_idx(c0+1,0): 1})
        # M6
        add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,c0): 1, v_idx(0,c0+1): 1}, {w_idx(c0+1,1): 1})
        # M7
        add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,c0): 1, v_idx(1,c0+1): 1}, {w_idx(c0,0): 1})
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_2x2(0)
    
    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_2x2(2)
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_2x2(4)
    
    # Part 4: Schoolbook for A[0:2, 0:2] * B[0:2, 6] (single column)
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,6): 1}, {w_idx(6,1): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7] (row 2 of A, all 7 columns of B)
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
    
    rank = len(terms)
    
    # Check deadline before constructing matrices
    if time.time() > _deadline:
        # Return a valid but potentially suboptimal result
        pass
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u_list[i]
        for i in range(m * p):
            V[i][c] = v_list[i]
        for i in range(p * n):
            W[i][c] = w_list[i]
            
    return {"U": U, "V": V, "W": W}
