"""Matrix multiplication 3,3,8 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    deadline = time.time() + 140
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Block decomposition: columns 0-2, 3-5, 6-7
    # Each 3x3 block: schoolbook rank 9
    # 3x2 block: schoolbook rank 6
    # Total: 24
    
    for col_start, col_end in [(0, 3), (3, 6), (6, 8)]:
        num_cols = col_end - col_start
        for i in range(n):
            for j in range(m):
                for k in range(num_cols):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    v[j * p + (col_start + k)] = 1.0
                    w[(col_start + k) * n + i] = 1.0
                    add_term(u, v, w)
    
    rank = len(U_cols)
    U = [[U_cols[c][i] for c in range(rank)] for i in range(n * m)]
    V = [[V_cols[c][i] for c in range(rank)] for i in range(m * p)]
    W = [[W_cols[c][i] for c in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
