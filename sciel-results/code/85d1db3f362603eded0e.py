"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def strassen_2x2x2x2(U, V, W, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, base_col):
        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        
        terms = [
            ([(u_idx(ar0, ac0), 1), (u_idx(ar1, ac1), 1)],
             [(v_idx(br0, bc0), 1), (v_idx(br1, bc1), 1)],
             [(w_idx(cc0, cr0), 1), (w_idx(cc1, cr0), 1), (w_idx(cc1, cr1), 1)]),
            ([(u_idx(ar1, ac0), 1), (u_idx(ar1, ac1), 1)],
             [(v_idx(br0, bc0), 1)],
             [(w_idx(cc1, cr0), 1), (w_idx(cc1, cr1), -1)]),
            ([(u_idx(ar0, ac0), 1)],
             [(v_idx(br0, bc1), 1), (v_idx(br1, bc1), -1)],
             [(w_idx(cc0, cr1), 1), (w_idx(cc1, cr1), 1)]),
            ([(u_idx(ar1, ac1), 1)],
             [(v_idx(br1, bc0), 1), (v_idx(br0, bc0), -1)],
             [(w_idx(cc0, cr0), 1), (w_idx(cc1, cr0), 1)]),
            ([(u_idx(ar0, ac0), 1), (u_idx(ar0, ac1), 1)],
             [(v_idx(br1, bc1), 1)],
             [(w_idx(cc0, cr0), -1), (w_idx(cc0, cr1), 1)]),
            ([(u_idx(ar1, ac0), 1), (u_idx(ar0, ac0), -1)],
             [(v_idx(br0, bc0), 1), (v_idx(br0, bc1), 1)],
             [(w_idx(cc1, cr1), 1)]),
            ([(u_idx(ar0, ac1), 1), (u_idx(ar1, ac1), -1)],
             [(v_idx(br1, bc0), 1), (v_idx(br1, bc1), 1)],
             [(w_idx(cc0, cr0), 1)]),
        ]
        
        col = base_col
        for t in terms:
            for u_i, c in t[0]:
                U[u_i, col] += c
            for v_i, c in t[1]:
                V[v_i, col] += c
            for w_i, c in t[2]:
                W[w_i, col] += c
            col += 1
        return col
    
    col = 0
    # Left block
    col = strassen_2x2x2x2(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], col)
    col = strassen_2x2x2x2(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col)
    col = strassen_2x2x2x2(U, V, W, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col)
    col = strassen_2x2x2x2(U, V, W, [0,1], [2,3], [2,3], [2,3], [0,1], [2,3], col)
    # Right block
    col = strassen_2x2x2x2(U, V, W, [0,1], [0,1], [0,1], [4,5], [0,1], [4,5], col)
    col = strassen_2x2x2x2(U, V, W, [0,1], [2,3], [2,3], [4,5], [0,1], [4,5], col)
    col = strassen_2x2x2x2(U, V, W, [0,1], [0,1], [0,1], [6,7], [0,1], [6,7], col)
    col = strassen_2x2x2x2(U, V, W, [0,1], [2,3], [2,3], [6,7], [0,1], [6,7], col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

