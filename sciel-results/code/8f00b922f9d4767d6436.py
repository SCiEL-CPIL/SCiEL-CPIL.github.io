import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for p_block in range(2):
        for q_block in range(2):
            for s_block in range(3):
                # A block indices: rows 2*p_block, 2*p_block+1; cols 2*q_block, 2*q_block+1
                # B block indices: rows 2*q_block, 2*q_block+1; cols 2*s_block, 2*s_block+1
                # C block indices: rows 2*p_block, 2*p_block+1; cols 2*s_block, 2*s_block+1
                
                # Helper to get A index
                def a_idx(r, c):
                    return (2*p_block + r) * m + (2*q_block + c)
                # Helper to get B index
                def b_idx(r, c):
                    return (2*q_block + r) * p + (2*s_block + c)
                # Helper to get W index (k*n + i)
                def w_idx(k, i):
                    return k * n + (2*p_block + i)
                
                # Strassen terms
                # M1
                U[a_idx(0,0), col] = 1
                U[a_idx(1,1), col] = 1
                V[b_idx(0,0), col] = 1
                V[b_idx(1,1), col] = 1
                W[w_idx(0,0), col] = 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M2
                U[a_idx(1,0), col] = 1
                U[a_idx(1,1), col] = 1
                V[b_idx(0,0), col] = 1
                W[w_idx(0,1), col] = 1 # Z10 is C[1,0] of block -> i=1, k=0
                col += 1
                
                # M3
                U[a_idx(0,0), col] = 1
                V[b_idx(1,0), col] = 1
                V[b_idx(1,1), col] = -1
                W[w_idx(1,0), col] = 1 # Z01 is C[0,1] -> i=0, k=1
                W[w_idx(1,1), col] = 1 # Z11 is C[1,1] -> i=1, k=1
                col += 1
                
                # M4
                U[a_idx(1,1), col] = 1
                V[b_idx(0,1), col] = 1
                V[b_idx(0,0), col] = -1
                W[w_idx(0,0), col] = 1 # Z00
                W[w_idx(0,1), col] = 1 # Z10
                col += 1
                
                # M5
                U[a_idx(0,0), col] = 1
                U[a_idx(0,1), col] = 1
                V[b_idx(1,1), col] = 1
                W[w_idx(0,0), col] = -1 # -Z00
                W[w_idx(1,0), col] = 1 # Z01
                col += 1
                
                # M6
                U[a_idx(1,0), col] = 1
                U[a_idx(0,0), col] = -1
                V[b_idx(0,0), col] = 1
                V[b_idx(0,1), col] = 1
                W[w_idx(1,1), col] = 1 # Z11
                col += 1
                
                # M7
                U[a_idx(0,1), col] = 1
                U[a_idx(1,1), col] = -1
                V[b_idx(1,0), col] = 1
                V[b_idx(1,1), col] = 1
                W[w_idx(0,0), col] = 1 # Z00
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
