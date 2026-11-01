import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i

    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        if time.time() > deadline:
            return False
            
        # M1
        U[u_idx(i_offset, j_offset), col] += 1
        U[u_idx(i_offset+1, j_offset+1), col] += 1
        V[v_idx(j_offset, k_offset), col] += 1
        V[v_idx(j_offset+1, k_offset+1), col] += 1
        W[w_idx(k_offset, i_offset), col] += 1
        W[w_idx(k_offset+1, i_offset+1), col] += 1
        col += 1
        
        # M2
        U[u_idx(i_offset+1, j_offset), col] += 1
        U[u_idx(i_offset+1, j_offset+1), col] += 1
        V[v_idx(j_offset, k_offset), col] += 1
        W[w_idx(k_offset, i_offset+1), col] += 1
        col += 1
        
        # M3
        U[u_idx(i_offset, j_offset), col] += 1
        V[v_idx(j_offset+1, k_offset), col] += 1
        V[v_idx(j_offset+1, k_offset+1), col] -= 1
        W[w_idx(k_offset+1, i_offset), col] += 1
        W[w_idx(k_offset+1, i_offset+1), col] += 1
        col += 1
        
        # M4
        U[u_idx(i_offset+1, j_offset+1), col] += 1
        V[v_idx(j_offset, k_offset+1), col] += 1
        V[v_idx(j_offset, k_offset), col] -= 1
        W[w_idx(k_offset, i_offset), col] += 1
        W[w_idx(k_offset, i_offset+1), col] += 1
        col += 1
        
        # M5
        U[u_idx(i_offset, j_offset), col] += 1
        U[u_idx(i_offset+1, j_offset), col] += 1
        V[v_idx(j_offset+1, k_offset+1), col] += 1
        W[w_idx(k_offset, i_offset), col] -= 1
        W[w_idx(k_offset+1, i_offset), col] += 1
        col += 1
        
        # M6
        U[u_idx(i_offset+1, j_offset), col] += 1
        U[u_idx(i_offset, j_offset), col] -= 1
        V[v_idx(j_offset, k_offset), col] += 1
        V[v_idx(j_offset+1, k_offset), col] += 1
        W[w_idx(k_offset+1, i_offset+1), col] += 1
        col += 1
        
        # M7
        U[u_idx(i_offset+1, j_offset+1), col] += 1
        U[u_idx(i_offset, j_offset+1), col] -= 1
        V[v_idx(j_offset, k_offset+1), col] += 1
        V[v_idx(j_offset+1, k_offset+1), col] += 1
        W[w_idx(k_offset, i_offset), col] += 1
        col += 1
        return True

    def add_schoolbook_2x2x1(i_offset, j_offset, k_offset):
        nonlocal col
        if time.time() > deadline:
            return False

        U[u_idx(i_offset, j_offset), col] += 1
        V[v_idx(j_offset, k_offset), col] += 1
        W[w_idx(k_offset, i_offset), col] += 1
        col += 1
        
        U[u_idx(i_offset, j_offset+1), col] += 1
        V[v_idx(j_offset+1, k_offset), col] += 1
        W[w_idx(k_offset, i_offset), col] += 1
        col += 1
        
        U[u_idx(i_offset+1, j_offset), col] += 1
        V[v_idx(j_offset, k_offset), col] += 1
        W[w_idx(k_offset, i_offset+1), col] += 1
        col += 1
        
        U[u_idx(i_offset+1, j_offset+1), col] += 1
        V[v_idx(j_offset+1, k_offset), col] += 1
        W[w_idx(k_offset, i_offset+1), col] += 1
        col += 1
        return True

    # Execute blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook_2x2x1(0, 0, 4)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook_2x2x1(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
