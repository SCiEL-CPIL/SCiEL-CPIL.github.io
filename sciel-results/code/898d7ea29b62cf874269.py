import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        def get_u_idx(i, j):
            return a_rows[i] * m + a_cols[j]
        def get_v_idx(j, k):
            return b_rows[j] * p + b_cols[k]
        def get_w_idx(k, i):
            return c_cols[k] * n + c_rows[i]
            
        # M1
        U[get_u_idx(0,0), start_col] += 1
        U[get_u_idx(1,1), start_col] += 1
        V[get_v_idx(0,0), start_col] += 1
        V[get_v_idx(1,1), start_col] += 1
        W[get_w_idx(0,0), start_col] += 1
        W[get_w_idx(1,1), start_col] += 1
        
        # M2
        U[get_u_idx(1,0), start_col+1] += 1
        U[get_u_idx(1,1), start_col+1] += 1
        V[get_v_idx(0,0), start_col+1] += 1
        W[get_w_idx(0,1), start_col+1] += 1
        W[get_w_idx(1,1), start_col+1] -= 1
        
        # M3
        U[get_u_idx(0,0), start_col+2] += 1
        V[get_v_idx(0,1), start_col+2] += 1
        V[get_v_idx(1,1), start_col+2] -= 1
        W[get_w_idx(0,0), start_col+2] += 1
        W[get_w_idx(1,1), start_col+2] += 1
        
        # M4
        U[get_u_idx(1,1), start_col+3] += 1
        V[get_v_idx(1,0), start_col+3] += 1
        V[get_v_idx(0,0), start_col+3] -= 1
        W[get_w_idx(0,0), start_col+3] += 1
        W[get_w_idx(0,1), start_col+3] += 1
        
        # M5
        U[get_u_idx(0,0), start_col+4] += 1
        U[get_u_idx(0,1), start_col+4] += 1
        V[get_v_idx(1,1), start_col+4] += 1
        W[get_w_idx(0,0), start_col+4] -= 1
        W[get_w_idx(0,1), start_col+4] += 1
        
        # M6
        U[get_u_idx(1,0), start_col+5] += 1
        U[get_u_idx(0,0), start_col+5] -= 1
        V[get_v_idx(0,0), start_col+5] += 1
        V[get_v_idx(0,1), start_col+5] += 1
        W[get_w_idx(1,1), start_col+5] += 1
        
        # M7
        U[get_u_idx(0,1), start_col+6] += 1
        U[get_u_idx(1,1), start_col+6] -= 1
        V[get_v_idx(1,0), start_col+6] += 1
        V[get_v_idx(1,1), start_col+6] += 1
        W[get_w_idx(0,0), start_col+6] += 1
        
        return start_col + 7

    def add_mv(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        def get_u_idx(i, j):
            return a_rows[i] * m + a_cols[j]
        def get_v_idx(j, k):
            return b_rows[j] * p + b_cols[k]
        def get_w_idx(k, i):
            return c_cols[k] * n + c_rows[i]
            
        U[get_u_idx(0,0), start_col] += 1
        V[get_v_idx(0,0), start_col] += 1
        W[get_w_idx(0,0), start_col] += 1
        
        U[get_u_idx(0,1), start_col+1] += 1
        V[get_v_idx(1,0), start_col+1] += 1
        W[get_w_idx(0,0), start_col+1] += 1
        
        U[get_u_idx(1,0), start_col+2] += 1
        V[get_v_idx(0,0), start_col+2] += 1
        W[get_w_idx(0,1), start_col+2] += 1
        
        U[get_u_idx(1,1), start_col+3] += 1
        V[get_v_idx(1,0), start_col+3] += 1
        W[get_w_idx(0,1), start_col+3] += 1
        
        return start_col + 4

    col = add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    col = add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col)
    col = add_mv([0,1], [0,1], [0,1], [4], [0,1], [4], col)
    
    col = add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col)
    col = add_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], col)
    col = add_mv([0,1], [2,3], [2,3], [4], [0,1], [4], col)
    
    return {"U": U, "V": V, "W": W}
