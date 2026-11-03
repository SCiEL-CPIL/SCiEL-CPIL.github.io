import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _build_strassen56(n, m, p):
    """Build the rank-56 Strassen-based decomposition."""
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(2):
        for j in range(4):
            a_idx = {
                (0,0): 0 * m + (2*s + 0),
                (0,1): 0 * m + (2*s + 1),
                (1,0): 1 * m + (2*s + 0),
                (1,1): 1 * m + (2*s + 1)
            }
            b_idx = {
                (0,0): (2*s + 0) * p + (2*j + 0),
                (0,1): (2*s + 0) * p + (2*j + 1),
                (1,0): (2*s + 1) * p + (2*j + 0),
                (1,1): (2*s + 1) * p + (2*j + 1)
            }
            c_idx = {
                (0,0): (2*j + 0) * n + 0,
                (0,1): (2*j + 1) * n + 0,
                (1,0): (2*j + 0) * n + 1,
                (1,1): (2*j + 1) * n + 1
            }
            
            U[a_idx[(0,0)]][col] += 1
            U[a_idx[(1,1)]][col] += 1
            U[a_idx[(1,0)]][col+1] += 1
            U[a_idx[(1,1)]][col+1] += 1
            U[a_idx[(0,0)]][col+2] += 1
            U[a_idx[(1,1)]][col+3] += 1
            U[a_idx[(0,0)]][col+4] += 1
            U[a_idx[(0,1)]][col+4] += 1
            U[a_idx[(1,0)]][col+5] += 1
            U[a_idx[(0,0)]][col+5] += -1
            U[a_idx[(0,1)]][col+6] += 1
            U[a_idx[(1,1)]][col+6] += -1
            
            V[b_idx[(0,0)]][col] += 1
            V[b_idx[(1,1)]][col] += 1
            V[b_idx[(0,0)]][col+1] += 1
            V[b_idx[(0,1)]][col+2] += 1
            V[b_idx[(1,1)]][col+2] += -1
            V[b_idx[(1,0)]][col+3] += 1
            V[b_idx[(0,0)]][col+3] += -1
            V[b_idx[(1,1)]][col+4] += 1
            V[b_idx[(0,0)]][col+5] += 1
            V[b_idx[(0,1)]][col+5] += 1
            V[b_idx[(1,0)]][col+6] += 1
            V[b_idx[(1,1)]][col+6] += 1
            
            W[c_idx[(0,0)]][col] += 1
            W[c_idx[(0,0)]][col+3] += 1
            W[c_idx[(0,0)]][col+4] += -1
            W[c_idx[(0,0)]][col+6] += 1
            W[c_idx[(0,1)]][col+2] += 1
            W[c_idx[(0,1)]][col+4] += 1
            W[c_idx[(1,0)]][col+1] += 1
            W[c_idx[(1,0)]][col+3] += 1
            W[c_idx[(1,1)]][col] += 1
            W[c_idx[(1,1)]][col+1] += -1
            W[c_idx[(1,1)]][col+2] += 1
            W[c_idx[(1,1)]][col+5] += 1
            
            col += 7
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen56(n, m, p)
    
    # Try to reduce rank if possible
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Simple reduction check
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(rank):
            if time.time() > deadline:
                break
            U_col = U_arr[:, r]
            V_col = V_arr[:, r]
            W_col = W_arr[:, r]
            removable = True
            for i in range(n):
                for j in range(m):
                    if abs(U_col[i*m+j]) > 1e-9:
                        for k in range(p):
                            if abs(V_col[j*p+k]) > 1e-9 and abs(W_col[k*n+i]) > 1e-9:
                                removable = False
                                break
                        if not removable: break
                if not removable: break
            if removable:
                U_arr = np.delete(U_arr, r, axis=1)
                V_arr = np.delete(V_arr, r, axis=1)
                W_arr = np.delete(W_arr, r, axis=1)
                rank -= 1
                improved = True
                break

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
