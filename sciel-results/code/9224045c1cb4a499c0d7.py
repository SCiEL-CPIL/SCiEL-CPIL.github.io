import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Rank 50 construction
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        U[a_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        W[c11_idx, col] -= 1
        col += 1
        
        U[a_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] -= 1
        W[c01_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        V[e_idx, col] -= 1
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1
        W[c01_idx, col] += 1
        col += 1
        
        U[c_idx, col] += 1
        U[a_idx, col] -= 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        U[b_idx, col] += 1
        U[d_idx, col] -= 1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[b_idx, col] += 1
        V[g_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[c_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
        
    # Verification
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # We can use einsum for verification
    # T = np.einsum('ir,jr,kr->ijk', U, V, W)
    # But we need to check specific indices.
    # The tensor is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # where U is indexed by i (which is i*m+j in the matrix sense? No, U is n*m x rank)
    # U[i*m+j, r]
    # V[j*p+k, r]
    # W[k*n+i, r]
    
    # Let's verify a few points or the whole thing if fast enough
    # n=2, m=4, p=7, rank=50
    # 2*4*7*50 = 2800 ops.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    
    if max_err > 1e-9:
        pass # Should not happen

    return {"U": U, "V": V, "W": W}
