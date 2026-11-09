import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 110.0
    
    a_size = 2 * 3
    b_size = 3 * 2
    c_size = 2 * 2
    
    T_small = np.zeros((a_size, b_size, c_size))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                a = i * 3 + j
                b = j * 2 + k
                c = k * 2 + i
                T_small[a, b, c] = 1.0
                
    from scipy.optimize import minimize
    
    def _try_rank(rank):
        if time.time() > deadline:
            return None
            
        def _objective(x):
            U = x[:a_size*rank].reshape(a_size, rank)
            V = x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
            W = x[a_size*rank+b_size*rank:].reshape(c_size, rank)
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T_small - recon
            return np.sum(residual**2)
            
        for seed in range(20):
            if time.time() > deadline:
                return None
            np.random.seed(seed)
            x0 = np.random.randn(a_size*rank + b_size*rank + c_size*rank) * 0.1
            try:
                res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < 1e-9:
                    U = res.x[:a_size*rank].reshape(a_size, rank)
                    V = res.x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
                    W = res.x[a_size*rank+b_size*rank:].reshape(c_size, rank)
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(T_small - recon)) < 1e-8:
                        return (U, V, W)
            except Exception:
                pass
        return None

    found_small = None
    final_rank_small = 0
    
    # Try ranks 5, 6, 7, 8
    for r in [5, 6, 7, 8]:
        found_small = _try_rank(r)
        if found_small is not None:
            final_rank_small = r
            break
            
    if found_small is None:
        return _schoolbook(n, m, p)
        
    U_small, V_small, W_small = found_small
    
    rank_total = 4 * final_rank_small
    U_total = np.zeros((n*m, rank_total))
    V_total = np.zeros((m*p, rank_total))
    W_total = np.zeros((p*n, rank_total))
    
    col_offset = 0
    for g in range(4):
        U_total[:, col_offset:col_offset+final_rank_small] = U_small
        
        for r_idx in range(b_size):
            j = r_idx // 2
            k_local = r_idx % 2
            b_global = j * 8 + 2 * g + k_local
            V_total[b_global, col_offset:col_offset+final_rank_small] = V_small[r_idx, :]
            
        for r_idx in range(c_size):
            k_local = r_idx // 2
            i = r_idx % 2
            c_global = 4 * g + 2 * k_local + i
            W_total[c_global, col_offset:col_offset+final_rank_small] = W_small[r_idx, :]
            
        col_offset += final_rank_small
        
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
                
    recon_full = np.einsum('ar,br,cr->abc', U_total, V_total, W_total)
    if np.max(np.abs(T_full - recon_full)) > 1e-8:
        return _schoolbook(n, m, p)
        
    return {
        "U": _to_list(U_total),
        "V": _to_list(V_total),
        "W": _to_list(W_total)
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {
        "U": _to_list(U),
        "V": _to_list(V),
        "W": _to_list(W)
    }

def _to_list(mat):
    result = []
    for row in mat:
        row_list = []
        for val in row:
            nearest_int = round(val)
            nearest_half = round(2 * val) / 2.0
            if abs(val - nearest_int) < 1e-6:
                row_list.append(int(nearest_int))
            elif abs(val - nearest_half) < 1e-6:
                row_list.append(nearest_half)
            else:
                row_list.append(float(val))
        result.append(row_list)
    return result
