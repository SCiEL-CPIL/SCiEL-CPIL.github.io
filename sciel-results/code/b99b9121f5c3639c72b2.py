import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

def _find_2x3x3_decomposition(max_time=25):
    """Find a rank-13 decomposition of the 2x3x3 MM tensor with half-integer coefficients."""
    n, m, p = 2, 3, 3
    rank = 13
    start_time = time.time()
    
    def compute_error(params):
        U = params[0:78].reshape(6, 13)
        V = params[78:162].reshape(9, 13)
        W = params[162:240].reshape(6, 13)
        error = 0.0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    val = sum(U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] for r in range(13))
                    error += (val - 1.0) ** 2
        return error
    
    best_params = None
    best_error = float('inf')
    
    np.random.seed(42)
    for trial in range(100):
        if time.time() - start_time > max_time - 3:
            break
        
        params = np.random.randn(240) * 0.5
        try:
            result = minimize(compute_error, params, method='BFGS',
                            options={'maxiter': 2000, 'gtol': 1e-12})
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
            if best_error < 1e-14:
                break
        except Exception:
            continue
    
    if best_params is None or best_error > 1e-10:
        return None
    
    U = best_params[0:78].reshape(6, 13)
    V = best_params[78:162].reshape(9, 13)
    W = best_params[162:240].reshape(6, 13)
    
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    for i in range(2):
        for j in range(3):
            for k in range(3):
                val = sum(U_r[i*3+j, r] * V_r[j*3+k, r] * W_r[k*2+i, r] for r in range(13))
                if abs(val - 1.0) > 1e-9:
                    return None
    
    return (U_r, V_r, W_r)

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
                    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        def gI(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def gJ(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def gK(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )
    
    def add_2x3x3(i_offset, j_offset, k_offset, decomp):
        if decomp is None:
            schoolbook(2, 3, 3, i_offset, j_offset, k_offset)
            return
        U_sub, V_sub, W_sub = decomp
        for r in range(13):
            u_vec = []
            v_vec = []
            w_vec = []
            for i_loc in range(2):
                for j_loc in range(3):
                    val = U_sub[i_loc*3+j_loc, r]
                    if abs(val) > 1e-12:
                        I = (i_offset + i_loc) * m + (j_offset + j_loc)
                        u_vec.append((I, val))
            for j_loc in range(3):
                for k_loc in range(3):
                    val = V_sub[j_loc*3+k_loc, r]
                    if abs(val) > 1e-12:
                        J = (j_offset + j_loc) * p + (k_offset + k_loc)
                        v_vec.append((J, val))
            for k_loc in range(3):
                for i_loc in range(2):
                    val = W_sub[k_loc*2+i_loc, r]
                    if abs(val) > 1e-12:
                        K = (k_offset + k_loc) * n + (i_offset + i_loc)
                        w_vec.append((K, val))
            add_term(u_vec, v_vec, w_vec)
    
    deadline = time.time() + 140
    decomp_2x3x3 = None
    if time.time() < deadline:
        decomp_2x3x3 = _find_2x3x3_decomposition(max_time=min(30, deadline - time.time()))
    
    if decomp_2x3x3 is not None:
        i_groups = [(0, 2), (2, 2)]
        j_groups = [(0, 3), (3, 2)]
        k_groups = [(0, 3), (3, 2)]
        
        for i_start, i_size in i_groups:
            for j_start, j_size in j_groups:
                for k_start, k_size in k_groups:
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        strassen_2x2x2(i_start, j_start, k_start)
                    elif i_size == 2 and j_size == 3 and k_size == 3:
                        add_2x3x3(i_start, j_start, k_start, decomp_2x3x3)
                    else:
                        schoolbook(i_size, j_size, k_size, i_start, j_start, k_start)
    else:
        i_groups = [(0, 2), (2, 2)]
        j_groups = [(0, 2), (2, 2), (4, 1)]
        k_groups = [(0, 2), (2, 2), (4, 1)]
        
        for i_start, i_size in i_groups:
            for j_start, j_size in j_groups:
                for k_start, k_size in k_groups:
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        strassen_2x2x2(i_start, j_start, k_start)
                    else:
                        schoolbook(i_size, j_size, k_size, i_start, j_start, k_start)
    
    if not U_cols:
        return {"U": [[0]*100 for _ in range(20)], "V": [[0]*100 for _ in range(25)], "W": [[0]*100 for _ in range(20)]}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
