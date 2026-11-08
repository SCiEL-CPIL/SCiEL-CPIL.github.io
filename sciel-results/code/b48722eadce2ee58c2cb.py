import numpy as np
import scipy.optimize
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank 14 decomposition for 2x3x3
    # Then replicate 3 times for 2x3x9 -> Rank 42
    n_sub = 2
    m_sub = 3
    p_sub = 3
    r_sub = 14
    
    # Time guard
    start_time = time.time()
    timeout = 10.0 # seconds for optimization
    
    def loss_func(params):
        # Reshape params to U, V, W
        size_u = n_sub * m_sub * r_sub
        size_v = m_sub * p_sub * r_sub
        size_w = p_sub * n_sub * r_sub
        
        U = params[:size_u].reshape((n_sub * m_sub, r_sub))
        V = params[size_u:size_u+size_v].reshape((m_sub * p_sub, r_sub))
        W = params[size_u+size_v:].reshape((p_sub * n_sub, r_sub))
        
        # Compute tensor T_{a,b,r} = sum_t U_{a,b,t} V_{b,r,t} W_{r,a,t}
        # U index: a*m_sub + b
        # V index: b*p_sub + r
        # W index: r*n_sub + a
        
        # We want T_{a,b,r} = 1 for all a,b,r
        error = 0.0
        for a in range(n_sub):
            for b in range(m_sub):
                for r in range(p_sub):
                    u_val = U[a * m_sub + b]
                    v_val = V[b * p_sub + r]
                    w_val = W[r * n_sub + a]
                    # sum_t u_val[t] * v_val[t] * w_val[t]
                    val = np.dot(np.dot(u_val, v_val), w_val)
                    error += (val - 1.0) ** 2
        return error

    # Initialize random
    np.random.seed(42)
    init_u = np.random.randn(n_sub * m_sub * r_sub) * 0.5
    init_v = np.random.randn(m_sub * p_sub * r_sub) * 0.5
    init_w = np.random.randn(p_sub * n_sub * r_sub) * 0.5
    init_params = np.concatenate([init_u, init_v, init_w])
    
    try:
        result = scipy.optimize.minimize(loss_func, init_params, method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
        
        if time.time() - start_time > timeout:
            raise TimeoutError()
            
        if result.fun < 1e-5:
            # Extract solution
            size_u = n_sub * m_sub * r_sub
            size_v = m_sub * p_sub * r_sub
            size_w = p_sub * n_sub * r_sub
            
            U_sol = result.x[:size_u].reshape((n_sub * m_sub, r_sub))
            V_sol = result.x[size_u:size_u+size_v].reshape((m_sub * p_sub, r_sub))
            W_sol = result.x[size_u+size_v:].reshape((p_sub * n_sub, r_sub))
            
            # Round to nearest integer
            U_int = np.round(U_sol).astype(int)
            V_int = np.round(V_sol).astype(int)
            W_int = np.round(W_sol).astype(int)
            
            # Verify exactness
            valid = True
            for a in range(n_sub):
                for b in range(m_sub):
                    for r in range(p_sub):
                        val = np.dot(np.dot(U_int[a * m_sub + b], V_int[b * p_sub + r]), W_int[r * n_sub + a])
                        if val != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Construct full 2x3x9 decomposition
                rank_full = r_sub * 3
                U_full = np.zeros((n * m, rank_full), dtype=int)
                V_full = np.zeros((m * p, rank_full), dtype=int)
                W_full = np.zeros((p * n, rank_full), dtype=int)
                
                for q in range(3):
                    offset = q * r_sub
                    # U is same for all blocks (A is shared)
                    U_full[:, offset:offset+r_sub] = U_int
                    # V maps B columns 3q..3q+2
                    for b in range(m_sub):
                        for r in range(p_sub):
                            idx_v_full = b * p + (q * p_sub + r)
                            idx_v_sub = b * p_sub + r
                            V_full[idx_v_full, offset:offset+r_sub] = V_int[idx_v_sub]
                    # W maps C columns 3q..3q+2
                    for r in range(p_sub):
                        for a in range(n_sub):
                            idx_w_full = (q * p_sub + r) * n + a
                            idx_w_sub = r * n_sub + a
                            W_full[idx_w_full, offset:offset+r_sub] = W_int[idx_w_sub]
                
                return {"U": U_full, "V": V_full, "W": W_full}
    except Exception:
        pass

    return get_schoolbook()
# EVOLVE-BLOCK-END
