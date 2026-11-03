import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Generate all equations
    # u_idx in [0, dim_u), v_idx in [0, dim_v), w_idx in [0, dim_w)
    # Target is 1 if (u_idx, v_idx, w_idx) corresponds to (i,j; j,k; i,k)
    # i = u_idx // m, j = u_idx % m
    # j' = v_idx // p, k' = v_idx % p
    # i' = w_idx % n, k = w_idx // n
    # Condition: i == i', j == j', k == k'
    
    eqs_u = []
    eqs_v = []
    eqs_w = []
    eqs_target = []
    
    for u_idx in range(dim_u):
        i = u_idx // m
        j = u_idx % m
        for v_idx in range(dim_v):
            j_prime = v_idx // p
            k_prime = v_idx % p
            for w_idx in range(dim_w):
                i_prime = w_idx % n
                k = w_idx // n
                
                if i == i_prime and j == j_prime and k == k_prime:
                    eqs_u.append(u_idx)
                    eqs_v.append(v_idx)
                    eqs_w.append(w_idx)
                    eqs_target.append(1.0)
                else:
                    eqs_u.append(u_idx)
                    eqs_v.append(v_idx)
                    eqs_w.append(w_idx)
                    eqs_target.append(0.0)
                    
    num_eqs = len(eqs_u)
    eqs_u = np.array(eqs_u)
    eqs_v = np.array(eqs_v)
    eqs_w = np.array(eqs_w)
    eqs_target = np.array(eqs_target)
    
    # Try rank 44
    R_target = 44
    
    def make_objective(gauge):
        def objective(params):
            if gauge == 0:
                U_rest = params[:R_target*(dim_u-1)].reshape(R_target, dim_u-1)
                V = params[R_target*(dim_u-1):R_target*(dim_u-1+dim_v)].reshape(R_target, dim_v)
                W = params[R_target*(dim_u-1+dim_v):].reshape(R_target, dim_w)
                
                U = np.zeros((R_target, dim_u))
                U[:, 0] = 1.0
                U[:, 1:] = U_rest
                
            elif gauge == 1:
                U = params[:R_target*dim_u].reshape(R_target, dim_u)
                V_rest = params[R_target*dim_u:R_target*dim_u+R_target*(dim_v-1)].reshape(R_target, dim_v-1)
                W = params[R_target*dim_u+R_target*(dim_v-1):].reshape(R_target, dim_w)
                
                V = np.zeros((R_target, dim_v))
                V[:, 0] = 1.0
                V[:, 1:] = V_rest
                
            else:
                U = params[:R_target*dim_u].reshape(R_target, dim_u)
                V = params[R_target*dim_u:R_target*dim_u+R_target*dim_v].reshape(R_target, dim_v)
                W_rest = params[R_target*dim_u+R_target*dim_v:R_target*dim_u+R_target*dim_v+R_target*(dim_w-1)].reshape(R_target, dim_w-1)
                
                W = np.zeros((R_target, dim_w))
                W[:, 0] = 1.0
                W[:, 1:] = W_rest
                
            # Compute residuals for all equations
            # val = sum_r U[:, u] * V[:, v] * W[:, w]
            # We can vectorize this using einsum or broadcasting
            # U shape (R, dim_u), V shape (R, dim_v), W shape (R, dim_w)
            # eqs_u, eqs_v, eqs_w are arrays of indices
            
            # U_sel = U[:, eqs_u] -> (R, num_eqs)
            # V_sel = V[:, eqs_v] -> (R, num_eqs)
            # W_sel = W[:, eqs_w] -> (R, num_eqs)
            # prod = U_sel * V_sel * W_sel -> (R, num_eqs)
            # sum_r = prod.sum(axis=0) -> (num_eqs,)
            
            U_sel = U[:, eqs_u]
            V_sel = V[:, eqs_v]
            W_sel = W[:, eqs_w]
            
            sums = np.sum(U_sel * V_sel * W_sel, axis=0)
            
            return sums - eqs_target
        return objective

    # Try to find a valid solution
    for gauge in range(3):
        if time.time() > deadline - 15:
            break
            
        objective = make_objective(gauge)
        
        if gauge == 0:
            num_vars = R_target * (dim_u - 1 + dim_v + dim_w)
        elif gauge == 1:
            num_vars = R_target * (dim_u + dim_v - 1 + dim_w)
        else:
            num_vars = R_target * (dim_u + dim_v + dim_w - 1)
            
        max_restarts = 5
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            x0 = np.random.uniform(-1, 1, num_vars)
            
            try:
                from scipy.optimize import least_squares
                result = least_squares(objective, x0, method='lm', max_nfev=5000)
                
                if result.cost < 1e-10:
                    if gauge == 0:
                        U_rest = result.x[:R_target*(dim_u-1)].reshape(R_target, dim_u-1)
                        V = result.x[R_target*(dim_u-1):R_target*(dim_u-1+dim_v)].reshape(R_target, dim_v)
                        W = result.x[R_target*(dim_u-1+dim_v):].reshape(R_target, dim_w)
                        
                        U = np.zeros((R_target, dim_u))
                        U[:, 0] = 1.0
                        U[:, 1:] = U_rest
                        
                    elif gauge == 1:
                        U = result.x[:R_target*dim_u].reshape(R_target, dim_u)
                        V_rest = result.x[R_target*dim_u:R_target*dim_u+R_target*(dim_v-1)].reshape(R_target, dim_v-1)
                        W = result.x[R_target*dim_u+R_target*(dim_v-1):].reshape(R_target, dim_w)
                        
                        V = np.zeros((R_target, dim_v))
                        V[:, 0] = 1.0
                        V[:, 1:] = V_rest
                        
                    else:
                        U = result.x[:R_target*dim_u].reshape(R_target, dim_u)
                        V = result.x[R_target*dim_u:R_target*dim_u+R_target*dim_v].reshape(R_target, dim_v)
                        W_rest = result.x[R_target*dim_u+R_target*dim_v:R_target*dim_u+R_target*dim_v+R_target*(dim_w-1)].reshape(R_target, dim_w-1)
                        
                        W = np.zeros((R_target, dim_w))
                        W[:, 0] = 1.0
                        W[:, 1:] = W_rest
                        
                    U_int = np.round(2 * U)
                    V_int = np.round(2 * V)
                    W_int = np.round(2 * W)
                    
                    valid = True
                    # Verify all equations
                    U_sel = U_int[:, eqs_u]
                    V_sel = V_int[:, eqs_v]
                    W_sel = W_int[:, eqs_w]
                    sums = np.sum(U_sel * V_sel * W_sel, axis=0)
                    
                    # Target is 1 or 0, scaled by 2*2*2 = 8
                    target_scaled = eqs_target * 8
                    if not np.allclose(sums, target_scaled, atol=1e-6):
                        valid = False
                    
                    if valid:
                        U_final = (U_int / 2.0).tolist()
                        V_final = (V_int / 2.0).tolist()
                        W_final = (W_int / 2.0).tolist()
                        return {"U": U_final, "V": V_final, "W": W_final}
                        
            except Exception:
                continue

    # Fallback to schoolbook rank 45
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
