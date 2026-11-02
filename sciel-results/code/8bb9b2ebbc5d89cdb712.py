import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n, m, p = DIMENSIONS
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Precompute indices for efficient residual computation
    num_terms = n * m * p
    
    def residual_fn(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R))  # (n*m, R)
        V = V_flat.reshape((m * p, R))  # (m*p, R)
        W = W_flat.reshape((p * n, R))  # (p*n, R)
        
        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Target is 1.0 for all (i,j,k)
        residual = np.empty(num_terms)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    residual[idx] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - 1.0
                    idx += 1
                    
        return residual

    def schoolbook_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = None
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(12, 28, 21) = 28
    # Known results suggest rank around 55-65 for 3x4x7
    # Start from a reasonable lower bound and search upward
    start_rank = max(30, int(0.8 * n * m * p))
    for R in range(start_rank, n*m*p):
        if time.time() > deadline:
            break
            
        num_restarts = 8 if R < 65 else 4
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            if restart == 0:
                # Initialization: Random linear combination of schoolbook basis
                C_U = np.random.randn(num_basis, R) * 0.5
                C_V = np.random.randn(num_basis, R) * 0.5
                C_W = np.random.randn(num_basis, R) * 0.5
                
                U_init = U_basis.T @ C_U
                V_init = V_basis.T @ C_V
                W_init = W_basis.T @ C_W
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            elif restart == 1:
                # Try structured initialization with sparse patterns
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                
                # Assign random non-zero entries
                for r in range(R):
                    # Pick random positions for each factor
                    u_pos = np.random.randint(0, n*m, size=max(1, R//4))
                    v_pos = np.random.randint(0, m*p, size=max(1, R//4))
                    w_pos = np.random.randint(0, p*n, size=max(1, R//4))
                    
                    U_init[u_pos, r] = np.random.choice([-1, 1])
                    V_init[v_pos, r] = np.random.choice([-1, 1])
                    W_init[w_pos, r] = np.random.choice([-1, 1])
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                x0 = np.random.randn(total_size) * 0.3
            
            try:
                res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=300)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                # Try snapping to half-integers
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                res_snapped = residual_fn(x_snapped, R)
                
                if np.max(np.abs(res_snapped)) < 1e-6:
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((n*m, R)).tolist(),
                        "V": V_snap.reshape((m*p, R)).tolist(),
                        "W": W_snap.reshape((p*n, R)).tolist()
                    }
                    break
        if best_solution:
            break

    if best_solution:
        return best_solution
    else:
        # Try one more time with the best rank we got close to
        if best_rank < n * m * p and time.time() < deadline - 10:
            R = best_rank
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            total_size = size_U + size_V + size_W
            
            for restart in range(5):
                if time.time() > deadline:
                    break
                
                C_U = np.random.randn(num_basis, R) * 0.5
                C_V = np.random.randn(num_basis, R) * 0.5
                C_W = np.random.randn(num_basis, R) * 0.5
                
                U_init = U_basis.T @ C_U
                V_init = V_basis.T @ C_V
                W_init = W_basis.T @ C_W
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    res = least_squares(residual_fn, x0, args=(R,), method='lm', max_nfev=500)
                except Exception:
                    continue
                    
                if res.cost < 1e-8:
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U + size_V]
                    W_flat = res.x[size_U + size_V:]
                    
                    U_snap = np.round(U_flat * 2) / 2
                    V_snap = np.round(V_flat * 2) / 2
                    W_snap = np.round(W_flat * 2) / 2
                    
                    x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                    res_snapped = residual_fn(x_snapped, R)
                    
                    if np.max(np.abs(res_snapped)) < 1e-6:
                        best_solution = {
                            "U": U_snap.reshape((n*m, R)).tolist(),
                            "V": V_snap.reshape((m*p, R)).tolist(),
                            "W": W_snap.reshape((p*n, R)).tolist()
                        }
                        break
        
        if best_solution:
            return best_solution
        else:
            return schoolbook_solution()
