"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _make_schoolbook(n, m, p):
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
    return U, V, W

def _to_lists(U, V, W):
    def clean_matrix(M):
        M_clean = np.round(M * 2) / 2
        return M_clean.tolist()
    return {
        "U": clean_matrix(U),
        "V": clean_matrix(V),
        "W": clean_matrix(W)
    }

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_set.add((u, v, w))
                if not np.isclose(T[u, v, w], 1.0, atol=tol):
                    return False
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in valid_set:
                    if not np.isclose(T[u, v, w], 0.0, atol=tol):
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    valid_indices = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.append((u, v, w))
                valid_set.add((u, v, w))
                
    invalid_indices = []
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in valid_set:
                    invalid_indices.append((u, v, w))
    
    Target = np.zeros((n*m, m*p, p*n))
    for (u, v, w) in valid_indices:
        Target[u, v, w] = 1.0

    def loss_and_grad(params, U_shape, V_shape, W_shape, r):
        num_u = U_shape[0] * r
        num_v = V_shape[0] * r
        num_w = W_shape[0] * r
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u+num_v]
        W_flat = params[num_u+num_v:num_u+num_v+num_w]
        
        U = U_flat.reshape(U_shape)
        V = V_flat.reshape(V_shape)
        W = W_flat.reshape(W_shape)
        
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        T_err = T_approx - Target
        
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', T_err, V, W)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', T_err, U, W)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', T_err, U, V)
        
        loss = np.sum(T_err**2)
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if _verify(U_s, V_s, W_s, n, m, p, tol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    best_result = None
    best_rank = n * m * p
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_result = _to_lists(U0, V0, W0)

    # Search ranks from 19 to 35
    for target_rank in range(19, 36):
        if time.time() > deadline - 10:
            break
            
        U_shape = (n * m, target_rank)
        V_shape = (m * p, target_rank)
        W_shape = (p * n, target_rank)
        
        found = False
        
        # More restarts for lower ranks? No, lower ranks are harder.
        # Let's use 20 restarts for 19, and 50 for others.
        num_restarts = 20 if target_rank == 19 else 50
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(*U_shape) * 0.5
            V = np.random.randn(*V_shape) * 0.5
            W = np.random.randn(*W_shape) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape, target_rank), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                num_u = U_shape[0] * target_rank
                num_v = V_shape[0] * target_rank
                num_w = W_shape[0] * target_rank
                U_final = res.x[:num_u].reshape(U_shape)
                V_final = res.x[num_u:num_u+num_v].reshape(V_shape)
                W_final = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                success, U_s, V_s, W_s = snap_and_check(U_final, V_final, W_final)
                if success:
                    best_rank = target_rank
                    best_result = _to_lists(U_s, V_s, W_s)
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
