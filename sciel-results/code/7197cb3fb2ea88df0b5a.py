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
        # Ensure entries are half-integers
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
    
    # Compute full tensor
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Check valid indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if not np.isclose(T[u, v, w], 1.0, atol=tol):
                    return False
    
    # Check invalid indices
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_set.add((u, v, w))
                
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
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute valid and invalid indices
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

    # Convert to numpy arrays for faster indexing in loss function
    valid_arr = np.array(valid_indices, dtype=int)
    invalid_arr = np.array(invalid_indices, dtype=int)
    u_v, v_v, w_v = valid_arr[:, 0], valid_arr[:, 1], valid_arr[:, 2]
    u_i, v_i, w_i = invalid_arr[:, 0], invalid_arr[:, 1], invalid_arr[:, 2]

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
        
        # Compute T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Vectorized loss and gradient for valid indices
        # T_val = T_approx[u_v, v_v, w_v]
        T_val = T_approx[u_v, v_v, w_v]
        err_val = T_val - 1.0
        loss += np.sum(err_val ** 2)
        
        # Gradient accumulation for valid indices
        # dL/dU[u,c] += sum_{(u,v,w) in valid} 2 * err(u,v,w) * V[v,c] * W[w,c]
        # This is equivalent to:
        # grad_U[u_v, :] += 2 * err_val[:, None] * V[v_v, :][u_v, :] * W[w_v, :][u_v, :]
        # Note: V[v_v, :] is a (N_valid, r) matrix where row i corresponds to V[v_v[i], :]
        # We need to multiply element-wise with U indices.
        
        V_val = V[v_v, :] # (N_valid, r)
        W_val = W[w_v, :] # (N_valid, r)
        
        # For each valid index i, we add 2 * err[i] * V_val[i] * W_val[i] to grad_U[u_v[i]]
        # We can use np.add.at or scatter. Since indices might repeat?
        # u_v = i*m+j. For a fixed u, there are multiple (v,w) pairs?
        # Yes. So we need to accumulate.
        # np.einsum('i,ir,is->us', err_val, V_val, W_val) doesn't work directly for scatter.
        # Let's use a loop if N_valid is small, or scatter.
        # N_valid = 36. Loop is fine and safer for accumulation if indices repeat.
        # But vectorized scatter is faster.
        # Let's stick to the loop for clarity and correctness with accumulation, 
        # or use np.add.at.
        
        # Actually, let's just use the loop for gradients as N is small (36+1260).
        # 1296 iterations is trivial.
        
        # Reverting to loop for gradients to ensure correctness of accumulation
        for idx in range(len(valid_indices)):
            u, v, w = valid_indices[idx]
            e = err_val[idx]
            if e != 0.0: # Optimization
                grad_U[u, :] += 2.0 * e * V[v, :] * W[w, :]
                grad_V[v, :] += 2.0 * e * U[u, :] * W[w, :]
                grad_W[w, :] += 2.0 * e * U[u, :] * V[v, :]
        
        # Vectorized loss for invalid indices
        T_inv = T_approx[u_i, v_i, w_i]
        err_inv = T_inv
        loss += np.sum(err_inv ** 2)
        
        # Gradients for invalid indices
        for idx in range(len(invalid_indices)):
            u, v, w = invalid_indices[idx]
            e = err_inv[idx]
            if e != 0.0:
                grad_U[u, :] += 2.0 * e * V[v, :] * W[w, :]
                grad_V[v, :] += 2.0 * e * U[u, :] * W[w, :]
                grad_W[w, :] += 2.0 * e * U[u, :] * V[v, :]
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        if _verify(U_s, V_s, W_s, n, m, p, tol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    # Strategy:
    # 1. Try known lower bounds (25-30) with high restarts.
    # 2. If not found, try 31-35.
    # 3. Fallback to schoolbook (36).
    
    best_result = None
    best_rank = n * m * p
    
    # Initialize with schoolbook as fallback
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_result = _to_lists(U0, V0, W0)
    
    # Search for lower rank
    # We prioritize lower ranks.
    for target_rank in range(25, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        U_shape = (n * m, target_rank)
        V_shape = (m * p, target_rank)
        W_shape = (p * n, target_rank)
        
        found = False
        
        # Number of restarts: depends on time left
        # Estimate time per restart.
        # Let's do 20 restarts per rank, checking time.
        num_restarts = 20
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initialization
            # Using {-1, 0, 1} is often better for tensor decomposition sparsity
            U = np.random.choice([-1.0, 0.0, 1.0], size=U_shape)
            V = np.random.choice([-1.0, 0.0, 1.0], size=V_shape)
            W = np.random.choice([-1.0, 0.0, 1.0], size=W_shape)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape, target_rank), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Extract solution
                num_u = U_shape[0] * target_rank
                num_v = V_shape[0] * target_rank
                num_w = W_shape[0] * target_rank
                U_final = res.x[:num_u].reshape(U_shape)
                V_final = res.x[num_u:num_u+num_v].reshape(V_shape)
                W_final = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                # Try to snap
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
