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
    # Convert numpy arrays to lists of lists of floats/ints
    # The task asks for integer/half-integer entries.
    # We ensure they are exactly representable or close enough.
    def clean_matrix(M):
        M_clean = np.round(M * 2) / 2
        # Convert to list
        return M_clean.tolist()
    
    return {
        "U": clean_matrix(U),
        "V": clean_matrix(V),
        "W": clean_matrix(W)
    }

def _verify(U, V, W, n, m, p, tol=1e-9):
    # Check if the decomposition is exact
    # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # Target: 1 if (u,v,w) corresponds to (i,j,k) with u=im+j, v=jp+k, w=kn+i
    #         0 otherwise
    
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
    # Iterate over all u, v, w and check if they form a valid triple
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
    
    # Precompute indices as arrays for vectorization
    valid_u = np.array([u for u, v, w in valid_indices])
    valid_v = np.array([v for u, v, w in valid_indices])
    valid_w = np.array([w for u, v, w in valid_indices])
    
    invalid_u = np.array([u for u, v, w in invalid_indices])
    invalid_v = np.array([v for u, v, w in invalid_indices])
    invalid_w = np.array([w for u, v, w in invalid_indices])
    
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
        
        # Compute T_approx for all indices needed
        # We can compute the full tensor or just the slices.
        # Full tensor is 9*12*12 = 1296 elements. Very small.
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Valid indices
        T_valid = T_approx[valid_u, valid_v, valid_w]
        err_valid = T_valid - 1.0
        
        # Invalid indices
        T_invalid = T_approx[invalid_u, invalid_v, invalid_w]
        err_invalid = T_invalid
        
        loss = np.sum(err_valid**2) + np.sum(err_invalid**2)
        
        # Gradients
        # dL/dU[u, c] = 2 * sum_{(u,v,w) in valid} err * V[v,c] * W[w,c] + 2 * sum_{(u,v,w) in invalid} err * V[v,c] * W[w,c]
        # We can use matrix multiplication or einsum for this.
        
        # Let's compute gradients using the full tensor approach for simplicity and speed.
        # grad_U[u, r] = 2 * sum_{v,w} T_err[u,v,w] * V[v,r] * W[w,r]
        # where T_err is the error tensor (T - Target)
        
        # Target tensor
        Target = np.zeros((n*m, m*p, p*n))
        for (u, v, w) in valid_indices:
            Target[u, v, w] = 1.0
            
        T_err = T_approx - Target
        
        # grad_U[u, r] = 2 * sum_{v,w} T_err[u, v, w] * V[v, r] * W[w, r]
        # This is equivalent to:
        # grad_U = 2 * np.einsum('uvw,vr,wr->ur', T_err, V, W)
        
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', T_err, V, W)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', T_err, U, W)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', T_err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        if _verify(U_s, V_s, W, n, m, p, tol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    # Try ranks from lower to upper
    # Schoolbook is 36.
    # Let's try down to 25.
    # If time is tight, we might need to limit restarts.
    
    best_result = None
    best_rank = n * m * p
    
    # Initialize with schoolbook as fallback
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_result = _to_lists(U0, V0, W0)
    
    # Search for lower rank
    # Known optimal rank for 3x3x4 is 19.
    # We will search from 19 upwards.
    for target_rank in range(19, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        U_shape = (n * m, target_rank)
        V_shape = (m * p, target_rank)
        W_shape = (p * n, target_rank)
        
        found = False
        
        # Number of restarts: depends on time left
        # Vectorization allows for more restarts.
        # Let's try 50 restarts per rank, but break early if time is tight.
        num_restarts = 50
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Gaussian initialization often works better for L-BFGS than discrete choices
            U = np.random.randn(*U_shape) * 0.5
            V = np.random.randn(*V_shape) * 0.5
            W = np.random.randn(*W_shape) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape, target_rank), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-6})
            except Exception:
                continue
                
            if res.fun < 1e-5:
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
