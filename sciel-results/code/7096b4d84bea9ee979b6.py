import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, leave 5s margin

    n, m, p = 2, 3, 4
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8

    # Create the target tensor T*
    # T* is a 3D array of shape (dim_u, dim_v, dim_w)
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0

    def loss(params, r):
        """
        params: flattened array of U, V, W
        r: rank
        """
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:size_u+size_v+size_w]
        
        U = U_flat.reshape((dim_u, r))
        V = V_flat.reshape((dim_v, r))
        W = W_flat.reshape((dim_w, r))
        
        # Compute decomposed tensor
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            
        error = T_decomp - T_target
        return np.sum(error**2)
        
    def grad(params, r):
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:size_u+size_v+size_w]
        
        U = U_flat.reshape((dim_u, r))
        V = V_flat.reshape((dim_v, r))
        W = W_flat.reshape((dim_w, r))
        
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            
        error = T_decomp - T_target
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            grad_U[:, c] = 2.0 * np.einsum('abg,b,g->a', error, V[:, c], W[:, c])
            grad_V[:, c] = 2.0 * np.einsum('abg,a,g->b', error, U[:, c], W[:, c])
            grad_W[:, c] = 2.0 * np.einsum('abg,a,b->g', error, U[:, c], V[:, c])
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def snap_and_verify(U, V, W, r):
        """
        Snaps to half-integers and verifies if the decomposition is exact.
        Returns (U_snapped, V_snapped, W_snapped) if valid, else (None, None, None).
        """
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U_snap[:, c], V_snap[:, c], W_snap[:, c])
            
        if np.allclose(T_decomp, T_target, atol=1e-6):
            return U_snap, V_snap, W_snap
        return None, None, None

    # Try to find rank 23
    r_target = 23
    best_loss = np.inf
    best_params = None
    
    # Number of restarts depends on available time
    # Each restart takes some time. Let's allow up to 10 restarts if time permits.
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        # Scale 0.1 is reasonable
        params = np.random.randn(dim_u * r_target + dim_v * r_target + dim_w * r_target) * 0.1
        
        # Use L-BFGS-B
        try:
            res = minimize(loss, params, method='L-BFGS-B', jac=grad, args=(r_target,), 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
            if best_loss < 1e-6:
                # Extract and verify
                size_u = dim_u * r_target
                size_v = dim_v * r_target
                size_w = dim_w * r_target
                
                U = best_params[:size_u].reshape((dim_u, r_target))
                V = best_params[size_u:size_u+size_v].reshape((dim_v, r_target))
                W = best_params[size_u+size_v:size_u+size_v+size_w].reshape((dim_w, r_target))
                
                U_s, V_s, W_s = snap_and_verify(U, V, W, r_target)
                if U_s is not None:
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    
        except Exception as e:
            # If optimization fails, continue to next restart
            pass

    # Fallback to schoolbook rank 24
    r_schoolbook = 24
    U = np.zeros((dim_u, r_schoolbook))
    V = np.zeros((dim_v, r_schoolbook))
    W = np.zeros((dim_w, r_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
