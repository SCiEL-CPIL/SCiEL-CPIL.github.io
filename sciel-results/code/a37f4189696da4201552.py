import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n, m, p = 2, 3, 4
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Create the target tensor T*
    T_target = np.zeros((dim_u, dim_v, dim_w))
    schoolbook_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0
                schoolbook_indices.append((alpha, beta, gamma))
                
    # Store schoolbook basis vectors
    # U_sb is (6, 24), V_sb is (12, 24), W_sb is (8, 24)
    U_sb = np.zeros((dim_u, n*m*p))
    V_sb = np.zeros((dim_v, n*m*p))
    W_sb = np.zeros((dim_w, n*m*p))
    
    for idx, (alpha, beta, gamma) in enumerate(schoolbook_indices):
        U_sb[alpha, idx] = 1.0
        V_sb[beta, idx] = 1.0
        W_sb[gamma, idx] = 1.0
        
    def verify(U, V, W):
        """Check if U, V, W with integer/half-integer entries reconstruct T_target exactly."""
        # Check if entries are integer or half-integer
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
            
        # Snap to exact values
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Reconstruct
        T_decomp = np.zeros((dim_u, dim_v, dim_w))
        r = U.shape[1]
        for c in range(r):
            T_decomp += np.einsum('a,b,c->abc', U_snap[:, c], V_snap[:, c], W_snap[:, c])
            
        if np.allclose(T_decomp, T_target, atol=1e-9):
            return True
        return False

    def loss(params, r):
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

    def optimize_from_init(U_init, V_init, W_init, r):
        """Run L-BFGS-B starting from given initialization."""
        U_flat = U_init.flatten()
        V_flat = V_init.flatten()
        W_flat = W_init.flatten()
        params = np.concatenate([U_flat, V_flat, W_flat])
        
        try:
            res = minimize(loss, params, method='L-BFGS-B', jac=grad, args=(r,), 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e-6:
                # Extract and verify
                size_u = dim_u * r
                size_v = dim_v * r
                size_w = dim_w * r
                U = res.x[:size_u].reshape((dim_u, r))
                V = res.x[size_u:size_u+size_v].reshape((dim_v, r))
                W = res.x[size_u+size_v:size_u+size_v+size_w].reshape((dim_w, r))
                
                if verify(U, V, W):
                    return U, V, W
        except Exception:
            pass
        return None

    # Try ranks from 23 down to 18
    for r in range(23, 17, -1):
        if time.time() > deadline:
            break
            
        # Strategy 1: Subsets of schoolbook terms
        # If r == 24, it's the whole set.
        # If r < 24, we pick r terms out of 24.
        
        # For r=23, there are 24 subsets.
        # For r=22, there are 276 subsets.
        # We can limit the number of subsets to try to save time.
        
        num_subsets_to_try = 100
        if r == 23:
            subsets = list(itertools.combinations(range(24), r))
        elif r == 22:
            subsets = list(itertools.combinations(range(24), r))[:num_subsets_to_try]
        else:
            # For lower ranks, random subsets might be better, or just random init
            subsets = []
            
        for subset in subsets:
            if time.time() > deadline:
                break
            
            U_init = U_sb[:, list(subset)]
            V_init = V_sb[:, list(subset)]
            W_init = W_sb[:, list(subset)]
            
            result = optimize_from_init(U_init, V_init, W_init, r)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        # Strategy 2: Random initialization
        # If subsets didn't work, try random starts
        num_random_restarts = 50
        for restart in range(num_random_restarts):
            if time.time() > deadline:
                break
                
            U_init = np.random.randn(dim_u, r) * 0.5
            V_init = np.random.randn(dim_v, r) * 0.5
            W_init = np.random.randn(dim_w, r) * 0.5
            
            result = optimize_from_init(U_init, V_init, W_init, r)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
    r_schoolbook = 24
    U = U_sb
    V = V_sb
    W = W_sb
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
