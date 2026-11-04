import scipy
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    n, m, p = 3, 3, 6
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct target tensor T_target
    # T_target[a, b, c] = 1 if a,b,c form a valid multiplication term (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Precompute the non-zero indices for faster residual computation
    nonzero_idx = np.argwhere(T_target == 1.0)
    nz_a = nonzero_idx[:, 0]
    nz_b = nonzero_idx[:, 1]
    nz_c = nonzero_idx[:, 2]
    
    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        # Check exactness
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        # Check if R matches T_target exactly
        if np.allclose(R, T_target, atol=1e-9):
            # Convert to lists
            U_list = U_s.tolist()
            V_list = V_s.tolist()
            W_list = W_s.tolist()
            return True, U_list, V_list, W_list
        return False, None, None, None

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def structured_init(rank, seed):
        """Generate structured random initialization based on Strassen-like patterns."""
        rng = np.random.RandomState(seed)
        
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        # For each rank component, assign random coefficients to a subset of entries
        for r in range(rank):
            # U: assign to n*m entries
            u_indices = rng.choice(dim_U, size=n, replace=False)
            for idx in u_indices:
                U[idx, r] = rng.choice([-1.0, 0.5, 1.0, 1.5])
            
            # V: assign to m*p entries  
            v_indices = rng.choice(dim_V, size=m, replace=False)
            for idx in v_indices:
                V[idx, r] = rng.choice([-1.0, 0.5, 1.0, 1.5])
            
            # W: assign to p*n entries
            w_indices = rng.choice(dim_W, size=p, replace=False)
            for idx in w_indices:
                W[idx, r] = rng.choice([-1.0, 0.5, 1.0, 1.5])
        
        return U, V, W

    def random_init(rank, seed):
        """Pure random initialization."""
        rng = np.random.RandomState(seed)
        U = rng.randn(dim_U, rank) * 0.3
        V = rng.randn(dim_V, rank) * 0.3
        W = rng.randn(dim_W, rank) * 0.3
        return U, V, W

    def objective(params, U, V, W):
        """Compute residual using precomputed nonzero indices for speed."""
        U_p = params[:dim_U*rank].reshape(dim_U, rank)
        V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
        
        # Compute R at nonzero positions only
        R_nz = np.einsum('ir,jr,kr->r', U_p[nz_a, :], V_p[nz_b, :], W_p[nz_c, :])
        R_nz = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)[nz_a, nz_b, nz_c]
        
        # Residual: R should be 1 at nonzero positions, 0 elsewhere
        # We only check nonzero positions for speed
        residual = R_nz - 1.0
        return np.sum(residual**2)

    def polish(U, V, W, max_iter=500):
        """Polish the solution with tighter tolerances."""
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def obj_polish(params):
            U_p = params[:dim_U*rank].reshape(dim_U, rank)
            V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
            W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
            R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            return np.sum((R - T_target)**2)
        
        try:
            res = scipy.optimize.minimize(
                obj_polish, params, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
            V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
            W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
            return U_opt, V_opt, W_opt
        except Exception:
            return U, V, W

    # Try ranks from 46 upwards
    for rank in range(46, 54):
        if time.time() > deadline - 20: break
        
        # Try multiple initializations with different strategies
        num_structured = 8 if rank <= 48 else 5
        num_random = 5 if rank <= 48 else 3
        
        # Structured initializations
        for attempt in range(num_structured):
            if time.time() > deadline - 10: break
            
            U, V, W = structured_init(rank, seed=1000 + rank * 100 + attempt)
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params):
                U_p = params[:dim_U*rank].reshape(dim_U, rank)
                V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                return np.sum((R - T_target)**2)
            
            try:
                res = scipy.optimize.minimize(
                    obj, params, method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                
                # Polish
                U_opt, V_opt, W_opt = polish(U_opt, V_opt, W_opt, max_iter=300)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
                    
            except Exception:
                pass
        
        # Random initializations
        for attempt in range(num_random):
            if time.time() > deadline - 10: break
            
            U, V, W = random_init(rank, seed=5000 + rank * 100 + attempt)
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params):
                U_p = params[:dim_U*rank].reshape(dim_U, rank)
                V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                return np.sum((R - T_target)**2)
            
            try:
                res = scipy.optimize.minimize(
                    obj, params, method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                
                # Polish
                U_opt, V_opt, W_opt = polish(U_opt, V_opt, W_opt, max_iter=300)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
                    
            except Exception:
                pass

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
