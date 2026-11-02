import numpy as np
import time
import itertools
import scipy.optimize

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Precompute indices for the tensor
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Target: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triplets = len(triplets)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Precompute index arrays for vectorized operations
    # This allows us to compute the residuals for all triplets at once
    # U_rows: shape (num_triplets,)
    # V_rows: shape (num_triplets,)
    # W_rows: shape (num_triplets,)
    U_rows = np.array([u_idx(i, j) for (i, j, k) in triplets])
    V_rows = np.array([v_idx(j, k) for (i, j, k) in triplets])
    W_rows = np.array([w_idx(k, i) for (i, j, k) in triplets])
    
    def calculate_residuals(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_c U[U_rows, c] * V[V_rows, c] * W[W_rows, c]
        # This is not a standard matrix multiplication.
        # Let rank be R.
        # For each triplet t, we have vectors U_t, V_t, W_t of length R.
        # Residual_t = sum_c U_t[c] * V_t[c] * W_t[c] - 1
        
        # We can vectorize this by creating arrays of shape (num_triplets, R)
        U_sel = U[U_rows]  # (num_triplets, R)
        V_sel = V[V_rows]  # (num_triplets, R)
        W_sel = W[W_rows]  # (num_triplets, R)
        
        # Element-wise product and sum along axis 1
        products = U_sel * V_sel * W_sel
        sums = np.sum(products, axis=1)
        
        return sums - 1.0

    def objective(params, U_shape, V_shape, W_shape):
        U_size = U_shape[0] * U_shape[1]
        V_size = V_shape[0] * V_shape[1]
        W_size = W_shape[0] * W_shape[1]
        
        U = params[:U_size].reshape(U_shape)
        V = params[U_size:U_size+V_size].reshape(V_shape)
        W = params[U_size+V_size:].reshape(W_shape)
        
        res = calculate_residuals(U, V, W)
        return np.sum(res * res)

    def verify_exact(U, V, W):
        # Check if U, V, W are integer or half-integer and if the decomposition is exact
        # The problem states "integer/half-integer entries".
        # Let's check if 2*U, 2*V, 2*W are close to integers.
        if not np.allclose(2*U, np.round(2*U), atol=1e-9):
            return False
        if not np.allclose(2*V, np.round(2*V), atol=1e-9):
            return False
        if not np.allclose(2*W, np.round(2*W), atol=1e-9):
            return False
            
        # Snap to half-integers
        U_snap = np.round(2*U) / 2.0
        V_snap = np.round(2*V) / 2.0
        W_snap = np.round(2*W) / 2.0
        
        res = calculate_residuals(U_snap, V_snap, W_snap)
        if np.allclose(res, 0.0, atol=1e-9):
            return True
        return False

    start_time = time.time()
    deadline = time.time() + 115  # 5s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook solution (Rank 48)
    rank = n * m * p
    U_init = np.zeros((n*m, rank))
    V_init = np.zeros((m*p, rank))
    W_init = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[u_idx(i, j), col] = 1.0
                V_init[v_idx(j, k), col] = 1.0
                W_init[w_idx(k, i), col] = 1.0
                col += 1
                
    # Verify schoolbook is exact (it should be)
    if verify_exact(U_init, V_init, W_init):
        best_U = U_init
        best_V = V_init
        best_W = W_init
        best_rank = rank
        
    # Try to reduce rank
    for r in range(rank - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Initialize for rank r
        # Strategy: Take the first r columns of the best known solution? 
        # Or random projection?
        # Let's try a random initialization first, then refine.
        # But random init for high rank is hard.
        # Better: Start with the schoolbook solution, but zero out some columns?
        # Or use the solution from rank r+1 and project?
        
        # Let's try a simple approach: Random initialization with small noise
        # This is risky. Let's try to use the previous solution if available.
        
        if best_U is not None:
            # Project the previous solution to rank r
            # This is tricky because the basis changes.
            # Let's just try random restarts for now, but with a good initial guess.
            # A good guess is to take the top r singular vectors of the "tensor"?
            # That's complex.
            
            # Alternative: Use the schoolbook basis but combine terms.
            # Let's just try a few random initializations for each rank.
            num_restarts = 3
            if r > 40: num_restarts = 1 # Fewer restarts for high rank if time is tight? No, high rank is easier.
            
            for restart in range(num_restarts):
                if time.time() > deadline:
                    break
                    
                # Initialize
                # If we have a solution for r+1, we can try to use it.
                # For simplicity, let's use random init scaled appropriately.
                U_curr = np.random.randn(n*m, r) * 0.1
                V_curr = np.random.randn(m*p, r) * 0.1
                W_curr = np.random.randn(p*n, r) * 0.1
                
                # Add the schoolbook terms if r >= 48? No, r < 48.
                # Let's try initializing with the first r schoolbook terms and adding noise.
                # This is a structured initialization.
                U_curr = np.zeros((n*m, r))
                V_curr = np.zeros((m*p, r))
                W_curr = np.zeros((p*n, r))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= r:
                                break
                            U_curr[u_idx(i, j), col] = 1.0
                            V_curr[v_idx(j, k), col] = 1.0
                            W_curr[w_idx(k, i), col] = 1.0
                            col += 1
                        if col >= r:
                            break
                    if col >= r:
                        break
                if col >= r:
                    break
                    
                # Add small noise to escape local minima
                U_curr += 0.01 * np.random.randn(n*m, r)
                V_curr += 0.01 * np.random.randn(m*p, r)
                W_curr += 0.01 * np.random.randn(p*n, r)
                
                # Flatten for optimization
                params0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
                
                # Optimize
                try:
                    res = scipy.optimize.minimize(
                        objective, 
                        params0, 
                        args=(U_curr.shape, V_curr.shape, W_curr.shape),
                        method='BFGS',
                        options={'maxiter': 50, 'gtol': 1e-5} # Limit iterations for speed
                    )
                    
                    params_opt = res.x
                    U_size = (n*m) * r
                    V_size = (m*p) * r
                    W_size = (p*n) * r
                    
                    U_opt = params_opt[:U_size].reshape((n*m, r))
                    V_opt = params_opt[U_size:U_size+V_size].reshape((m*p, r))
                    W_opt = params_opt[U_size+V_size:].reshape((p*n, r))
                    
                    # Check error
                    err = objective(params_opt, U_opt.shape, V_opt.shape, W_opt.shape)
                    
                    if err < 1e-6:
                        # Try to snap to half-integers
                        if verify_exact(U_opt, V_opt, W_opt):
                            best_rank = r
                            best_U = np.round(2*U_opt) / 2.0
                            best_V = np.round(2*V_opt) / 2.0
                            best_W = np.round(2*W_opt) / 2.0
                            break # Move to next rank
                except Exception:
                    pass
        
        if time.time() > deadline:
            break
            
    # Return the best found
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook (should not happen as we init with it)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
