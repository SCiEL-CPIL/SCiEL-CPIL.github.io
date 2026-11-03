import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)
    
    # We try ranks starting from the lower bound upwards
    # Given time constraints, we try a few small ranks
    max_attempts = 5
    base_rank = min_rank
    
    for rank_offset in range(0, 10): # Try ranks from min_rank to min_rank + 9
        R = base_rank + rank_offset
        
        # Precompute index mappings for efficient tensor evaluation
        # We need to evaluate: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for all i in 0..n-1, j in 0..m-1, k in 0..p-1
        
        # Create index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        # Precompute linear indices
        U_idx = i_idx * m + j_idx  # shape (n*m*p,)
        V_idx = j_idx * p + k_idx  # shape (n*m*p,)
        W_idx = k_idx * n + i_idx  # shape (n*m*p,)
        
        def objective(params):
            # params is a flat vector of length (n*m + m*p + p*n) * R
            U_flat = params[:n * m * R]
            V_flat = params[n * m * R : n * m * R + m * p * R]
            W_flat = params[n * m * R + m * p * R :]
            
            U = U_flat.reshape(n * m, R)
            V = V_flat.reshape(m * p, R)
            W = W_flat.reshape(p * n, R)
            
            # Compute tensor values: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This can be vectorized using einsum or manual broadcasting
            # Let's use: for each r, U[:,r] * V[:,r] * W[:,r] contributes to the tensor
            # But the indices are coupled.
            
            # Alternative: iterate over r and accumulate
            tensor_vals = np.zeros(n * m * p)
            for r in range(R):
                u_r = U[:, r]  # shape (n*m,)
                v_r = V[:, r]  # shape (m*p,)
                w_r = W[:, r]  # shape (p*n,)
                
                # For each (i,j,k), add u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                # This is still expensive if done naively.
                # Let's use the precomputed indices
                tensor_vals += u_r[U_idx] * v_r[V_idx] * w_r[W_idx]
            
            # Target is all 1s
            err = np.sum((tensor_vals - 1.0) ** 2)
            return err
        
        # Initialize with random values
        np.random.seed(42 + rank_offset)
        initial_params = np.random.randn((n * m + m * p + p * n) * R)
        
        # Use gradient descent with scipy
        # We'll use L-BFGS-B for efficiency
        result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        
        if result.success or result.fun < 1e-6:
            best_params = result.x
            best_error = result.fun
        else:
            # Try a few more random restarts if first attempt fails
            best_error = float('inf')
            best_params = None
            for restart in range(5):
                np.random.seed(42 + rank_offset + restart * 100)
                init = np.random.randn((n * m + m * p + p * n) * R)
                res = minimize(objective, init, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
            
            if best_error >= 1e-6:
                continue  # Try next rank
        
        # Extract U, V, W from best_params
        U_flat = best_params[:n * m * R]
        V_flat = best_params[n * m * R : n * m * R + m * p * R]
        W_flat = best_params[n * m * R + m * p * R :]
        
        U = U_flat.reshape(n * m, R)
        V = V_flat.reshape(m * p, R)
        W = W_flat.reshape(p * n, R)
        
        # Round to half-integers
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify exactness
        is_exact = True
        for r in range(R):
            pass  # We'll check the full tensor
        
        tensor_vals = np.zeros(n * m * p)
        for r in range(R):
            u_r = U_int[:, r]
            v_r = V_int[:, r]
            w_r = W_int[:, r]
            tensor_vals += u_r[U_idx] * v_r[V_idx] * w_r[W_idx]
        
        if np.all(np.abs(tensor_vals - 1.0) < 1e-9):
            # Convert to lists of lists
            U_list = U_int.tolist()
            V_list = V_int.tolist()
            W_list = W_int.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not exact, try next rank

    # Fallback to schoolbook if no low-rank solution found
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
# EVOLVE-BLOCK-END
