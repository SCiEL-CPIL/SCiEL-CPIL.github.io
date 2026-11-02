import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Build the known rank 28 solution
    U28, V28, W28 = _build_rank28_arrays(n, m, p)
    
    # 2. Try to find rank 27
    if time.time() < deadline - 5:
        result = _search_rank(n, m, p, 27, U28, V28, W28, deadline - 5)
        if result is not None:
            return _to_lists(result[0], result[1], result[2])
            
    # 3. Try to find rank 26 (only if time permits and 27 failed)
    if time.time() < deadline - 2:
        result = _search_rank(n, m, p, 26, U28, V28, W28, deadline - 2)
        if result is not None:
            return _to_lists(result[0], result[1], result[2])

    # 4. Fallback to rank 28
    return _to_lists(U28, V28, W28)

def _build_rank28_arrays(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Block 1: j=0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Block 2a: j=1,2, k=0,1, Strassen
    col = _add_strassen(U, V, W, n, m, p, 1, 0, col)
    
    # Block 2b: j=1,2, k=2, schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
            
    # Block 2c: j=1,2, k=3,4, Strassen
    col = _add_strassen(U, V, W, n, m, p, 1, 3, col)
    
    return U, V, W

def _add_strassen(U, V, W, n, m, p, base_j, base_k, col):
    # Same as before
    U[0 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    U[1 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    W[base_k * n + 1, col] = 1
    W[(base_k + 1) * n + 1, col] = -1
    col += 1
    U[0 * m + base_j, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    U[1 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + base_k, col] = 1
    V[base_j * p + base_k, col] = -1
    W[base_k * n + 0, col] = 1
    W[base_k * n + 1, col] = 1
    col += 1
    U[0 * m + base_j, col] = 1
    U[0 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    col += 1
    U[1 * m + base_j, col] = 1
    U[0 * m + base_j, col] = -1
    V[base_j * p + base_k, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    U[0 * m + (base_j + 1), col] = 1
    U[1 * m + (base_j + 1), col] = -1
    V[(base_j + 1) * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    col += 1
    return col

def _search_rank(n, m, p, target_rank, U_ref, V_ref, W_ref, deadline):
    num_constraints = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_params = target_rank * (dim_u + dim_v + dim_w)
    
    def residuals(params):
        U = params[:target_rank*dim_u].reshape(target_rank, dim_u).T
        V = params[target_rank*dim_u:target_rank*dim_u+target_rank*dim_v].reshape(target_rank, dim_v).T
        W = params[target_rank*dim_u+target_rank*dim_v:].reshape(target_rank, dim_w).T
        
        res = np.zeros(num_constraints)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    # Strategy 1: Random initializations
    # Strategy 2: Initializations derived from the reference solution (drop columns)
    
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        if attempt < 10:
            # Random normal
            np.random.seed(42 + attempt)
            params_init = np.random.normal(0, 1, total_params)
        else:
            # Derived from reference
            # Pick target_rank columns from the reference (28 cols)
            # We need to map the 28-col solution to target_rank params.
            # This is tricky because the reference is 28, target is 27.
            # We can try dropping one random column from the reference and using the rest as init.
            # But the reference is a valid solution for rank 28. Dropping a column makes it invalid.
            # The optimizer will try to fix it.
            
            # Flatten reference
            U_ref_flat = U_ref.T.flatten() # 28 * dim_u
            V_ref_flat = V_ref.T.flatten()
            W_ref_flat = W_ref.T.flatten()
            
            # Select target_rank columns
            cols_to_keep = np.random.choice(28, target_rank, replace=False)
            
            # Reconstruct params
            # U part: rows are cols of U_ref? No, U_ref is (dim_u, 28).
            # We want U_init (dim_u, target_rank).
            # So we take U_ref[:, cols_to_keep]
            U_init = U_ref[:, cols_to_keep]
            V_init = V_ref[:, cols_to_keep]
            W_init = W_ref[:, cols_to_keep]
            
            params_init = np.concatenate([
                U_init.T.flatten(),
                V_init.T.flatten(),
                W_init.T.flatten()
            ])
            
        try:
            result = least_squares(residuals, params_init, method='lm', max_nfev=5000)
            if result.success and np.max(np.abs(result.fun)) < 1e-5:
                U_sol = result.x[:target_rank*dim_u].reshape(target_rank, dim_u).T
                V_sol = result.x[target_rank*dim_u:target_rank*dim_u+target_rank*dim_v].reshape(target_rank, dim_v).T
                W_sol = result.x[target_rank*dim_u+target_rank*dim_v:].reshape(target_rank, dim_w).T
                return U_sol, V_sol, W_sol
        except Exception:
            pass
            
    return None

def _to_lists(U, V, W):
    U_list = [[int(round(val)) for val in row] for row in U.tolist()]
    V_list = [[int(round(val)) for val in row] for row in V.tolist()]
    W_list = [[int(round(val)) for val in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
