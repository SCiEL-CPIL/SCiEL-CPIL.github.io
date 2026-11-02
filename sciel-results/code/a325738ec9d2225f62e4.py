import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140
    
    # Base rank 28 solution
    U28, V28, W28 = _build_rank28_arrays(n, m, p)
    
    # Try to find lower rank
    for rank in [25, 26, 27]:
        if time.time() > deadline - 20:
            break
        result = _search_rank(n, m, p, rank, deadline - 20)
        if result is not None:
            return _to_lists(result[0], result[1], result[2])
            
    return _to_lists(U28, V28, W28)

def _build_rank28_arrays(n, m, p):
    # Same as before
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Schoolbook for j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Strassen for j=1, k=0,1
    col = _add_strassen(U, V, W, n, m, p, 1, 0, col)
    
    # Schoolbook for j=1,2 k=2
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
            
    # Strassen for j=1, k=3,4
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

def _to_lists(U, V, W):
    U_list = [[int(round(val)) for val in row] for row in U.tolist()]
    V_list = [[int(round(val)) for val in row] for row in V.tolist()]
    W_list = [[int(round(val)) for val in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}

def _search_rank(n, m, p, rank, deadline):
    # Random search for rank decomposition
    num_constraints = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_params = rank * (dim_u + dim_v + dim_w)
    
    def residuals(params):
        U = params[:rank*dim_u].reshape(rank, dim_u).T
        V = params[rank*dim_u:rank*dim_u+rank*dim_v].reshape(rank, dim_v).T
        W = params[rank*dim_u+rank*dim_v:].reshape(rank, dim_w).T
        
        res = np.zeros(num_constraints)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    # Try multiple random initializations
    max_attempts = 5
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
            
        # Initialize with small random values
        np.random.seed(42 + attempt)
        params_init = np.random.normal(0, 1, total_params)
        
        try:
            result = least_squares(residuals, params_init, method='lm', max_nfev=5000)
            if result.success and np.max(np.abs(result.fun)) < 1e-5:
                U_sol = result.x[:rank*dim_u].reshape(rank, dim_u).T
                V_sol = result.x[rank*dim_u:rank*dim_u+rank*dim_v].reshape(rank, dim_v).T
                W_sol = result.x[rank*dim_u+rank*dim_v:].reshape(rank, dim_w).T
                return U_sol, V_sol, W_sol
        except Exception:
            pass
            
    return None
