import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140
    
    U28, V28, W28 = _build_rank28_arrays(n, m, p)
    
    result = None
    if time.time() < deadline:
        result = _search_rank27(n, m, p, deadline)
    
    if result is None:
        return _to_lists(U28, V28, W28)
    else:
        U, V, W = result
        return _to_lists(U, V, W)

def _build_rank28_arrays(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    col = _add_strassen(U, V, W, n, m, p, 1, 0, col)
    
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    col = _add_strassen(U, V, W, n, m, p, 1, 3, col)
    
    return U, V, W

def _add_strassen(U, V, W, n, m, p, base_j, base_k, col):
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

def _search_rank27(n, m, p, deadline):
    """Search for rank-27 decomposition."""
    # Strategy: fix 14 Strassen terms, find 13 terms for remaining constraints
    # The remaining constraints are:
    # - j=0, all k: 10 constraints
    # - j∈{1,2}, k=2: 4 constraints
    # Total: 14 constraints
    
    # Parameterize 13 terms with restricted entries
    # U entries per term: U[i*3+0, r] for i=0,1 (2), U[i*3+1, r] for i=0,1 (2), U[i*3+2, r] for i=0,1 (2) = 6
    # V entries per term: V[0*5+k, r] for k=0..4 (5), V[1*5+2, r] (1), V[2*5+2, r] (1) = 7
    # W entries per term: W[k*2+i, r] for all k=0..4, i=0,1 (10)
    # Total per term: 23, for 13 terms: 299 parameters
    
    # This is a large nonlinear problem. Let me try with a smaller parameterization.
    # Actually, let me just try it and see if scipy can handle it.
    
    num_terms = 13
    # Define the constraint indices
    # j=0 constraints: (i, j, k) = (i, 0, k) for i in {0,1}, k in {0..4}
    # k=2, j∈{1,2} constraints: (i, j, k) = (i, j, 2) for i in {0,1}, j in {1,2}
    
    constraint_indices = []
    for i in range(n):
        for k in range(p):
            constraint_indices.append((i, 0, k))
    for i in range(n):
        for j in [1, 2]:
            constraint_indices.append((i, j, 2))
    
    # Build the Strassen terms (fixed)
    U_strassen = np.zeros((n * m, 14))
    V_strassen = np.zeros((m * p, 14))
    W_strassen = np.zeros((p * n, 14))
    
    col = 0
    col = _add_strassen(U_strassen, V_strassen, W_strassen, n, m, p, 1, 0, col)
    col = _add_strassen(U_strassen, V_strassen, W_strassen, n, m, p, 1, 3, col)
    
    # Now define the residual function
    def residuals(params):
        # params: 13 × 23 = 299 values
        # First 13*6 = 78 are U entries
        # Next 13*7 = 91 are V entries
        # Last 13*10 = 130 are W entries
        
        U_new = params[:13*6].reshape(13, 6)
        V_new = params[13*6:13*6+13*7].reshape(13, 7)
        W_new = params[13*6+13*7:].reshape(13, 10)
        
        # Build full U, V, W for the 13 new terms
        U_full = np.zeros((n * m, 13))
        V_full = np.zeros((m * p, 13))
        W_full = np.zeros((p * n, 13))
        
        for r in range(13):
            # U entries: U[i*3+j, r]
            U_full[0*3+0, r] = U_new[r, 0]
            U_full[1*3+0, r] = U_new[r, 1]
            U_full[0*3+1, r] = U_new[r, 2]
            U_full[1*3+1, r] = U_new[r, 3]
            U_full[0*3+2, r] = U_new[r, 4]
            U_full[1*3+2, r] = U_new[r, 5]
            
            # V entries: V[j*5+k, r]
            V_full[0*5+0, r] = V_new[r, 0]
            V_full[0*5+1, r] = V_new[r, 1]
            V_full[0*5+2, r] = V_new[r, 2]
            V_full[0*5+3, r] = V_new[r, 3]
            V_full[0*5+4, r] = V_new[r, 4]
            V_full[1*5+2, r] = V_new[r, 5]
            V_full[2*5+2, r] = V_new[r, 6]
            
            # W entries: W[k*2+i, r]
            for k in range(5):
                for i in range(2):
                    W_full[k*2+i, r] = W_new[r, k*2+i]
        
        # Compute residuals for each constraint
        res = np.zeros(len(constraint_indices))
        for idx, (i, j, k) in enumerate(constraint_indices):
            # T[i,j,k] = sum_r U[i*3+j, r] * V[j*5+k, r] * W[k*2+i, r]
            # For Strassen terms:
            val_strassen = np.sum(U_strassen[i*3+j, :] * V_strassen[j*5+k, :] * W_strassen[k*2+i, :])
            # For new terms:
            val_new = np.sum(U_full[i*3+j, :] * V_full[j*5+k, :] * W_full[k*2+i, :])
            res[idx] = val_strassen + val_new - 1.0
        
        return res
    
    # Initial guess: use the first 13 of the 14 non-Strassen terms from rank-28
    # and set the 14th to zero
    U_init = np.zeros((n * m, 13))
    V_init = np.zeros((m * p, 13))
    W_init = np.zeros((p * n, 13))
    
    # Use schoolbook for j=0 and k=2 as initial guess
    col = 0
    for i in range(n):
        for k in range(p):
            if col >= 13:
                break
            U_init[i * m + 0, col] = 1
            V_init[0 * p + k, col] = 1
            W_init[k * n + i, col] = 1
            col += 1
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            if col >= 13:
                break
            U_init[i * m + j, col] = 1
            V_init[j * p + 2, col] = 1
            W_init[2 * n + i, col] = 1
            col += 1
    
    # Flatten to params
    params_init = np.zeros(299)
    for r in range(13):
        params_init[r*6+0] = U_init[0*3+0, r]
        params_init[r*6+1] = U_init[1*3+0, r]
        params_init[r*6+2] = U_init[0*3+1, r]
        params_init[r*6+3] = U_init[1*3+1, r]
        params_init[r*6+4] = U_init[0*3+2, r]
        params_init[r*6+5] = U_init[1*3+2, r]
        params_init[13*6+r*7+0] = V_init[0*5+0, r]
        params_init[13*6+r*7+1] = V_init[0*5+1, r]
        params_init[13*6+r*7+2] = V_init[0*5+2, r]
        params_init[13*6+r*7+3] = V_init[0*5+3, r]
        params_init[13*6+r*7+4] = V_init[0*5+4, r]
        params_init[13*6+r*7+5] = V_init[1*5+2, r]
        params_init[13*6+r*7+6] = V_init[2*5+2, r]
        for k in range(5):
            for i in range(2):
                params_init[13*6+13*7+r*10+k*2+i] = W_init[k*2+i, r]
    
    try:
        result = least_squares(residuals, params_init, method='lm', max_nfev=10000)
        if result.success and np.max(np.abs(result.fun)) < 1e-6:
            # Extract solution
            U_sol = np.zeros((n * m, 13))
            V_sol = np.zeros((m * p, 13))
            W_sol = np.zeros((p * n, 13))
            
            U_new = result.x[:13*6].reshape(13, 6)
            V_new = result.x[13*6:13*6+13*7].reshape(13, 7)
            W_new = result.x[13*6+13*7:].reshape(13, 10)
            
            for r in range(13):
                U_sol[0*3+0, r] = U_new[r, 0]
                U_sol[1*3+0, r] = U_new[r, 1]
                U_sol[0*3+1, r] = U_new[r, 2]
                U_sol[1*3+1, r] = U_new[r, 3]
                U_sol[0*3+2, r] = U_new[r, 4]
                U_sol[1*3+2, r] = U_new[r, 5]
                
                V_sol[0*5+0, r] = V_new[r, 0]
                V_sol[0*5+1, r] = V_new[r, 1]
                V_sol[0*5+2, r] = V_new[r, 2]
                V_sol[0*5+3, r] = V_new[r, 3]
                V_sol[0*5+4, r] = V_new[r, 4]
                V_sol[1*5+2, r] = V_new[r, 5]
                V_sol[2*5+2, r] = V_new[r, 6]
                
                for k in range(5):
                    for i in range(2):
                        W_sol[k*2+i, r] = W_new[r, k*2+i]
            
            # Combine with Strassen terms
            U_full = np.hstack([U_sol, U_strassen])
            V_full = np.hstack([V_sol, V_strassen])
            W_full = np.hstack([W_sol, W_strassen])
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_full[i*m+j, :] * V_full[j*p+k, :] * W_full[k*n+i, :])
                        if abs(val - 1.0) > 1e-5:
                            return None
            
            return U_full, V_full, W_full
    
    except Exception as e:
        pass
    
    return None
