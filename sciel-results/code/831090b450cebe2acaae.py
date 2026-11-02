import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # Build the known rank 28 solution
    U28, V28, W28 = _build_rank28_arrays(n, m, p)
    
    # Try to find rank 27 using a refined search strategy
    if time.time() < deadline - 10:
        result = _search_rank_refined(n, m, p, 27, U28, V28, W28, deadline - 10)
        if result is not None:
            return _to_lists(result[0], result[1], result[2])
    
    # Fallback to rank 28
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

def _search_rank_refined(n, m, p, target_rank, U_ref, V_ref, W_ref, deadline):
    """
    Refined search using tensor unfolding approach.
    Key insight: The matrix multiplication tensor can be viewed as a bilinear map.
    We use the fact that for rank r decomposition, we need:
    sum_{t=1}^{r} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    We parameterize this as a system of nonlinear equations and use a
    multi-start Levenberg-Marquardt approach with better initialization.
    """
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    num_constraints = n * m * p  # 30
    
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
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Use the reference solution and try to "compress" it
    # by finding a rank-deficient perturbation that still satisfies the constraints
    
    max_attempts = 50
    np.random.seed(12345)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        if attempt < 20:
            # Random initialization with smaller scale
            scale = 0.5 + attempt * 0.1
            params_init = np.random.normal(0, scale, target_rank * (dim_u + dim_v + dim_w))
        elif attempt < 40:
            # Derived from reference: perturb the 28-col solution
            # Take 27 columns and add a small random perturbation
            cols_to_keep = np.random.choice(28, target_rank, replace=False)
            U_init = U_ref[:, cols_to_keep] + np.random.normal(0, 0.1, (dim_u, target_rank))
            V_init = V_ref[:, cols_to_keep] + np.random.normal(0, 0.1, (dim_v, target_rank))
            W_init = W_ref[:, cols_to_keep] + np.random.normal(0, 0.1, (dim_w, target_rank))
            params_init = np.concatenate([
                U_init.T.flatten(),
                V_init.T.flatten(),
                W_init.T.flatten()
            ])
        else:
            # Try to find a sparse solution: initialize with zeros and sparse patterns
            params_init = np.zeros(target_rank * (dim_u + dim_v + dim_w))
            # Add sparse random entries
            num_nonzero_u = np.random.randint(target_rank, target_rank * 3)
            num_nonzero_v = np.random.randint(target_rank, target_rank * 5)
            num_nonzero_w = np.random.randint(target_rank, target_rank * 2)
            
            u_indices = np.random.choice(target_rank * dim_u, num_nonzero_u, replace=False)
            v_indices = np.random.choice(target_rank * dim_v, num_nonzero_v, replace=False)
            w_indices = np.random.choice(target_rank * dim_w, num_nonzero_w, replace=False)
            
            params_init[u_indices] = np.random.choice([-1, 1], num_nonzero_u)
            params_init[target_rank*dim_u + v_indices] = np.random.choice([-1, 1], num_nonzero_v)
            params_init[target_rank*dim_u + target_rank*dim_v + w_indices] = np.random.choice([-1, 1], num_nonzero_w)
        
        try:
            from scipy.optimize import least_squares
            result = least_squares(residuals, params_init, method='lm', max_nfev=10000)
            
            if np.max(np.abs(result.fun)) < 1e-6:
                U_sol = result.x[:target_rank*dim_u].reshape(target_rank, dim_u).T
                V_sol = result.x[target_rank*dim_u:target_rank*dim_u+target_rank*dim_v].reshape(target_rank, dim_v).T
                W_sol = result.x[target_rank*dim_u+target_rank*dim_v:].reshape(target_rank, dim_w).T
                
                if check_solution(U_sol, V_sol, W_sol, tol=1e-6):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
            
    return None

def _to_lists(U, V, W):
    """
    Convert numpy arrays to lists of integers or half-integers.
    The entries should be exactly integer or half-integer for valid solutions.
    """
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Check if it's close to an integer
                if abs(val - round(val)) < 1e-8:
                    row_list.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    # Half-integer
                    half = round(val * 2) / 2
                    if abs(half - int(half)) < 1e-8:
                        row_list.append(int(half))
                    else:
                        row_list.append(float(half))
                else:
                    # Shouldn't happen for valid solutions, but handle gracefully
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
