import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if indices valid."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _reconstruction_error(params, n, m, p, rank):
    """
    Compute Frobenius norm of reconstruction error.
    params: flattened [U_flat, V_flat, W_flat]
    """
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n * m, rank)
    V = params[U_size:U_size+V_size].reshape(m * p, rank)
    W = params[U_size+V_size:].reshape(p * n, rank)
    
    # Tensor reconstruction: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                recon = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                error += (recon - 1.0) ** 2
    return error

def _try_rank(n, m, p, rank, time_budget=30.0):
    """Try to find a decomposition of given rank using numerical optimization."""
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    total_size = U_size + V_size + W_size
    
    # Initialize with random values
    np.random.seed(42 + rank)
    params = np.random.randn(total_size) * 0.5
    
    def objective(x):
        return _reconstruction_error(x, n, m, p, rank)
    
    def gradient(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Use L-BFGS-B for optimization
    result = minimize(
        objective, 
        params, 
        method='L-BFGS-B',
        options={'maxiter': 1000, 'maxfun': 10000, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < 1e-6:
        # Extract solution
        best_params = result.x
        U = best_params[:U_size].reshape(n * m, rank)
        V = best_params[U_size:U_size+V_size].reshape(m * p, rank)
        W = best_params[U_size+V_size:].reshape(p * n, rank)
        
        # Verify exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    recon = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(recon - 1.0) > 1e-6:
                        return None
        
        return U, V, W
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor of dimensions 2x5 by 5x5.
    Attempts to find a lower-rank decomposition using numerical optimization,
    falling back to schoolbook if no better solution is found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try ranks from low to high
    best_solution = None
    best_rank = None
    
    # Schoolbook rank as upper bound
    schoolbook_rank = n * m * p
    
    # Try various ranks
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
            
        try:
            solution = _try_rank(n, m, p, rank, time_budget=20.0)
            if solution is not None:
                best_solution = solution
                best_rank = rank
                break
        except Exception:
            continue
    
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists with integer/half-integer rounding
        U_list = [[round(x * 2) / 2 for x in row] for row in U]
        V_list = [[round(x * 2) / 2 for x in row] for row in V]
        W_list = [[round(x * 2) / 2 for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
