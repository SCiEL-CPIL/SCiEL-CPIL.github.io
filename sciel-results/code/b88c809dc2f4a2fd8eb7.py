"""Program to find rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying 
    a 3x3 matrix by a 3x7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 7
    
    def schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition for n x m x p."""
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
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """
        Verify that the decomposition exactly computes matrix multiplication.
        T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
        """
        rank = U.shape[1]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if abs(u_val) < 1e-15:
                        continue
                    for k in range(p):
                        v_val = v_col[j * p + k]
                        w_val = w_col[k * n + i]
                        T_reconstructed[i, j, k] += u_val * v_val * w_val
        
        T_expected = np.ones((n, m, p))
        return np.allclose(T_reconstructed, T_expected, atol=tol)
    
    def compute_tensor_error(U, V, W, n, m, p):
        """Compute the sum of squared errors for the tensor decomposition."""
        rank = U.shape[1]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U[i * m + j, r]
                    if abs(u_val) < 1e-12:
                        continue
                    for k in range(p):
                        v_val = V[j * p + k, r]
                        w_val = W[k * n + i, r]
                        T_reconstructed[i, j, k] += u_val * v_val * w_val
        
        T_expected = np.ones((n, m, p))
        return np.sum((T_reconstructed - T_expected) ** 2)
    
    def try_rank(r, max_time=30, seed=42):
        """
        Try to find a rank-r exact decomposition for 3x3x7.
        Returns (U, V, W) if found, else None.
        """
        start = time.time()
        if r < 1:
            return None
        
        np.random.seed(seed)
        
        def objective(params):
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r:2*n*m*r].reshape(m*p, r)
            W = params[2*n*m*r:3*n*m*r].reshape(p*n, r)
            return compute_tensor_error(U, V, W, n, m, p)
        
        best_error = np.inf
        best_params = None
        
        # Try multiple random initializations
        max_attempts = min(10, max(1, int((max_time - (time.time() - start)) / 2)))
        
        for attempt in range(max_attempts):
            if time.time() - start > max_time - 5:
                break
            
            # Random initialization with varying scales
            scale = 1.0 + 0.5 * attempt
            x0 = np.random.randn(3 * n * m * r) * scale
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-10}
                )
                
                if result.fun < best_error:
                    best_error = result.fun
                    best_params = result.x.copy()
                    
                    if best_error < 1e-8:
                        break
                        
            except Exception:
                continue
        
        if best_params is not None and best_error < 1e-6:
            U = best_params[:n*m*r].reshape(n*m, r)
            V = best_params[n*m*r:2*n*m*r].reshape(m*p, r)
            W = best_params[2*n*m*r:3*n*m*r].reshape(p*n, r)
            
            if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
        
        return None
    
    # Start with schoolbook as baseline
    U_school, V_school, W_school = schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p  # 63
    
    # Try decreasing ranks from a reasonable upper bound down to 1
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can potentially do better
    # Try ranks from 30 down to 21 first, then if time permits, try lower
    
    # First, try to find a decomposition with rank significantly below 63
    # Start with rank 30 and work down
    for r in [30, 28, 26, 24, 22, 21]:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, max_time=15, seed=42 + r)
        
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break  # Found the smallest rank in this range
    
    # If we didn't find anything below 30, try a few more ranks
    if best_rank > 30 and time.time() < deadline - 20:
        for r in [35, 40, 45, 50, 55]:
            if time.time() > deadline - 15:
                break
            
            result = try_rank(r, max_time=10, seed=100 + r)
            
            if result is not None:
                U, V, W = result
                if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
