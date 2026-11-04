import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 3x3 matrix mult, T[i,j,k] should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=3):
    """Return the standard rank-27 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _find_rank_23(n=3, m=3, p=3, deadline=None):
    """
    Attempt to find a rank-23 decomposition using numerical optimization.
    """
    if deadline is None:
        deadline = time.time() + 100.0
    
    target_rank = 23
    num_constraints = n * m * p # 27
    
    # We need to solve for U, V, W of shape (9, 23)
    # such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    
    # Flatten U, V, W into vectors for optimization
    # Total variables: 3 * 9 * 23 = 621
    
    # Residual function
    def residuals(params):
        U_flat = params[:9*target_rank].reshape(9, target_rank)
        V_flat = params[9*target_rank:18*target_rank].reshape(9, target_rank)
        W_flat = params[18*target_rank:27*target_rank].reshape(9, target_rank)
        
        res = np.zeros(num_constraints)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i*m+j, :] dot V[j*p+k, :] dot W[k*n+i, :]
                    # This is not a simple dot product, it's a sum of products
                    val = 0
                    for r in range(target_rank):
                        val += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Jacobian for faster convergence (optional, but helps)
    # We can use finite differences or derive it.
    # Derivative of res[idx] w.r.t U[i,j,r] is V[j,k,r]*W[k,i,r]
    # Derivative w.r.t V[j,k,r] is U[i,j,r]*W[k,i,r]
    # Derivative w.r.t W[k,i,r] is U[i,j,r]*V[j,k,r]
    
    def jacobian(params):
        U_flat = params[:9*target_rank].reshape(9, target_rank)
        V_flat = params[9*target_rank:18*target_rank].reshape(9, target_rank)
        W_flat = params[18*target_rank:27*target_rank].reshape(9, target_rank)
        
        jac = np.zeros((num_constraints, 27*target_rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        # d/dU
                        jac[idx, i*m+j*target_rank + r] = V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        # d/dV
                        jac[idx, 9*target_rank + j*p+k*target_rank + r] = U_flat[i*m+j, r] * W_flat[k*n+i, r]
                        # d/dW
                        jac[idx, 18*target_rank + k*n+i*target_rank + r] = U_flat[i*m+j, r] * V_flat[j*p+k, r]
                    idx += 1
        return jac

    # Try multiple random initializations
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Scale is important. Start with small values.
        params = np.random.randn(27 * target_rank) * 0.1
        
        # Use scipy.optimize.least_squares
        try:
            from scipy.optimize import least_squares
            res = least_squares(residuals, params, jac=jacobian, max_nfev=1000)
            
            if res.success or np.allclose(res.fun, 0, atol=1e-9):
                # Check if it's really close
                if np.max(np.abs(res.fun)) < 1e-9:
                    # Extract U, V, W
                    U_sol = res.x[:9*target_rank].reshape(9, target_rank).tolist()
                    V_sol = res.x[9*target_rank:18*target_rank].reshape(9, target_rank).tolist()
                    W_sol = res.x[18*target_rank:27*target_rank].reshape(9, target_rank).tolist()
                    
                    # Verify
                    if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                        return U_sol, V_sol, W_sol, target_rank
        except Exception:
            pass
            
    return None, None, None, 27

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, get the guaranteed schoolbook solution
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find rank 23
    U_23, V_23, W_23, rank_23 = _find_rank_23(n, m, p, deadline=start_time + timeout)
    if U_23 is not None and rank_23 < best_rank:
        best_U, best_V, best_W = U_23, V_23, W_23
        best_rank = rank_23
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
