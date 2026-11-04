import time
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _round_to_half(x):
    return np.round(2 * x) / 2.0

def _als_decomposition(r, n=3, m=3, p=3, max_iter=100, time_budget=10.0):
    """
    Try to find a rank-r decomposition using ALS (Alternating Least Squares).
    """
    start = time.time()
    
    # Initialize randomly
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Update U: for each (i,j), solve for U[i,j,:]
        for i in range(n):
            for j in range(m):
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                # This is: U[i,j,:] @ (V[j,:] * W[:,i]) = ones(p)
                # where (V[j,:] * W[:,i])[c] = V[j*p+0+c] * W[0*n+i+c] ... wait, need to be careful
                
                # Actually: for fixed i,j, we need:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for k=0,1,2
                # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then: U[i*m+j, :] @ M^T = ones(p)
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j * p + k] * W[k * n + i]
                # Solve: U[i*m+j, :] @ M.T = ones(p)
                # This is a least squares problem: M @ U[i*m+j, :]^T = ones(p)
                try:
                    U_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j,k,:]
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                # Let M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # Then: V[j*p+k, :] @ M^T = ones(n)
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i * m + j] * W[k * n + i]
                try:
                    V_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k,i,:]
        for k in range(p):
            for i in range(n):
                # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                # Let M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Then: W[k*n+i, :] @ M^T = ones(m)
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i * m + j] * V[j * p + k]
                try:
                    W_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err > 1e-6:
        return None
    
    # Round to half-integers
    U_rounded = _round_to_half(best_U)
    V_rounded = _round_to_half(best_V)
    W_rounded = _round_to_half(best_W)
    
    # Verify
    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    return None

def _find_decomposition(r, n=3, m=3, p=3, time_budget=10.0, restarts=5):
    """
    Try to find a rank-r decomposition using ALS with random restarts.
    """
    start = time.time()
    
    for restart in range(restarts):
        if time.time() - start > time_budget:
            break
        
        # Set random seed for reproducibility within restart
        np.random.seed(hash((r, restart)) % 2**32)
        
        result = _als_decomposition(r, n, m, p, max_iter=50, time_budget=time_budget / restarts)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the guaranteed schoolbook rank-27 decomposition
    rank_schoolbook = n * m * p  # 27
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > timeout - 5:
            break
            
        budget = 15.0 if r == 23 else 10.0
        restarts = 5 if r == 23 else 3
        
        result = _find_decomposition(r, n, m, p, time_budget=budget, restarts=restarts)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
            
    return best_result
# EVOLVE-BLOCK-END
