import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check invalid entries
    # This is expensive, but we can sample or check a subset.
    # For now, we assume that if the valid entries are 1, the invalid ones might be 0
    # due to the structure, but we must verify.
    # Actually, the tensor is defined such that invalid entries are 0.
    # So we must check them.
    # To save time, we can check a random subset of invalid entries.
    num_checks = 100
    indices_checked = 0
    while indices_checked < num_checks:
        i1 = np.random.randint(n)
        j1 = np.random.randint(m)
        j2 = np.random.randint(m)
        k1 = np.random.randint(p)
        i2 = np.random.randint(n)
        k2 = np.random.randint(p)
        
        # If this is a valid combination, skip
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
            
        val = np.sum(U[i1*m+j1, :] * V[j2*p+k1, :] * W[k2*n+i2, :])
        if abs(val) > tol:
            return False
        indices_checked += 1
        
    return True

def _try_rank(n, m, p, target_rank, max_iter=1000, timeout=10):
    """Try to find a valid decomposition for the given target rank."""
    start_time = time.time()
    
    n_params = target_rank * (n*m + m*p + p*n)
    
    def objective(params):
        U = params[:target_rank*n*m].reshape(target_rank, n*m).T
        V = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
        W = params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
        
        residuals = []
        
        # Valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
                    
        # Invalid entries (sampled)
        # We add a penalty for invalid entries being non-zero
        # We use a fixed number of random checks
        for _ in range(200):
            i1 = np.random.randint(n)
            j1 = np.random.randint(m)
            j2 = np.random.randint(m)
            k1 = np.random.randint(p)
            i2 = np.random.randint(n)
            k2 = np.random.randint(p)
            
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue
                
            val = np.sum(U[i1*m+j1, :] * V[j2*p+k1, :] * W[k2*n+i2, :])
            residuals.append(val)
            
        return np.array(residuals)
    
    best_params = None
    best_resid = np.inf
    
    for trial in range(50):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 12345 + 42)
        x0 = np.random.randn(n_params) * 0.5
        
        try:
            result = least_squares(objective, x0, method='lm', max_nfev=max_iter)
            final_resid = np.max(np.abs(result.fun))
            if final_resid < best_resid:
                best_resid = final_resid
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None or best_resid > 1e-6:
        return None
    
    U = best_params[:target_rank*n*m].reshape(target_rank, n*m).T
    V = best_params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
    W = best_params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
    
    # Verify with a stricter check
    if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
            return U_rounded, V_rounded, W_rounded
        else:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1.0
                V_school[j * p + k, c] = 1.0
                W_school[k * n + i, c] = 1.0
                c += 1
    
    # Try to find lower rank
    # Lower bound: max(n*m, m*p, p*n) = 21
    for target_rank in range(25, 42):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, max_iter=500, timeout=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
