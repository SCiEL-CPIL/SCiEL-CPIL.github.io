"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _objective_3x3(x, rank):
    """Objective for 3x3x3 matrix multiplication."""
    n, m, p = 3, 3, 3
    U = x[:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
    
    err_sum = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err_sum += (val - 1.0) ** 2
    return err_sum

def _find_rank_3x3(target_rank, max_time=10.0):
    """Try to find a rank-target_rank decomposition for 3x3x3."""
    n, m, p = 3, 3, 3
    start_time = time.time()
    total_params = n*m*target_rank + m*p*target_rank + p*n*target_rank
    
    # Try a few random seeds
    for seed in range(10):
        if time.time() - start_time > max_time:
            break
        np.random.seed(seed)
        x0 = np.random.randn(total_params) * 0.5
        
        try:
            result = minimize(_objective_3x3, x0, args=(target_rank,), 
                              method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < 1e-8:
                U = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                if _check_decomposition(U, V, W, n, m, p) < 1e-6:
                    return U, V, W
        except Exception:
            pass
    return None, None, None

def _schoolbook_3x3():
    """Schoolbook decomposition for 3x3x3."""
    n, m, p = 3, 3, 3
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # Step 1: Find best rank for 3x3x3
    # Known min rank is 19. Schoolbook is 27.
    # Try to find rank 19, 20, ..., 27
    best_U_3x3, best_V_3x3, best_W_3x3 = None, None, None
    best_rank_3x3 = 27
    
    for r in range(19, 28):
        if time.time() - start_time > deadline - 20:
            break
        U_r, V_r, W_r = _find_rank_3x3(r, max_time=5.0)
        if U_r is not None:
            best_U_3x3, best_V_3x3, best_W_3x3 = U_r, V_r, W_r
            best_rank_3x3 = r
            break
    
    if best_U_3x3 is None:
        best_U_3x3, best_V_3x3, best_W_3x3 = _schoolbook_3x3()
        best_rank_3x3 = 27

    # Step 2: Construct 3x3x6 decomposition
    # The 3x3x6 tensor is the direct sum of two 3x3x3 tensors
    # Rank = 2 * best_rank_3x3
    total_rank = 2 * best_rank_3x3
    
    # U: (9, total_rank)
    # V: (18, total_rank)
    # W: (18, total_rank)
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # First half (columns 0-2 of B, rows 0-2 of C)
    U[:, :best_rank_3x3] = best_U_3x3
    V[:m*3, :best_rank_3x3] = best_V_3x3
    W[:p*3, :best_rank_3x3] = best_W_3x3
    
    # Second half (columns 3-5 of B, rows 3-5 of C)
    U[:, best_rank_3x3:] = best_U_3x3
    V[m*3:, best_rank_3x3:] = best_V_3x3
    W[p*3:, best_rank_3x3:] = best_W_3x3
    
    # Verify
    err = _check_decomposition(U, V, W, n, m, p)
    if err > 1e-5:
        # Fallback to schoolbook 54 if something went wrong
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        total_rank = rank

    # Optional: Try to optimize the full 3x3x6 for lower rank?
    # Given the disjoint supports, it's unlikely to go below 2*R(3,3,3).
    # But let's try a quick check for rank total_rank - 1 if time permits.
    if time.time() - start_time < deadline - 10 and total_rank > 20:
        # Try to find rank total_rank - 1
        # This is expensive, so we skip unless we have lots of time.
        # For now, we return the constructed solution.
        pass

    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
