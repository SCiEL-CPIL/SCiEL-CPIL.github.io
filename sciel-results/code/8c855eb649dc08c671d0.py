"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _build_tensor_constraints(n, m, p):
    """Return list of (U_idx, V_idx, W_idx) for each tensor entry that must equal 1."""
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    return constraints

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

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

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _polish_decomposition(U, V, W, n, m, p, max_iter=1000, tol=1e-12):
    """High-precision polish pass to ensure exactness."""
    rank = U.shape[1]
    
    def objective(x):
        U_mat = x[:n*m*rank].reshape(n * m, rank)
        V_mat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
        W_mat = x[n*m*rank + m*p*rank :].reshape(p * n, rank)
        
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    err_sum += (val - 1.0) ** 2
        return err_sum
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-14})
        
        if result.fun < 1e-10:
            U_polished = result.x[:n*m*rank].reshape(n * m, rank)
            V_polished = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
            W_polished = result.x[n*m*rank + m*p*rank :].reshape(p * n, rank)
            
            # Verify
            err = _check_decomposition(U_polished, V_polished, W_polished, n, m, p)
            if err < 1e-8:
                return U_polished, V_polished, W_polished, True
    except Exception:
        pass
    
    return U, V, W, False

def _try_find_decomposition(n, m, p, target_rank, seed=0, max_iter=500):
    """Try to find a rank-target_rank decomposition using structured initialization."""
    np.random.seed(seed)
    
    # Initialize with structured patterns based on schoolbook
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Distribute the 54 tensor entries across target_rank columns
    # Each column handles some subset of (i,j,k) triples
    entries_per_col = 54 // target_rank + (1 if 54 % target_rank > 0 else 0)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= target_rank:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Add random perturbation to escape local minima
    U += np.random.randn(n * m, target_rank) * 0.1
    V += np.random.randn(m * p, target_rank) * 0.1
    W += np.random.randn(p * n, target_rank) * 0.1
    
    def objective(x):
        U_mat = x[:n*m*target_rank].reshape(n * m, target_rank)
        V_mat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
        W_mat = x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
        
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    err_sum += (val - 1.0) ** 2
        return err_sum
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < 1e-8:
            U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
            V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
            W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
            
            # Verify
            err = _check_decomposition(U_found, V_found, W_found, n, m, p)
            if err < 1e-6:
                return U_found, V_found, W_found, True
    except Exception:
        pass
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook is valid
    err = _check_decomposition(U, V, W, n, m, p)
    if err < 1e-10:
        U_int = [[int(round(x)) for x in row] for row in U]
        V_int = [[int(round(x)) for x in row] for row in V]
        W_int = [[int(round(x)) for x in row] for row in W]
    
    # For 3x3x6, try ranks from 54 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But tensor rank lower bound is higher; try down to 20
    rank_targets = list(range(54, 19, -1))
    
    for target_rank in rank_targets:
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations with different seeds
        for attempt in range(10):
            if time.time() - start_time > deadline - 10:
                break
            
            U_r, V_r, W_r, success = _try_find_decomposition(
                n, m, p, target_rank, seed=attempt * 7919 + target_rank * 31, max_iter=300
            )
            
            if success:
                # Polish for high precision
                U_pol, V_pol, W_pol, polish_success = _polish_decomposition(
                    U_r, V_r, W_r, n, m, p, max_iter=500, tol=1e-14
                )
                
                if polish_success:
                    U_r, V_r, W_r = U_pol, V_pol, W_pol
                
                # Try to round to half-integers
                U_rounded = _to_half_int(U_r)
                V_rounded = _to_half_int(V_r)
                W_rounded = _to_half_int(W_r)
                
                if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-8:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                else:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                
                found = True
                break
        
        # If we found a valid decomposition at this rank, continue to try lower
        if found:
            continue
    
    # Final result
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
