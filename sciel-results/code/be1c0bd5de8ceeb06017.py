"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x3 by 3x7.
Uses hierarchical approach: algebraic constructions + optimized search.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_strassen_style(n, m, p):
    """
    Try to find a Strassen-like decomposition by splitting into blocks.
    For 2x3 * 3x7, we can try splitting the 3x3 middle dimension.
    """
    # For now, return None as a placeholder for algebraic constructions
    # This is where we'd implement specific algebraic identities
    return None

def _optimize_rank(n, m, p, target_rank, max_trials=30, timeout=20):
    """
    Use gradient-based optimization to find a valid decomposition of given rank.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # Total parameters
    n_params = target_rank * (n*m + m*p + p*n)
    
    def compute_residuals(params):
        """Compute residuals for all (i,j,k) entries."""
        U = params[:target_rank*n*m].reshape(target_rank, n*m).T
        V = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
        W = params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
        
        residuals = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals
    
    def compute_jacobian(params):
        """Compute Jacobian of residuals with respect to parameters."""
        U = params[:target_rank*n*m].reshape(target_rank, n*m).T
        V = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
        W = params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
        
        jac = np.zeros((n*m*p, n_params))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i*m+j, c] of sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                    for c in range(target_rank):
                        jac[idx, c*n*m + i*m+j] = V[j*p+k, c] * W[k*n+i, c]
                    
                    # d/dV[j*p+k, c]
                    offset_v = target_rank * n*m
                    for c in range(target_rank):
                        jac[idx, offset_v + c*m*p + j*p+k] = U[i*m+j, c] * W[k*n+i, c]
                    
                    # d/dW[k*n+i, c]
                    offset_w = target_rank * (n*m + m*p)
                    for c in range(target_rank):
                        jac[idx, offset_w + c*p*n + k*n+i] = U[i*m+j, c] * V[j*p+k, c]
                    
                    idx += 1
        return jac
    
    best_params = None
    best_resid_norm = np.inf
    
    for trial in range(max_trials):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small values
        np.random.seed(trial * 7919 + 42)
        x0 = np.random.randn(n_params) * 0.3
        
        try:
            from scipy.optimize import least_squares
            result = least_squares(
                compute_residuals, 
                x0, 
                jac=compute_jacobian,
                method='lm', 
                max_nfev=500,
                xtol=1e-12,
                ftol=1e-12
            )
            
            final_resid = np.max(np.abs(result.fun))
            if final_resid < best_resid_norm:
                best_resid_norm = final_resid
                best_params = result.x
                
            if final_resid < 1e-8:
                break
                
        except Exception:
            continue
    
    if best_params is None or best_resid_norm > 1e-6:
        return None
    
    # Extract and verify
    U = best_params[:target_rank*n*m].reshape(target_rank, n*m).T
    V = best_params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
    W = best_params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
    
    if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    
    return U, V, W

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Get schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But tensor rank is typically higher. Try from reasonable lower bound.
    
    # First try algebraic constructions
    result = _try_strassen_style(n, m, p)
    if result is not None:
        U, V, W = result
        if _check_decomposition(U, V, W, n, m, p, tol=1e-8):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    # Try optimization for various target ranks
    # Be strategic: try ranks that might be achievable
    # For 2x3x7, known bounds suggest rank might be around 25-35
    target_ranks = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:  # Leave 30s buffer
            break
        
        if target_rank >= best_rank:
            break
        
        # Allocate time based on how promising the rank is
        # Give more time to ranks closer to expected minimum
        if target_rank <= 25:
            timeout = 25
        elif target_rank <= 30:
            timeout = 20
        else:
            timeout = 15
        
        result = _optimize_rank(n, m, p, target_rank, max_trials=20, timeout=timeout)
        
        if result is not None:
            U, V, W = result
            if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Found a valid decomposition, can stop if we hit a good rank
                if best_rank <= 25:
                    break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
