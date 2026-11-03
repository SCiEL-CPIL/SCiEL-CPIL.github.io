"""Matrix multiplication rank decomposition for 3x4x5."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from itertools import product
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use known optimal or near-optimal decompositions
    # For 3x4x5, try to find minimal rank using structured search
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound, using systematic integer search
    for rank in range(lower_bound, n*m*p + 1):
        result = _systematic_search(n, m, p, rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _compute_tensor_from_factors(U, V, W):
    """Compute tensor from factor matrices."""
    rank = U.shape[1]
    T = np.zeros(U.shape[0] * V.shape[0] * W.shape[0])
    for r in range(rank):
        T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    return T.reshape(U.shape[0], V.shape[0], W.shape[0])

def _systematic_search(n, m, p, rank, max_coeff=3):
    """Systematic search for integer decomposition with given rank."""
    target = _build_target_tensor(n, m, p)
    target_flat = target.flatten()
    
    # Number of constraints: n*m*m*p*n*p entries, but most are zero
    # Only n*m*p entries are 1, rest are 0
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # And = 0 for all other (a,b,c) combinations
    
    # This is a hard nonlinear problem. Use local optimization from multiple starts.
    best_solution = None
    best_error = float('inf')
    
    np.random.seed(42)
    
    for trial in range(50):
        # Initialize with small random integers
        U = np.random.randint(-max_coeff, max_coeff+1, size=(n*m, rank)).astype(float)
        V = np.random.randint(-max_coeff, max_coeff+1, size=(m*p, rank)).astype(float)
        W = np.random.randint(-max_coeff, max_coeff+1, size=(n*p, rank)).astype(float)
        
        # Add a bias to avoid all zeros
        if trial == 0:
            # Start with schoolbook-like structure truncated
            U[:min(rank, n*m*p), :] = 0
            V[:min(rank, m*p), :] = 0
            W[:min(rank, n*p), :] = 0
            for r in range(min(rank, n*m*p)):
                i = r // (m*p)
                j = (r // p) % m
                k = r % p
                if i*m+j < n*m:
                    U[i*m+j, r] = 1.0
                if j*p+k < m*p:
                    V[j*p+k, r] = 1.0
                if k*n+i < n*p:
                    W[k*n+i, r] = 1.0
        
        # Optimize
        def objective(params):
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_flat = params[2*n*m*rank:].reshape(n*p, rank)
            T = _compute_tensor_from_factors(U_flat, V_flat, W_flat)
            return np.sum((T - target)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            U_opt = res.x[:n*m*rank].reshape(n*m, rank)
            V_opt = res.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_opt = res.x[2*n*m*rank:].reshape(n*p, rank)
            
            # Check if solution is close to integers or half-integers
            T_opt = _compute_tensor_from_factors(U_opt, V_opt, W_opt)
            error = np.sum((T_opt - target)**2)
            
            if error < 1e-8:
                # Round to nearest integer or half-integer
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                T_rounded = _compute_tensor_from_factors(U_rounded, V_rounded, W_rounded)
                error_rounded = np.sum((T_rounded - target)**2)
                
                if error_rounded < 1e-8:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                
                # Try pure integer rounding
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                
                T_int = _compute_tensor_from_factors(U_int, V_int, W_int)
                error_int = np.sum((T_int - target)**2)
                
                if error_int < 1e-8:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
            
            if error < best_error:
                best_error = error
                best_solution = (U_opt, V_opt, W_opt)
        except Exception:
            continue
    
    return None

# EVOLVE-BLOCK-END
