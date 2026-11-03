"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(V[0]) if V else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _target(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all valid i,j,k."""
    T = np.ones((n, m, p), dtype=float)
    return T

def _error_and_grad(params, n, m, p, r):
    """Compute error and gradient for a given set of parameters."""
    U = params[:n*m*r].reshape(n, m, r)
    V = params[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
    W = params[n*m*r + m*p*r:].reshape(p, n, r)
    
    # Compute S_ijk = sum_r U_ijk V_jk W_ki
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    # Error is sum (S_ijk - 1)^2
    err = np.sum((S - 1.0)**2)
    
    # Gradient
    # dE/dU_ijk = 2 * sum_k (S_ijk - 1) V_jk W_ki
    # dE/dV_jk = 2 * sum_i (S_ijk - 1) U_ijk W_ki
    # dE/dW_ki = 2 * sum_j (S_ijk - 1) U_ijk V_jk
    
    dU = 2 * np.einsum('ijk,jkc,kic->ijc', S - 1.0, V, W)
    dV = 2 * np.einsum('ijk,ijc,kic->jkc', S - 1.0, U, W)
    dW = 2 * np.einsum('ijk,ijc,jkc->kic', S - 1.0, U, V)
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    return err, grad

def _residuals(params, n, m, p, r):
    """Compute residuals for least_squares."""
    U = params[:n*m*r].reshape(n, m, r)
    V = params[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
    W = params[n*m*r + m*p*r:].reshape(p, n, r)
    
    # Compute S_ijk = sum_r U_ijk V_jk W_ki
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    # Residuals are S_ijk - 1
    return (S - 1.0).flatten()

def _try_rank(r, n, m, p, seed, time_limit):
    """Try to find a decomposition of rank r using least_squares."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    for attempt in range(10):
        if time.time() > time_limit:
            break
            
        # Random initialization with small values to help convergence
        U = rng.normal(0, 1, size=(n, m, r))
        V = rng.normal(0, 1, size=(m, p, r))
        W = rng.normal(0, 1, size=(p, n, r))
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use Trust Region Reflective method which handles bounds well if needed, 
            # but here we use default which is 'trf' for general unconstrained.
            res = least_squares(_residuals, params, args=(n, m, p, r), method='trf', 
                                max_nfev=200, xtol=1e-12, ftol=1e-12)
            
            if res.success or res.cost < 1e-10:
                opt_params = res.x
                U_opt = opt_params[:n*m*r].reshape(n, m, r)
                V_opt = opt_params[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
                W_opt = opt_params[n*m*r + m*p*r:].reshape(p, n, r)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify
                if _validate(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130 # Safety margin
    
    # Lower bound
    min_rank = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to schoolbook
    for r in range(min_rank, n*m*p + 1):
        if time.time() > deadline - 15:
            break
            
        # Allocate time per rank attempt
        time_left = deadline - 15 - time.time()
        if time_left < 5:
            break
            
        result = _try_rank(r, n, m, p, seed=1000 + r, time_limit=start_time + time_left)
        
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
