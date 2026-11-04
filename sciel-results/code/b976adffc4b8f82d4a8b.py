import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _generate_target_tensor(n, m, p):
    """Generate the target tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _compute_residuals(params, n, m, p, rank, T_flat):
    """Compute residuals for a given parameter vector."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    residuals = T_hat - T_flat.reshape(n * m, m * p, p * n)
    return residuals.flatten()

def _check_and_round(U, V, W, n, m, p, rank, T_flat):
    """
    Attempt to round U, V, W to nearest half-integers and check if they exactly
    reconstruct the tensor. Returns (U_round, V_round, W_round) or None.
    """
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    
    # Check exactness
    T_hat = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
    if np.allclose(T_hat, T_flat.reshape(n * m, m * p, p * n), atol=1e-9, rtol=1e-9):
        return U_r, V_r, W_r
    return None

def _try_rank_fast(n, m, p, target_rank, T_flat, timeout=2.0):
    """
    Try to find a decomposition for the given target rank using random restarts
    and a simple gradient descent step, followed by rounding check.
    """
    start_time = time.time()
    
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_params = size_u + size_v + size_w
    
    # Number of restarts depends on remaining time, but we cap it to avoid infinite loops
    max_restarts = 50 
    
    for trial in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 999 + 42)
        # Initialize with small random values
        x0 = np.random.randn(total_params) * 0.5
        
        # Simple gradient descent for a few steps
        # We don't use scipy minimize here to keep it lightweight and allow many restarts
        # We use a simple finite difference gradient or just random walk?
        # Let's use a few steps of Nelder-Mead via scipy? No, too slow.
        # Let's use a simple coordinate descent or just check if random init works (unlikely).
        # Better: Use a few steps of gradient descent with numerical gradient.
        
        # For speed, we will just try a few random perturbations and check rounding.
        # If the rank is low, the solution space is small.
        # Actually, for matrix multiplication, solutions are often integer.
        # Let's try initializing with integers/half-integers directly.
        
        # Strategy: Random integer/half-integer initialization
        # This is a discrete search.
        
        # Generate random half-integers
        # Scale: values are likely small (0, 1, -1, 0.5, -0.5)
        
        # If we just sample random half-integers, we might hit it if rank is close to schoolbook?
        # Unlikely for rank 41 if schoolbook is 42.
        
        # Let's stick to a very fast local optimization:
        # Use scipy.optimize.minimize with 'Nelder-Mead' which is derivative-free and fast for small problems.
        
        from scipy.optimize import minimize
        
        def objective(params):
            res = _compute_residuals(params, n, m, p, target_rank, T_flat)
            return np.sum(res * res)
        
        try:
            # Nelder-Mead is good for small dimensions and doesn't need gradients
            result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if result.fun < 1e-4: # Close enough to try rounding
                U = result.x[:size_u].reshape(n * m, target_rank)
                V = result.x[size_u:size_u + size_v].reshape(m * p, target_rank)
                W = result.x[size_u + size_v:].reshape(p * n, target_rank)
                
                rounded = _check_and_round(U, V, W, n, m, p, target_rank, T_flat)
                if rounded:
                    return rounded
                    
        except Exception:
            continue
            
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    T = _generate_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 21
    # We search from 41 down to 21
    # If we find one, we return it.
    
    for target_rank in range(41, 20, -1):
        if time.time() > deadline:
            break
            
        # Give more time to higher ranks (closer to schoolbook) as they are more likely to exist
        # and have more structured solutions?
        # Actually, lower ranks are harder.
        # Let's give 5 seconds per rank.
        result = _try_rank_fast(n, m, p, target_rank, T_flat, timeout=5.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
