"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try rank 11 first (known minimal for 2x2x3)
    for rank in [11, 12]:
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    # Fallback to schoolbook
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

def _try_rank(n, m, p, rank):
    """Try to find an exact rank-r decomposition with integer/half-integer entries."""
    start_time = time.time()
    deadline = start_time + 100  # Leave margin for evaluation
    
    # Build target tensor T[i,j,k] = 1 for all i,j,k
    # T is indexed as T[i*m+j, j*p+k, k*n+i] in the decomposition
    # Actually T[i,j,k] = delta (the multiplication tensor)
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 (it's the all-ones tensor in this representation)
    
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Initialize with random values
    np.random.seed(42)
    
    def objective(params):
        if time.time() > deadline:
            return float('inf')
        # Unpack parameters
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:2*n*m*rank + m*p*rank - n*m*rank].reshape(m*p, rank)
        W = params[2*n*m*rank + m*p*rank - n*m*rank:].reshape(p*n, rank)
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (val - 1.0)**2
        return error
    
    # This is a high-dimensional non-convex problem. Let's try a smarter initialization.
    # Use the schoolbook decomposition as a starting point and add structure.
    
    # Actually, let's try a different approach: use known algebraic constructions.
    # For 2x2x3, a rank-11 decomposition exists. Let me try to find it via
    # a structured search.
    
    # Alternative: try to use the fact that we can combine some of the 12 schoolbook
    # terms. The key insight is that one of the 12 terms can be "absorbed" using
    # the other 11.
    
    # Let's try a numerical approach with careful initialization.
    # Start from schoolbook and perturb.
    
    U_init = np.zeros((n*m, rank))
    V_init = np.zeros((m*p, rank))
    W_init = np.zeros((p*n, rank))
    
    # First 11 columns: schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U_init[i*m+j, col] = 1.0
                    V_init[j*p+k, col] = 1.0
                    W_init[k*n+i, col] = 1.0
                    col += 1
    
    # Last column: try to absorb the missing term
    # The missing term is for some (i,j,k) that we didn't include
    # Let's say we're missing (i=1, j=1, k=2) -> indices U[3], V[5], W[5]
    # We need: sum_c U[3,c]*V[5,c]*W[5,c] = 1
    # With the first 11 columns all having 1s in various places, the 12th column
    # needs to contribute the right amount.
    
    # Actually, this structured approach is tricky. Let me try full numerical optimization.
    
    # Reshape for optimization
    U_flat = U_init.flatten()
    V_flat = V_init.flatten()
    W_flat = W_init.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    best_result = None
    best_error = float('inf')
    
    # Try multiple random initializations
    n_trials = 5
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x = x0.copy()
        else:
            # Add random perturbation
            x = x0 + np.random.randn(len(x0)) * 0.5
        
        # Use Nelder-Mead or similar
        try:
            res = minimize(objective, x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-10})
            if res.fun < best_error:
                best_error = res.fun
                best_result = res.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    if best_result is None or best_error > 1e-6:
        return None
    
    # Unpack best result
    U = best_result[:n*m*rank].reshape(n*m, rank)
    V = best_result[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = best_result[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Round to nearest half-integer
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify exactness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-9:
        return {
            "U": U_rounded.tolist(),
            "V": V_rounded.tolist(),
            "W": W_rounded.tolist()
        }
    
    return None

# EVOLVE-BLOCK-END
