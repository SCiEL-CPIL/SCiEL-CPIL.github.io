import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if output (i,k) depends on (i,j)*(j,k)."""
    # T has shape (n*m, m*p, p*n) in the trace convention W[k*n+i, r]
    # Actually, the multiplication tensor T[a,b,c] where:
    # a = i*m + j (input A index)
    # b = j*p + k (input B index)  
    # c = k*n + i (output C index)
    # T[a,b,c] = 1 if and only if the triple (i,j,k) is consistent:
    #   a = i*m+j, b = j*p+k, c = k*n+i
    # For a given (a,b,c), check if there exist i,j,k such that:
    #   i = a // m, j = a % m
    #   j' = b // p, k = b % p
    #   k' = c // n, i' = c % n
    #   and i == i', j == j', k == k'
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Check if U, V, W exactly decompose T: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
    rank = U.shape[1]
    # Compute reconstruction
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    max_err = np.max(np.abs(recon - T))
    return max_err < tol

def _try_rank(r, n, m, p, T, time_limit, seed=42):
    """Try to find a rank-r decomposition using structured parameterization."""
    rng = np.random.RandomState(seed)
    
    # Parameterize U, V, W with Kronecker structure:
    # U[i*m+j, r] = sum over some basis
    # For a structured approach, try:
    # U = U1 ⊗ U2 type factors, but let's use a more direct approach
    # with random initialization and gradient descent on the non-convex problem.
    
    # Initialize with small random values
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    
    # Use L-BFGS or similar optimization
    # Objective: ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
    
    def objective(params):
        U_p = params[:n*m*r].reshape(n*m, r)
        V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
        recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p, optimize=True)
        return np.sum((recon - T)**2)
    
    def gradient(params):
        U_p = params[:n*m*r].reshape(n*m, r)
        V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
        recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p, optimize=True)
        err = recon - T
        
        # Gradient with respect to U
        gU = np.einsum('abc,br,cr->ar', err, V_p, W_p, optimize=True)
        gV = np.einsum('abc,ar,cr->br', err, U_p, W_p, optimize=True)
        gW = np.einsum('abc,ar,br->cr', err, U_p, V_p, optimize=True)
        
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use scipy.optimize.minimize with L-BFGS-B
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_params = params0.copy()
    best_obj = objective(params0)
    
    # Try a few iterations of gradient descent as fallback if scipy is slow
    lr = 0.01
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        g = gradient(params0)
        params0 = params0 - lr * g
        obj = objective(params0)
        if obj < best_obj:
            best_obj = obj
            best_params = params0.copy()
        # Adaptive learning rate
        if obj > best_obj * 1.01:
            lr *= 0.9
        else:
            lr = min(lr * 1.05, 0.1)
    
    U_best = best_params[:n*m*r].reshape(n*m, r)
    V_best = best_params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
    W_best = best_params[n*m*r + m*p*r:].reshape(p*n, r)
    
    if _verify_decomposition(U_best, V_best, W_best, T):
        return U_best, V_best, W_best
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 120.0
    
    T = _build_tensor(n, m, p)
    
    # First, get the schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = (rank_schoolbook, U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 2x5*5x5 is at least max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we can try ranks between 25 and 50
    
    for r in range(25, rank_schoolbook):
        if time.time() - start_time > time_limit - 10:
            break
        
        remaining_time = time_limit - 10 - (time.time() - start_time)
        if remaining_time < 1:
            break
        
        # Try with a few different seeds
        for seed in [42, 123, 456]:
            if time.time() - start_time > time_limit - 5:
                break
            
            time_per_attempt = max(0.5, remaining_time / 3.0)
            result = _try_rank(r, n, m, p, T, time_per_attempt, seed)
            
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_decomposition(U_r, V_r, W_r, T):
                    best_result = (r, U_r, V_r, W_r)
                    break
        
        if best_result[0] < rank_schoolbook:
            break
    
    r_best, U_best, V_best, W_best = best_result
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
