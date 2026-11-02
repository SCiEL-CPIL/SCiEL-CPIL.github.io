import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,3,3}.
    Attempts to find a rank 14 decomposition. Falls back to rank 18 if not found.
    """
    n, m, p = 2, 3, 3
    target_rank = 14
    
    # Define the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a%3 == b//3 and b%3 == c//2 and c%2 == a//3 else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def objective(params):
        """
        params: flattened array of U, V, W
        """
        # Unpack
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        
        U = params[:size_u].reshape(n*m, target_rank)
        V = params[size_u:size_u+size_v].reshape(m*p, target_rank)
        W = params[size_u+size_v:size_u+size_v+size_w].reshape(p*n, target_rank)
        
        # Compute tensor decomposition
        # T_calc[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Use einsum for efficiency
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error
        err = np.sum((T - T_calc) ** 2)
        return err

    def check_and_round(params):
        """
        Checks if the solution is close to exact and rounds to half-integers.
        Returns dict if successful, else None.
        """
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        
        U = params[:size_u].reshape(n*m, target_rank)
        V = params[size_u:size_u+size_v].reshape(m*p, target_rank)
        W = params[size_u+size_v:size_u+size_v+size_w].reshape(p*n, target_rank)
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        
        if np.allclose(T_calc, T, atol=1e-9):
            # Convert to lists of lists
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        return None

    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_solution = None
    
    # Try random restarts
    np.random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random initialization
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimize
        try:
            res = minimize(objective, params, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
        except Exception:
            continue
        
        if res.fun < 1e-6:
            sol = check_and_round(res.x)
            if sol:
                best_solution = sol
                break

    if best_solution:
        return best_solution

    # Fallback to schoolbook (rank 18)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
