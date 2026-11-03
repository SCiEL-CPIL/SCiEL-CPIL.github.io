"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    num_u = n * m  # 6
    num_v = m * p  # 18
    num_w = p * n  # 12
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from a lower bound upward
    # Lower bound: max(num_u, num_v, num_w) is not quite right for tensor rank
    # But a simple lower bound is max(n*m, m*p, p*n) / something... 
    # Actually, the rank must be at least max(num_u, num_v, num_w) is NOT a lower bound for tensor rank
    # The rank must be at least the matrix rank of any matricization
    # Matricization 1: (n*m) x (m*p) -> rank at least... the all-ones 6x18 matrix has rank 1
    # So that doesn't help.
    
    # Let's just try small ranks and see what works
    # For this specific case, I'll try to find a construction
    
    # Known result: rank([2,3,6]) <= 12
    # Let's try to construct rank 12
    
    deadline = time.time() + 120
    best_result = None
    
    # Try rank 12 first (known to be achievable)
    for r in [12, 13, 14, 15, 16, 18, 20, 24, 36]:
        if time.time() > deadline - 10:
            break
            
        # Random initialization with multiple restarts
        best_sol = None
        best_cost = np.inf
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = np.random.randn(num_u, r) * 0.5
            V = np.random.randn(num_v, r) * 0.5
            W = np.random.randn(num_w, r) * 0.5
            
            def residuals(params):
                U_flat = params[:num_u * r].reshape(num_u, r)
                V_flat = params[num_u * r : num_u * r + num_v * r].reshape(num_v, r)
                W_flat = params[num_u * r + num_v * r :].reshape(num_w, r)
                
                # Compute all 36 entries
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_flat[i * m + j, c] * V_flat[j * p + k, c] * W_flat[k * n + i, c]
                            res.append(val - 1.0)
                return np.array(res)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residuals, params, method='lm', max_nfev=200)
                cost = np.sum(result.fun ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = result.x
            except Exception:
                pass
        
        if best_sol is not None and best_cost < 1e-10:
            U = best_sol[:num_u * r].reshape(num_u, r)
            V = best_sol[num_u * r : num_u * r + num_v * r].reshape(num_v, r)
            W = best_sol[num_u * r + num_v * r :].reshape(num_w, r)
            
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Convert to list of lists with integer/half-integer if possible
                U_list = [[float(U[i, c]) for c in range(r)] for i in range(num_u)]
                V_list = [[float(V[i, c]) for c in range(r)] for i in range(num_v)]
                W_list = [[float(W[i, c]) for c in range(r)] for i in range(num_w)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(num_u)]
    V = [[0.0] * rank for _ in range(num_v)]
    W = [[0.0] * rank for _ in range(num_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
