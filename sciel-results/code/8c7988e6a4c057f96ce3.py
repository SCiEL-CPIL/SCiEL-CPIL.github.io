import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # 1. Immediately build the known Rank 14 solution
    # This serves as the fallback and the baseline.
    rank14_solution = _build_rank14()
    
    # Check if we have time to try for Rank 13
    # The evaluator kills after 150s. We should aim to finish by 145s.
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find Rank 13
    # We use the rank 14 solution to initialize the rank 13 search by dropping a column
    # or simply using random init but with more restarts.
    # Given the difficulty, random init is often used, but we can try to exploit the structure.
    
    try:
        from scipy.optimize import least_squares
        import random
        
        target_rank = 13
        
        # Build target tensor T
        T = np.zeros((u_dim, v_dim, w_dim), dtype=float)
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[u_idx, v_idx, w_idx] = 1.0
        
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
        def unflatten_params(x):
            U_size = u_dim * target_rank
            V_size = v_dim * target_rank
            U = x[:U_size].reshape(u_dim, target_rank)
            V = x[U_size:U_size+V_size].reshape(v_dim, target_rank)
            W = x[U_size+V_size:].reshape(w_dim, target_rank)
            return U, V, W
            
        def residuals(x):
            U, V, W = unflatten_params(x)
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            return (T - T_approx).flatten()
            
        best_cost = np.inf
        best_x = None
        
        # Perform a few restarts with random initialization
        # We limit the number of restarts based on remaining time
        max_restarts = 20
        attempts = 0
        while attempts < max_restarts and time.time() < deadline:
            attempts += 1
            
            # Random initialization
            U = np.random.randn(u_dim, target_rank)
            V = np.random.randn(v_dim, target_rank)
            W = np.random.randn(w_dim, target_rank)
            
            # Normalize columns to help convergence
            for r in range(target_rank):
                norm_u = np.linalg.norm(U[:, r])
                if norm_u > 1e-9:
                    U[:, r] /= norm_u
                    V[:, r] *= norm_u
                norm_v = np.linalg.norm(V[:, r])
                if norm_v > 1e-9:
                    V[:, r] /= norm_v
                    W[:, r] *= norm_v
                    
            x0 = flatten_params(U, V, W)
            
            try:
                # Use a tighter max_nfev to ensure we finish within time
                # LM is fast per step, but can take many steps
                res = least_squares(residuals, x0, method='lm', max_nfev=5000)
                cost = res.cost
                if cost < best_cost:
                    best_cost = cost
                    best_x = res.x
                    if cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_x is not None and best_cost < 1e-6:
            U, V, W = unflatten_params(best_x)
            # Check for integer/half-integer entries
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            T_approx = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
            if np.allclose(T, T_approx, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
    except ImportError:
        pass
    except Exception:
        pass
        
    # Fallback to Rank 14
    return rank14_solution

def _build_rank14():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for t in range(7):
        u_dict, v_dict, w_dict = strassen_terms[t]
        col = t
        for (i, j), coeff in u_dict.items():
            U[i*m+j][col] += coeff
        for (j, kl), coeff in v_dict.items():
            V[j*p+kl][col] += coeff
        for (kl, i), coeff in w_dict.items():
            W[kl*n+i][col] += coeff
            
    for t in range(7):
        u_dict, v_dict, w_dict = strassen_terms[t]
        col = 7 + t
        for (i, j), coeff in u_dict.items():
            U[i*m+j][col] += coeff
        for (j, kl), coeff in v_dict.items():
            V[j*p+(kl+2)][col] += coeff
        for (kl, i), coeff in w_dict.items():
            W[(kl+2)*n+i][col] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
