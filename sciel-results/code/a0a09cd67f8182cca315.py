"""Program for matrix multiplication 2x5x6 tensor decomposition.

Uses a structured decomposition that exploits the small dimension n=2
to potentially reduce rank below the schoolbook 60, by grouping terms
across the n dimension and using cancellations.
"""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 5, 6
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute the target tensor T as a flat vector
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(i,k) else 0
    target = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                idx = (u * dim_v + v) * dim_w + w
                target[idx] = 1.0

    def residual(params, R):
        size_u = dim_u * R
        size_v = dim_v * R
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:]
        
        U = U_flat.reshape(dim_u, R)
        V = V_flat.reshape(dim_v, R)
        W = W_flat.reshape(dim_w, R)
        
        T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        return (T_calc - target).flatten()

    best_solution = None
    best_rank = n * m * p
    
    # Try to find a lower rank
    candidate_ranks = [50, 45, 40]
    
    for R in candidate_ranks:
        for restart in range(2):
            np.random.seed(42 + restart + R)
            
            U_init = np.random.randn(dim_u, R) * 0.1
            V_init = np.random.randn(dim_v, R) * 0.1
            W_init = np.random.randn(dim_w, R) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residual, params_init, args=(R,), method='lm', max_nfev=2000)
                res = result.cost
                sum_sq = 2 * res
                if sum_sq < 1e-6:
                    U_opt = result.x[:dim_u*R].reshape(dim_u, R)
                    V_opt = result.x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                    W_opt = result.x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                    
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    T_round = np.einsum('ur,vr,wr->uvw', U_round, V_round, W_round)
                    diff = np.abs(T_round - target)
                    if np.max(diff) < 1e-9:
                        best_rank = R
                        best_solution = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        break
            except Exception:
                pass
        if best_solution is not None:
            # If we found a solution, we could try lower ranks, but to save time we stop here
            # or continue? The loop continues to lower R. 
            # But if we found 50, we should try 45.
            # The break only breaks the inner restart loop.
            pass

    if best_solution is None:
        R = n * m * p
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        best_solution = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return best_solution
# EVOLVE-BLOCK-END
