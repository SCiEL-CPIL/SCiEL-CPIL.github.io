import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Return the rank-17 decomposition of the 2x3 x 3x3 matrix
    multiplication tensor if found, otherwise rank-18.
    """
    n, m, p = DIMENSIONS
    rank_target = 17
    
    # Build the target tensor T as a 3D array
    # Dimensions: (n*m, m*p, p*n) = (6, 9, 6)
    T = np.zeros((n*m, m*p, p*n))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def residuals(params):
        # params: flattened U, V, Z
        size_U = n*m * rank_target
        size_V = m*p * rank_target
        # size_Z = p*n * (rank_target - n)
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        Z_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, rank_target)
        V = V_flat.reshape(m*p, rank_target)
        
        # Construct W
        W = np.zeros((p*n, rank_target))
        # First n columns? No, first p*n columns?
        # W is (p*n, rank_target) = (6, 17).
        # We set the first 6 columns to Identity?
        # I_6 is 6x6. So W[:, 0:6] = I_6.
        # This means W[w, w] = 1 for w in 0..5.
        # And W[:, 6:17] = Z.
        
        # Check dimensions:
        # p*n = 6.
        # rank_target = 17.
        # So W has 6 rows.
        # We want W[:, 0:6] to be I_6.
        # This requires rank_target >= 6. 17 >= 6. OK.
        
        W[np.arange(p*n), np.arange(p*n)] = 1.0
        # Wait, if p*n = 6, then W[0,0]=1, W[1,1]=1... W[5,5]=1.
        # This sets the first 6 columns to Identity.
        
        # Fill Z
        # Z is (6, 11).
        # W[:, 6:17] = Z.
        Z = Z_flat.reshape(p*n, rank_target - p*n)
        W[:, p*n:rank_target] = Z
        
        # Compute predicted tensor S
        # S[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Residuals
        res = (S - T).flatten()
        
        return res

    # Initial guess
    np.random.seed(42)
    
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    best_params = None
    best_res_norm = np.inf
    
    # Try a few restarts
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Initialize U, V, Z
        # Small random values
        U_init = np.random.randn(n*m, rank_target) * 0.1
        V_init = np.random.randn(m*p, rank_target) * 0.1
        Z_init = np.random.randn(p*n, rank_target - p*n) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), Z_init.flatten()])
        
        try:
            sol = least_squares(residuals, params_init, method='lm', max_nfev=2000)
            if sol.cost < best_res_norm:
                best_res_norm = sol.cost
                best_params = sol.x
                # If we get very close, we can break early
                if sol.cost < 1e-8:
                    break
        except Exception:
            pass
            
    # Check if best_params gives a good solution
    if best_params is not None and best_res_norm < 1e-6:
        # Extract U, V, Z
        size_U = n*m * rank_target
        size_V = m*p * rank_target
        U_flat = best_params[:size_U]
        V_flat = best_params[size_U:size_U+size_V]
        Z_flat = best_params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, rank_target)
        V = V_flat.reshape(m*p, rank_target)
        
        W = np.zeros((p*n, rank_target))
        W[np.arange(p*n), np.arange(p*n)] = 1.0
        Z = Z_flat.reshape(p*n, rank_target - p*n)
        W[:, p*n:rank_target] = Z
        
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify exact decomposition with rounded values
        S_check = np.einsum('ur,vr,wr->uvw', U_rounded, V_rounded, W_rounded)
        if np.allclose(S_check, T, atol=1e-9):
            def to_list(mat):
                return [[float(x) for x in row] for row in mat]
            
            return {"U": to_list(U_rounded), "V": to_list(V_rounded), "W": to_list(W_rounded)}

    # Fallback to rank 18
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
