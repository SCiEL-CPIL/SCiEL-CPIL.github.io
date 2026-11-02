import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][column] = 1
                V_seed[j * p + k][column] = 1
                W_seed[k * n + i][column] = 1
                column += 1
    seed_result = {"U": U_seed, "V": V_seed, "W": W_seed}
    
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Try to find a lower rank decomposition
    # Target rank 45
    target_rank = 45
    
    # Indices for the 54 non-zero entries
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
                
    num_eqs = len(indices)
    
    def residuals(params):
        # params: flattened U, V, W
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        res = np.zeros(num_eqs)
        for idx, (a, b, c) in enumerate(indices):
            res[idx] = np.sum(U[a, :] * V[b, :] * W[c, :]) - 1.0
        return res
    
    # Initial guess
    np.random.seed(42)
    # Try a few random initializations
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Random initialization
        U_init = np.random.randn(n * m, target_rank) * 0.1
        V_init = np.random.randn(m * p, target_rank) * 0.1
        W_init = np.random.randn(p * n, target_rank) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, params_init, method='lm', max_nfev=5000)
            if result.success or result.cost < 1e-8:
                # Extract solution
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                size_W = p * n * target_rank
                
                U_sol = result.x[:size_U].reshape(n * m, target_rank)
                V_sol = result.x[size_U:size_U + size_V].reshape(m * p, target_rank)
                W_sol = result.x[size_U + size_V:].reshape(p * n, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify exactness
                valid = True
                for a, b, c in indices:
                    val = np.sum(U_rounded[a, :] * V_rounded[b, :] * W_rounded[c, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass
            
    return seed_result
# EVOLVE-BLOCK-END
