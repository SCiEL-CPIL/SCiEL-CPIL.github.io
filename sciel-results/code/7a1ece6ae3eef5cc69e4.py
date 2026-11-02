import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Identity matrix for the inner product dimension m
    I_m = np.eye(m)
    
    # Precompute indices
    # We need to compute M_{i,k} = sum_r W[k*n+i, r] outer(U[i*m+j', r], V[j*p+k, r])?
    # No, M_{i,k} is a matrix of size m x m.
    # M_{i,k}[j', j] = sum_r W[k*n+i, r] * U[i*m+j', r] * V[j*p+k, r]
    
    # Residual function
    def residual(x):
        # x is flattened U, V, W
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Split x
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = x[:size_U].reshape(size_U, -1)
        V = x[size_U:size_U + size_V].reshape(size_V, -1)
        W = x[size_U + size_V:].reshape(size_W, -1)
        
        R = U.shape[1]
        
        # Compute residuals for all i, k
        res = []
        for i in range(n):
            for k in range(p):
                # M_{i,k} is m x m
                # M_{i,k}[j', j] = sum_r W[k*n+i, r] * U[i*m+j', r] * V[j*p+k, r]
                # We can compute this as:
                # Let U_i be U[i*m : (i+1)*m, :] (m x R)
                # Let V_k be V[k*p : (k+1)*p, :]? No, V index is j*p+k.
                # So V_k is not contiguous in V if we reshape by j then k.
                # V is (m*p, R). Index is j*p+k.
                # So for fixed k, the rows are k, p+k, 2p+k, ...
                # We need to extract these rows.
                
                # Extract U_i: rows i*m to (i+1)*m
                U_i = U[i*m : (i+1)*m, :] # m x R
                
                # Extract V_k: rows k, p+k, 2p+k, ... (m rows)
                # Indices: np.arange(m)*p + k
                V_k_indices = np.arange(m) * p + k
                V_k = V[V_k_indices, :] # m x R
                
                # W_val: scalar for this i, k
                W_val = W[k*n + i, :] # 1 x R
                
                # M_{i,k} = U_i * diag(W_val) * V_k^T ?
                # M[j', j] = sum_r U_i[j', r] * W_val[r] * V_k[j, r]
                # This is (U_i * W_val) @ V_k^T
                # U_i * W_val is (m x R) * (R) -> broadcast
                M_ik = (U_i * W_val) @ V_k.T # m x m
                
                res.append((M_ik - I_m).flatten())
        
        return np.concatenate(res)

    best_result = None
    best_rank = rank_school
    
    # Try to find lower rank
    # Start from a reasonable lower bound, e.g., 80
    # We can try a few values
    for R in range(90, 40, -5):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale to help convergence
            x0 = np.random.randn(n*m + m*p + p*n, R) * 0.1
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
                if sol.cost < 1e-10:
                    # Check if it's truly valid
                    res_val = residual(sol.x)
                    if np.max(np.abs(res_val)) < 1e-8:
                        best_rank = R
                        # Extract U, V, W
                        size_U = n * m
                        size_V = m * p
                        size_W = p * n
                        U_sol = sol.x[:size_U].reshape(size_U, R)
                        V_sol = sol.x[size_U:size_U + size_V].reshape(size_V, R)
                        W_sol = sol.x[size_U + size_V:].reshape(size_W, R)
                        
                        # Convert to list of lists
                        U_list = U_sol.tolist()
                        V_list = V_sol.tolist()
                        W_list = W_sol.tolist()
                        
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        # We found a lower rank, continue to see if we can go lower
                        break # Break attempt loop, try lower R
            except Exception as e:
                pass

    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
