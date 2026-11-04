import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Construct Schoolbook Decomposition (Rank 120)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    # Target Tensor T
    # T is a 3D tensor of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # 2. Try to find a decomposition of Rank 119
    # This is a heuristic attempt.
    # We use a random initialization and least_squares.
    # Given the complexity, we limit the time to 10 seconds.
    
    target_rank = 119
    if time.time() < deadline - 10.0:
        try:
            # Flatten U, V, W into a single vector
            # Sizes: 20, 30, 24
            size_u = n * m
            size_v = m * p
            size_w = p * n
            
            def residual(params):
                # Unpack
                u_flat = params[:size_u * target_rank]
                v_flat = params[size_u * target_rank : size_u * target_rank + size_v * target_rank]
                w_flat = params[size_u * target_rank + size_v * target_rank : ]
                
                U = u_flat.reshape(size_u, target_rank)
                V = v_flat.reshape(size_v, target_rank)
                W = w_flat.reshape(size_w, target_rank)
                
                # Compute sum of outer products
                # Result shape: (size_u, size_v, size_w)
                # R = sum_r U[:, r] outer V[:, r] outer W[:, r]
                # This is equivalent to np.einsum('ir,jr,kr->ijk', U, V, W)
                R = np.einsum('ir,jr,kr->ijk', U, V, W)
                
                return (R - T).flatten()
            
            # Initial guess: Random
            np.random.seed(42)
            x0 = np.random.randn(size_u * target_rank + size_v * target_rank + size_w * target_rank)
            
            # Run optimization
            result = least_squares(residual, x0, method='trf', max_nfev=500)
            
            if result.cost < 1e-8:
                # Valid decomposition found
                u_flat = result.x[:size_u * target_rank]
                v_flat = result.x[size_u * target_rank : size_u * target_rank + size_v * target_rank]
                w_flat = result.x[size_u * target_rank + size_v * target_rank : ]
                
                U = u_flat.reshape(size_u, target_rank)
                V = v_flat.reshape(size_v, target_rank)
                W = w_flat.reshape(size_w, target_rank)
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        except Exception:
            pass
            
    # 3. Fallback to Schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
