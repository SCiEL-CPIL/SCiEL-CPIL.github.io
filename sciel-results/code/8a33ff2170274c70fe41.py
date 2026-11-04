import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, leave 5s margin
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    num_eqs = n * m * p # 84
    
    # Pre-compute index mappings for vectorization
    # e = i * (m * p) + j * p + k
    # idx_U[e] = i * m + j
    # idx_V[e] = j * p + k
    # idx_W_row[e] = k * n + i
    
    idx_U = np.zeros(num_eqs, dtype=int)
    idx_V = np.zeros(num_eqs, dtype=int)
    idx_W_row = np.zeros(num_eqs, dtype=int)
    
    e = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U[e] = i * m + j
                idx_V[e] = j * p + k
                idx_W_row[e] = k * n + i
                e += 1
                
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for better rank
    # Theoretical lower bound is max(dim_U, dim_V, dim_W) = 28.
    # Schoolbook is 84.
    # We try ranks from 28 upwards.
    
    # Improved search parameters
    # Larger integers reduce linear dependencies
    # More tries for lower ranks
    
    for r in range(28, 84):
        if time.time() > deadline - 5:
            break
            
        # Allocate more tries to lower ranks
        if r < 35:
            tries = 200
        elif r < 50:
            tries = 100
        else:
            tries = 50
            
        for _ in range(tries):
            if time.time() > deadline - 2:
                break
                
            # Random U and V with larger range
            U = np.random.randint(-10, 11, size=(dim_U, r))
            V = np.random.randint(-10, 11, size=(dim_V, r))
            
            # Build A vectorized
            U_exp = U[idx_U, :]
            V_exp = V[idx_V, :]
            
            Coeffs = U_exp * V_exp # Shape (84, r)
            
            A = np.zeros((num_eqs, dim_W * r))
            cols = idx_W_row[:, None] * r + np.arange(r)[None, :]
            A[np.arange(num_eqs)[:, None], cols] = Coeffs
            
            b = np.ones(num_eqs)
            
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=1e-12)
            except Exception:
                continue
                
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-8:
                continue
                
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check half-integer property
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-5:
                W_int = W_rounded / 2.0
                
                # Verify
                W_exp = W_flat.reshape(dim_W, r)[idx_W_row, :] # Shape (84, r)
                vals = np.sum(Coeffs * W_exp, axis=1)
                
                if np.all(np.abs(vals - 1.0) < 1e-5):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
                    
    return get_schoolbook()
