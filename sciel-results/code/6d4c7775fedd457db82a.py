import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 115
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor b
    # b index: (u, v, w) -> u * dim_v * dim_w + v * dim_w + w
    b = np.zeros(dim_u * dim_v * dim_w, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                idx = u * dim_v * dim_w + v * dim_w + w
                b[idx] = 1.0
                
    # Function to check if a decomposition is valid
    def check_valid(U, V, W, R):
        # U: (dim_u, R), V: (dim_v, R), W: (dim_w, R)
        # Compute T = sum_r U[:,r] V[:,r]^T ... wait.
        # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        
        # Vectorized check
        # Reshape U, V, W to broadcast
        # U: (dim_u, 1, 1, R)
        # V: (1, dim_v, 1, R)
        # W: (1, 1, dim_w, R)
        # Sum over R: (dim_u, dim_v, dim_w)
        
        U_exp = U[:, np.newaxis, np.newaxis, :]
        V_exp = V[np.newaxis, :, np.newaxis, :]
        W_exp = W[np.newaxis, np.newaxis, :, :]
        
        T = np.sum(U_exp * V_exp * W_exp, axis=3)
        
        # Flatten T and compare with b
        T_flat = T.flatten()
        if not np.allclose(T_flat, b, atol=1e-6):
            return False
        
        # Check if entries are half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-6):
                return False
        return True

    # Function to solve for W given U, V
    def solve_for_W(U, V, R):
        # A matrix: (dim_u * dim_v * dim_w, R * dim_w)
        # We only need to fill non-zero entries.
        # Row (u, v, w) has non-zeros at cols (r, w) for r in 0..R-1.
        # Value is U[u, r] * V[v, r].
        
        num_rows = dim_u * dim_v * dim_w
        num_cols = R * dim_w
        
        A = np.zeros((num_rows, num_cols), dtype=np.float64)
        
        # Iterate over u, v
        for u in range(dim_u):
            for v in range(dim_v):
                # Check if this (u,v) pair is "active" in the tensor?
                # No, A is built based on U, V.
                # For each w, the row is (u, v, w).
                # The non-zeros are at columns corresponding to w.
                
                # Col block for w is w * R + r ?
                # Let's define col index for (r, w) as r * dim_w + w.
                
                # For a fixed u, v, and w:
                # Row idx = u * dim_v * dim_w + v * dim_w + w
                # Non-zero cols: r * dim_w + w for r in 0..R-1
                # Values: U[u, r] * V[v, r]
                
                for w in range(dim_w):
                    row_idx = u * dim_v * dim_w + v * dim_w + w
                    for r in range(R):
                        val = U[u, r] * V[v, r]
                        if val != 0:
                            col_idx = r * dim_w + w
                            A[row_idx, col_idx] = val
        
        # Solve A x = b
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, np.inf
            
        # Check residual
        if len(residuals) > 0:
            res = residuals[0]
        else:
            res = np.linalg.norm(A @ x - b)
            
        return x, res

    best_rank = n * m * p
    best_result = None
    
    # Initialize schoolbook
    U_school = np.zeros((dim_u, best_rank))
    V_school = np.zeros((dim_v, best_rank))
    W_school = np.zeros((dim_w, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
                
    if check_valid(U_school, V_school, W_school, best_rank):
        best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    # Try to find lower rank
    # Search ranks from low to high
    for R in range(20, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random U, V
            # Use values from {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(dim_u, R))
            V = np.random.choice([-1, 0, 1], size=(dim_v, R))
            
            # Solve for W
            W_flat, res = solve_for_W(U, V, R)
            
            if W_flat is None:
                continue
                
            # Check if W is close to half-integers
            W_rounded = np.round(W_flat * 2) / 2
            
            # Check if W_flat is close to W_rounded
            if not np.allclose(W_flat, W_rounded, atol=1e-3):
                continue
                
            W = W_rounded.reshape(R, dim_w).T # W is (dim_w, R)
            
            # Check validity
            if check_valid(U, V, W, R):
                # Found a valid decomposition
                best_rank = R
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                return best_result
                
    # If no lower rank found, return schoolbook
    return best_result
