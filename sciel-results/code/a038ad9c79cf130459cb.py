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
        # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        
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
        # Row (u, v, w) has non-zeros at cols (r, w) for r in 0..R-1.
        # Value is U[u, r] * V[v, r].
        
        num_rows = dim_u * dim_v * dim_w
        num_cols = R * dim_w
        
        A = np.zeros((num_rows, num_cols), dtype=np.float64)
        
        for u in range(dim_u):
            for v in range(dim_v):
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
            
        # Try multiple random initializations with different strategies
        max_attempts = 80
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Different randomization strategies
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(dim_u, R), p=[0.3, 0.4, 0.3])
                V = np.random.choice([-1, 0, 1], size=(dim_v, R), p=[0.3, 0.4, 0.3])
            elif strategy == 1:
                # Denser {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(dim_u, R), p=[0.4, 0.2, 0.4])
                V = np.random.choice([-1, 0, 1], size=(dim_v, R), p=[0.4, 0.2, 0.4])
            elif strategy == 2:
                # Half-integer values {-1, -0.5, 0, 0.5, 1}
                vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                U = np.random.choice(vals, size=(dim_u, R))
                V = np.random.choice(vals, size=(dim_v, R))
            else:
                # Mixed: some structured, some random
                U = np.random.choice([-1, 0, 1], size=(dim_u, R), p=[0.3, 0.4, 0.3])
                V = np.random.choice([-1, 0, 1], size=(dim_v, R), p=[0.3, 0.4, 0.3])
                # Add some structure: make rows with same j more similar
                for j in range(m):
                    rows_u = [i * m + j for i in range(n)]
                    if len(rows_u) > 1:
                        # Average some entries
                        for r in range(R):
                            if np.random.random() < 0.3:
                                avg = np.mean(U[rows_u, r])
                                U[rows_u, r] = np.round(avg * 2) / 2 if abs(avg) > 0.1 else 0.0
                    
                    rows_v = [j * p + k for k in range(p)]
                    if len(rows_v) > 1:
                        for r in range(R):
                            if np.random.random() < 0.2:
                                avg = np.mean(V[rows_v, r])
                                V[rows_v, r] = np.round(avg * 2) / 2 if abs(avg) > 0.1 else 0.0
            
            # Solve for W
            W_flat, res = solve_for_W(U, V, R)
            
            if W_flat is None:
                continue
                
            # Check if W is close to half-integers
            W_rounded = np.round(W_flat * 2) / 2
            
            # Check if W_flat is close to W_rounded
            if not np.allclose(W_flat, W_rounded, atol=1e-3):
                # Try a local polish: adjust U, V slightly to make W more half-integer
                # This is a heuristic polish step
                if res < 1.0:  # Only polish if we're close
                    # Try to find small perturbations to U, V that make W more half-integer
                    polish_success = False
                    for polish_iter in range(20):
                        if time.time() > deadline - 3:
                            break
                        # Compute current residual for W half-integer constraint
                        W_err = W_flat - W_rounded
                        err_norm = np.linalg.norm(W_err)
                        
                        if err_norm < 1e-3:
                            polish_success = True
                            break
                        
                        # Try perturbing a random entry of U or V
                        if np.random.random() < 0.5:
                            mat = U
                            mat_name = 'U'
                        else:
                            mat = V
                            mat_name = 'V'
                        
                        row = np.random.randint(0, mat.shape[0])
                        col = np.random.randint(0, mat.shape[1])
                        
                        old_val = mat[row, col]
                        # Try small perturbation
                        new_val = np.round((old_val + np.random.uniform(-0.5, 0.5)) * 2) / 2
                        if new_val == old_val:
                            continue
                        mat[row, col] = new_val
                        
                        # Re-solve for W
                        W_flat_new, res_new = solve_for_W(U, V, R)
                        if W_flat_new is not None:
                            W_rounded_new = np.round(W_flat_new * 2) / 2
                            err_new = np.linalg.norm(W_flat_new - W_rounded_new)
                            if err_new < err_norm:
                                W_flat = W_flat_new
                                W_rounded = W_rounded_new
                                # Keep the perturbation
                            else:
                                # Revert
                                mat[row, col] = old_val
                        else:
                            mat[row, col] = old_val
                    
                    if not polish_success:
                        continue
                
                W = W_rounded.reshape(R, dim_w).T
                
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
            else:
                W = W_rounded.reshape(R, dim_w).T
                
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
