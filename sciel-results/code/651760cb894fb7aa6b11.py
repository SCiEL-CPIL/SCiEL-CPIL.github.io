import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank_target = 43
    
    # Generate valid indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    valid_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_terms.append((a, b, c))
                
    invalid_terms = []
    all_a = np.arange(n * m)
    all_b = np.arange(m * p)
    all_c = np.arange(p * n)
    valid_set = set(valid_terms)
    for a in all_a:
        for b in all_b:
            for c in all_c:
                if (a, b, c) not in valid_set:
                    invalid_terms.append((a, b, c))
    
    # Precompute arrays for faster evaluation
    valid_a = np.array([t[0] for t in valid_terms])
    valid_b = np.array([t[1] for t in valid_terms])
    valid_c = np.array([t[2] for t in valid_terms])
    invalid_a = np.array([t[0] for t in invalid_terms])
    invalid_b = np.array([t[1] for t in invalid_terms])
    invalid_c = np.array([t[2] for t in invalid_terms])
    
    def compute_error(U, V, W):
        """Compute total squared error using vectorized operations."""
        # For valid terms: sum_t U[a,t] * V[b,t] * W[c,t]
        # We can compute this as element-wise products summed over t
        # U[valid_a, :] has shape (num_valid, rank)
        # V[valid_b, :] has shape (num_valid, rank)  
        # W[valid_c, :] has shape (num_valid, rank)
        valid_vals = np.sum(U[valid_a, :] * V[valid_b, :] * W[valid_c, :], axis=1)
        invalid_vals = np.sum(U[invalid_a, :] * V[invalid_b, :] * W[invalid_c, :], axis=1)
        error = np.sum((valid_vals - 1.0) ** 2) + np.sum(invalid_vals ** 2)
        return error
    
    def try_round_and_verify(U, V, W, tolerance=1e-8):
        """Try to round to half-integers and verify exactness."""
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify valid terms
        valid_vals = np.sum(U_int[valid_a, :] * V_int[valid_b, :] * W_int[valid_c, :], axis=1)
        if not np.allclose(valid_vals, 1.0, atol=tolerance):
            return None
            
        # Verify invalid terms
        invalid_vals = np.sum(U_int[invalid_a, :] * V_int[invalid_b, :] * W_int[invalid_c, :], axis=1)
        if not np.allclose(invalid_vals, 0.0, atol=tolerance):
            return None
            
        return {
            "U": U_int.tolist(),
            "V": V_int.tolist(),
            "W": W_int.tolist()
        }
    
    start_time = time.time()
    deadline = 130  # 20s margin from 150s limit
    
    best_result = None
    best_error = float('inf')
    
    # Multiple restarts with gradient descent
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization with different scales
        scale = 1.0
        U = np.random.randn(n * m, rank_target) * scale
        V = np.random.randn(m * p, rank_target) * scale
        W = np.random.randn(p * n, rank_target) * scale
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # dE/dU[a,t] = 2 * sum over valid (val-1) * V[b,t]*W[c,t] + 2 * sum over invalid val * V[b,t]*W[c,t]
            # For efficiency, compute in chunks
            
            # Valid contribution
            valid_vals = np.sum(U[valid_a, :] * V[valid_b, :] * W[valid_c, :], axis=1)
            invalid_vals = np.sum(U[invalid_a, :] * V[invalid_b, :] * W[invalid_c, :], axis=1)
            
            # Gradient for U
            grad_U = np.zeros_like(U)
            # Valid terms: dE/dU[a,t] += 2*(val-1)*V[b,t]*W[c,t]
            for idx in range(len(valid_a)):
                a, b, c = valid_a[idx], valid_b[idx], valid_c[idx]
                grad_U[a, :] += 2.0 * (valid_vals[idx] - 1.0) * V[b, :] * W[c, :]
            # Invalid terms
            for idx in range(len(invalid_a)):
                a, b, c = invalid_a[idx], invalid_b[idx], invalid_c[idx]
                grad_U[a, :] += 2.0 * invalid_vals[idx] * V[b, :] * W[c, :]
            
            # Gradient for V
            grad_V = np.zeros_like(V)
            for idx in range(len(valid_a)):
                a, b, c = valid_a[idx], valid_b[idx], valid_c[idx]
                grad_V[b, :] += 2.0 * (valid_vals[idx] - 1.0) * U[a, :] * W[c, :]
            for idx in range(len(invalid_a)):
                a, b, c = invalid_a[idx], invalid_b[idx], invalid_c[idx]
                grad_V[b, :] += 2.0 * invalid_vals[idx] * U[a, :] * W[c, :]
            
            # Gradient for W
            grad_W = np.zeros_like(W)
            for idx in range(len(valid_a)):
                a, b, c = valid_a[idx], valid_b[idx], valid_c[idx]
                grad_W[c, :] += 2.0 * (valid_vals[idx] - 1.0) * U[a, :] * V[b, :]
            for idx in range(len(invalid_a)):
                a, b, c = invalid_a[idx], invalid_b[idx], invalid_c[idx]
                grad_W[c, :] += 2.0 * invalid_vals[idx] * U[a, :] * V[b, :]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if step % 50 == 0:
                lr *= 0.95
        
        # Check if we found a good solution
        error = compute_error(U, V, W)
        if error < best_error:
            best_error = error
        
        if error < 1e-4:
            result = try_round_and_verify(U, V, W)
            if result is not None:
                return result
    
    # Fallback to Rank 44
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
