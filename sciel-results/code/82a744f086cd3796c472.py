import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = 3, 4, 4
    
    # We'll try to find a low-rank decomposition for 3x4x4 matrix multiplication
    # T[i,j,k] = A[i,j] * B[j,k] where i in 0..2, j in 0..3, k in 0..3
    # U shape: (n*m, r) = (12, r)
    # V shape: (m*p, r) = (16, r)  
    # W shape: (p*n, r) = (12, r)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = 1.0  # T[i,j,k] = A[i,j]*B[j,k], coefficient is 1
    
    # Try to find a rank-r decomposition using iterative optimization
    # We'll try ranks from low to high
    
    def try_rank(r, max_iter=5000, tol=1e-8):
        """Try to find a rank-r decomposition. Returns (U,V,W) if successful, else None."""
        if time.time() > deadline - 5:
            return None
            
        # Initialize with random values
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Alternating least squares approach
        # For each r, fix U and V, solve for W; fix U and W, solve for V; etc.
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
                
            # Solve for W: for each k, T[:, :, k] = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r]
            # Actually, let's think about this more carefully.
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # For fixed U and V, W is determined by:
            # T[i,j,k] = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
            # This is a linear system in W for each (i,k) pair across j.
            
            # Let's reshape: for fixed i,k, T[i,:,k] (length m) = sum_r W[k*n+i, r] * (U[i*m: i*m+m, r] * V[:, k]_reshaped)
            # V[j*p+k, r] for j=0..m-1 gives a vector of length m for fixed k,r
            # U[i*m+j, r] for j=0..m-1 gives a vector of length m for fixed i,r
            
            # So T[i,:,k] = sum_r W[k*n+i, r] * (U[i*m:(i+1)*m, r] * V[::p, k]_reshaped)
            # Wait, V[j*p+k, r] for j=0..m-1: indices are k, p+k, 2p+k, ...
            
            # Let me build the design matrix for W
            # For each (i,k), we have m equations:
            # T[i,j,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
            
            # Let A_ik be an m x r matrix where A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then T[i,:,k] = A_ik @ W[k*n+i, :]
            
            # Solve: W[k*n+i, :] = (A_ik^T A_ik)^{-1} A_ik^T T[i,:,k]
            
            W_new = np.zeros((p*n, r))
            for i in range(n):
                for k in range(p):
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        for r_idx in range(r):
                            A_ik[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                    # Solve A_ik @ W[k*n+i, :] = T[i,:,k]
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_ik, T[i,:,k], rcond=None)[0]
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            # Similarly solve for V
            V_new = np.zeros((m*p, r))
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # For fixed j,k: T[:,j,k] (length n) = sum_r V[j*p+k, r] * (U[:,j]_reshaped * W[:,k]_reshaped)
                    # U[i*m+j, r] for i=0..n-1: indices are j, m+j, 2m+j, ...
                    # W[k*n+i, r] for i=0..n-1: indices are k*n, k*n+1, ..., k*n+n-1
                    A_jk = np.zeros((n, r))
                    for i in range(n):
                        for r_idx in range(r):
                            A_jk[i, r_idx] = U[i*m+j, r_idx] * W[k*n+i, r_idx]
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_jk, T[:,j,k], rcond=None)[0]
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            # Similarly solve for U
            U_new = np.zeros((n*m, r))
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # For fixed i,j: T[i,j,:] (length p) = sum_r U[i*m+j, r] * (V[j*p: j*p+p, r] * W[k*n+i, r] for k)
                    # V[j*p+k, r] for k=0..p-1
                    # W[k*n+i, r] for k=0..p-1
                    A_ij = np.zeros((p, r))
                    for k in range(p):
                        for r_idx in range(r):
                            A_ij[k, r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_ij, T[i,j,:], rcond=None)[0]
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            # Compute error
            U, V, W = U_new, V_new, W_new
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        error += (val - T[i,j,k])**2
            error = np.sqrt(error)
            
            if error < tol:
                return U, V, W
            
            # Early stopping if error isn't decreasing
            if it > 100 and it % 100 == 0:
                pass  # Continue for now
        
        # Final error check
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    error += (val - T[i,j,k])**2
        error = np.sqrt(error)
        
        if error < tol:
            return U, V, W
        return None
    
    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But tensor rank can be higher. Let's try 16, 17, 18, 19, 20, ...
    
    best_result = None
    best_rank = None
    
    for r in range(16, 30):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=3000, tol=1e-8)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = r
            break  # Found a solution, this is the minimum rank we tried
    
    # If no solution found via optimization, fall back to a known construction
    if best_result is None:
        # Use the block-based approach with rank 44
        rank = 44
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Local factors for 3x2x2 multiplication (rank 11)
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6) for 2x2x2
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
        
        # Extra part (cols 7-10) for Row 2 of A (the third row)
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
            
            for r_idx in range(3):
                for c in range(2):
                    local_a_idx = r_idx * 2 + c
                    global_a_row = a_row_start + r_idx
                    global_a_col = a_col_start + c
                    global_a_idx = global_a_row * m + global_a_col
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            
            for r_idx in range(2):
                for c in range(2):
                    local_b_idx = r_idx * 2 + c
                    global_b_row = b_row_start + r_idx
                    global_b_col = b_col_start + c
                    global_b_idx = global_b_row * p + global_b_col
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            
            for r_idx in range(3):
                for c in range(2):
                    local_c_idx = r_idx * 2 + c
                    global_c_row = c_row_start + r_idx
                    global_c_col = c_col_start + c
                    global_c_idx = global_c_col * n + global_c_row
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        best_result = (U, V, W)
        best_rank = 44
    
    U, V, W = best_result
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
