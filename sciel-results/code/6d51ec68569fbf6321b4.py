import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # Precompute index mapping for diagonal terms
    # a = i*m + j, b = j*p + k, c = k*n + i
    diag_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                diag_indices.append((a, b, c))
                
    # Create a mask for off-diagonal terms? 
    # Or just compute full tensor and check.
    # Full tensor size: 9 * 21 * 21 = 3969. Very small.
    
    def _compute_full_error(U, V, W):
        """Compute max deviation on diagonal and max absolute value on off-diagonal."""
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        max_diag_err = 0.0
        for a, b, c in diag_indices:
            err = abs(T[a, b, c] - 1.0)
            if err > max_diag_err:
                max_diag_err = err
        
        max_off_err = 0.0
        # Iterate over all indices and skip diagonal ones
        # To be faster, we can sum squares or just find max.
        # Since size is small, we can just loop or use masking.
        # Let's use a mask for off-diagonal.
        # But creating a mask every time might be slow? 
        # 3969 elements is nothing.
        
        # Alternative: 
        # The diagonal set is small (63 elements).
        # Total elements 3969.
        # We can compute sum of squares of all T, then subtract diagonal squares?
        # No, we need max absolute value.
        
        # Simple loop is fine for 3969 items.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) is in diag_indices
                    # This lookup is O(1) if we use a set, but set creation is overhead.
                    # Better: precompute a boolean mask or list of diagonal tuples.
                    pass
        
        # Let's just use the diagonal list to zero out in a copy?
        # Or simpler:
        # We know the structure.
        # i = a // m, j = a % m
        # j2 = b // p, k = b % p
        # k2 = c // n, i2 = c % n
        # If i==i2 and j==j2 and k==k2, it's diagonal.
        
        # Vectorized check:
        # a_idx = np.arange(n*m)[:, None, None]
        # b_idx = np.arange(m*p)[None, :, None]
        # c_idx = np.arange(p*n)[None, None, :]
        
        # i = a_idx // m
        # j = a_idx % m
        # j2 = b_idx // p
        # k = b_idx % p
        # k2 = c_idx // n
        # i2 = c_idx % n
        
        # is_diag = (i == i2) & (j == j2) & (k == k2)
        
        # This is efficient.
        
        a_idx = np.arange(n * m)
        b_idx = np.arange(m * p)
        c_idx = np.arange(p * n)
        
        i = a_idx[:, None, None] // m
        j = a_idx[:, None, None] % m
        j2 = b_idx[None, :, None] // p
        k = b_idx[None, :, None] % p
        k2 = c_idx[None, None, :] // n
        i2 = c_idx[None, None, :] % n
        
        is_diag = (i == i2) & (j == j2) & (k == k2)
        
        # Diagonal error
        diag_T = T[is_diag]
        max_diag_err = np.max(np.abs(diag_T - 1.0))
        
        # Off-diagonal error
        off_T = T[~is_diag]
        max_off_err = np.max(np.abs(off_T))
        
        return max_diag_err, max_off_err

    def _is_exact(U, V, W, tol=1e-8):
        max_diag_err, max_off_err = _compute_full_error(U, V, W)
        return max_diag_err < tol and max_off_err < tol

    # ALS with vectorized error check
    for R in range(trivial_rank - 1, 26, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(10): # Increased trials due to speed
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + R * 10 + trial)
            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3
            
            prev_err = float('inf')
            converged = False
            
            for iteration in range(100): # More iterations possible now
                if time.time() > deadline - 5:
                    break
                
                # Update U
                # For each i, j, solve A x = 1
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        # Construct A
                        # V_j = V[j*p : (j+1)*p, :] shape (p, R)
                        # W_i = W[:, i] ... wait W is (p*n, R). 
                        # W[k*n + i, :] for k in 0..p-1.
                        # This is a strided access.
                        # We can form A by element-wise multiplication of V_j and W_i_col.
                        
                        V_j = V[j*p : (j+1)*p, :] # (p, R)
                        # W_k_i for each k
                        # W[k*n + i, :]
                        # Indices: i, n+i, 2n+i, ...
                        W_i_cols = W[i::n, :] # (p, R)
                        
                        A = V_j * W_i_cols # (p, R)
                        b = np.ones(p)
                        
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # U_i_j for each i
                        U_i_j_cols = U[:, j] # (n, R) ? 
                        # U[i*m + j, :]
                        # Indices: j, m+j, 2m+j
                        U_j_cols = U[j::m, :] # (n, R)
                        
                        W_k_i = W[k*n : (k+1)*n, :] # (n, R)
                        
                        B = U_j_cols * W_k_i # (n, R)
                        b = np.ones(n)
                        
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # U_i_j for each j
                        U_i_row = U[i*m : (i+1)*m, :] # (m, R)
                        
                        # V_j_k for each j
                        # V[j*p + k, :]
                        # Indices: k, p+k, 2p+k
                        V_k_cols = V[k::p, :] # (m, R)
                        
                        C = U_i_row * V_k_cols # (m, R)
                        b = np.ones(m)
                        
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Check convergence
                max_diag_err, max_off_err = _compute_full_error(U, V, W)
                err = max_diag_err + max_off_err
                
                if err < 1e-8:
                    converged = True
                    break
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
            
            if converged:
                if _is_exact(U, V, W):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    found = True
                    break
        
        if found:
            continue
        else:
            if time.time() > deadline - 2:
                break

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
