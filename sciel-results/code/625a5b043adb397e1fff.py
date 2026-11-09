import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def verify(U, V, W, tol=1e-9):
        """
        Verify that the decomposition exactly reproduces matrix multiplication.
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        Convention: W[k*n+i, r] where C[i,k] = sum_r (sum_{a,b} U[a*m+b, r]*A[a,b]) 
                                            (sum_{b,c} V[b*p+c, r]*B[b,c]) * W[c*n+i, r]
        """
        # Use deterministic random matrices for verification to avoid flakiness
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_calc = np.zeros((n, p))
        R = U.shape[1]
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)  # W is (p*n, R), reshape to (p, n)
            
            scalar_u = np.sum(u_r * A)
            scalar_v = np.sum(v_r * B)
            
            # C[i, k] += scalar_u * scalar_v * w_r[k, i]
            # So C += scalar_u * scalar_v * w_r.T
            C_calc += (scalar_u * scalar_v) * w_r.T
        
        if np.allclose(C_true, C_calc, atol=tol):
            return True
        return False

    def build_schoolbook():
        """Build the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook
    U_best, V_best, W_best = build_schoolbook()
    best_rank = n * m * p

    # ALS search for lower rank
    # We try ranks from 104 down to 80
    # We use a limited number of restarts and iterations
    
    def als_decompose(target_rank, max_iters=100, num_samples=10):
        if time.time() > deadline - 10:
            return None
            
        np.random.seed(target_rank)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Generate fixed random samples for ALS
        A_samples = np.random.rand(num_samples, n, m)
        B_samples = np.random.rand(num_samples, m, p)
        C_samples = A_samples @ B_samples
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
                
            # 1. Update U
            # Fix V, W
            # Compute beta_{s,r} = V_r . B_s
            # beta shape: (num_samples, target_rank)
            # V_r is (m*p,). B_s is (m*p,).
            # V is (m*p, R). B_samples is (S, m, p) -> (S, m*p).
            B_flat = B_samples.reshape(num_samples, m * p)
            beta = B_flat @ V  # (S, R)
            
            # Solve for alpha_{s,r}
            # C_s = sum_r alpha_{s,r} beta_{s,r} W_r^T
            # Flatten C_s to vector c_s (n*p,)
            C_flat = C_samples.reshape(num_samples, n * p)
            
            # Construct matrix M for least squares
            # We want to find alpha (S, R) such that C_flat @ alpha.T * beta? No.
            # C_s = sum_r alpha_{s,r} (beta_{s,r} * vec(W_r^T))
            # Let w_vec_r = vec(W_r^T) (n*p,)
            # W is (p*n, R). Reshape to (p, n, R) then transpose last two?
            # W[:, r] is (p*n,). Reshape to (p, n). Transpose to (n, p). Flatten to (n*p,).
            W_vecs = np.zeros((target_rank, n * p))
            for r in range(target_rank):
                w_r = W[:, r].reshape(p, n).T.reshape(-1)
                W_vecs[r, :] = w_r
            
            # For each sample s:
            # c_s = sum_r alpha_{s,r} beta_{s,r} W_vecs[r]
            # This is a linear system in alpha_{s,r}.
            # Let M_s be a matrix of size (n*p, R) where M_s[:, r] = beta_{s,r} * W_vecs[r]
            # Then c_s = M_s @ alpha_s
            # We can solve this for each s, or jointly.
            # Jointly: Stack all c_s into a vector C_all (S*n*p,).
            # Stack all M_s into a block diagonal matrix? That's huge.
            # Better: Solve for each s independently? Or use vectorized operations.
            # Since S is small (10), we can loop over s.
            
            alpha = np.zeros((num_samples, target_rank))
            for s in range(num_samples):
                M_s = np.outer(W_vecs.T, beta[s]) # (R, n*p) -> wait.
                # M_s should be (n*p, R).
                # M_s[:, r] = beta[s, r] * W_vecs[r]
                M_s = W_vecs.T * beta[s] # (n*p, R)
                # Solve M_s @ alpha[s] = c_s
                # Use least squares
                try:
                    alpha[s], _, _, _ = np.linalg.lstsq(M_s, C_flat[s], rcond=None)
                except:
                    alpha[s] = np.zeros(target_rank)
            
            # Now solve for U
            # Constraints: sum_{i,j} U_{ij,r} A_{s,ij} = alpha_{s,r}
            # For each r, we have a system:
            # A_flat_s @ U_r = alpha_{s,r}
            # A_flat_s is (n*m,). U_r is (n*m,).
            # We have S equations for R variables (U_r).
            # Solve least squares for U_r.
            
            A_flat = A_samples.reshape(num_samples, n * m)
            U_new = np.zeros((n * m, target_rank))
            for r in range(target_rank):
                # A_flat @ U_new[:, r] = alpha[:, r]
                # A_flat is (S, n*m). U_new[:, r] is (n*m,).
                # This is (S, 1) = (S, n*m) @ (n*m, 1)
                try:
                    U_new[:, r], _, _, _ = np.linalg.lstsq(A_flat, alpha[:, r], rcond=None)
                except:
                    U_new[:, r] = np.zeros(n * m)
            U = U_new
            
            # 2. Update V
            # Fix U, W
            # Compute alpha_{s,r} = U_r . A_s
            # U is (n*m, R). A_flat is (S, n*m).
            alpha = A_flat @ U # (S, R)
            
            # Solve for gamma_{s,r} (which is V_r . B_s)
            # C_s = sum_r alpha_{s,r} gamma_{s,r} W_vecs[r]
            # M_s[:, r] = alpha_{s,r} * W_vecs[r]
            gamma = np.zeros((num_samples, target_rank))
            for s in range(num_samples):
                M_s = W_vecs.T * alpha[s] # (n*p, R)
                try:
                    gamma[s], _, _, _ = np.linalg.lstsq(M_s, C_flat[s], rcond=None)
                except:
                    gamma[s] = np.zeros(target_rank)
            
            # Solve for V
            # B_flat @ V_new[:, r] = gamma[:, r]
            V_new = np.zeros((m * p, target_rank))
            for r in range(target_rank):
                try:
                    V_new[:, r], _, _, _ = np.linalg.lstsq(B_flat, gamma[:, r], rcond=None)
                except:
                    V_new[:, r] = np.zeros(m * p)
            V = V_new
            
            # 3. Update W
            # Fix U, V
            # Compute alpha_{s,r}, beta_{s,r}
            alpha = A_flat @ U # (S, R)
            beta = B_flat @ V # (S, R)
            
            # Solve for W
            # C_s = sum_r (alpha_{s,r} * beta_{s,r}) W_r^T
            # For each entry (i, k) of C (which is index idx in 0..n*p-1):
            # C_s[idx] = sum_r (alpha_{s,r} * beta_{s,r}) W_r[idx]
            # Let coeff_{s,r} = alpha_{s,r} * beta_{s,r}
            coeff = alpha * beta # (S, R)
            
            W_new = np.zeros((p * n, target_rank))
            # We solve for each column of W_new (which corresponds to r? No, W_new is (p*n, R))
            # W_new[:, r] is the vector for the r-th term.
            # The equation is:
            # C_flat[s] = sum_r coeff[s, r] * W_new[:, r]
            # This is a linear system in the vectors W_new[:, r].
            # We can solve for each entry idx of W_new[:, r] independently?
            # C_flat[s, idx] = sum_r coeff[s, r] * W_new[idx * n + ? , r]
            # Wait, W is (p*n, R). Index is k*n + i.
            # C is (n, p). Index is i*p + k? Or k*n + i?
            # The convention is W[k*n+i, r].
            # C[i, k] corresponds to W[k*n+i, r].
            # So for a fixed (i, k), we have a system for W[k*n+i, r] for r=1..R.
            
            for i in range(n):
                for k in range(p):
                    idx = k * n + i
                    c_col = C_flat[:, i * p + k] # (S,)
                    # Solve: c_col = coeff @ W_new[idx, :]
                    # coeff is (S, R). W_new[idx, :] is (R,).
                    try:
                        W_new[idx, :], _, _, _ = np.linalg.lstsq(coeff, c_col, rcond=None)
                    except:
                        W_new[idx, :] = np.zeros(target_rank)
            W = W_new
            
        # Round to nearest integer or half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        if verify(U, V, W):
            return U, V, W
        return None

    # Try ranks from 104 down to 80
    for target_rank in range(104, 79, -1):
        if time.time() > deadline - 5:
            break
            
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            result = als_decompose(target_rank)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = target_rank
                break
        
        if best_rank < n * m * p:
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
