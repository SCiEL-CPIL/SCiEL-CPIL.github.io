import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit
    
    # Baseline: schoolbook rank
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Target: identity matrix flattened
    b_id = np.eye(m).flatten()
    
    # Precompute index arrays for vectorized operations
    # For W update: for each (k,i), need A[j1*m+j2, r] = U[i*m+j1, r] * V[j2*p+k, r]
    # For U update: for each (i,j1), need A[k*m+j2, r] = W[k*n+i, r] * V[j2*p+k, r]
    # For V update: for each (j2,k), need A[i*m+j1, r] = W[k*n+i, r] * U[i*m+j1, r]
    
    # Precompute index mappings
    U_idx = np.arange(n*m).reshape(n, m)  # U_idx[i, j1] = i*m + j1
    V_idx = np.arange(m*p).reshape(m, p)  # V_idx[j2, k] = j2*p + k
    W_idx = np.arange(p*n).reshape(p, n)  # W_idx[k, i] = k*n + i
    
    def _verify_fast(U, V, W):
        """Fast verification using matrix operations."""
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        # For each (i,k), check: sum_r W[k*n+i,r] * U[i*m+j1,r] * V[j2*p+k,r] == delta_{j1,j2}
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # For each i,k: result[j1,j2] = sum_r W3[k,i,r] * U3[i,j1,r] * V3[j2,k,r]
        # = W3[k,i,:] outer U3[i,:,:] outer V3[:,k,:] contracted over r
        # Equivalent to: for each i,k: (U3[i,:,:] * W3[k,i,:]) @ V3[:,k,:].T
        # Wait: sum_r U3[i,j1,r] * V3[j2,k,r] * W3[k,i,r]
        # = (U3[i,:,:] * W3[k,i,:]) dot V3[:,k,:].T? No.
        # sum_r (U3[i,j1,r] * W3[k,i,r]) * V3[j2,k,r]
        # Let A[i,j1,r] = U3[i,j1,r] * W3[k,i,r] for fixed k,i
        # Then result[j1,j2] = sum_r A[i,j1,r] * V3[j2,k,r]
        # = (A[i,:,:] * 1) ... this is element-wise product summed over r
        # = np.einsum('j1r,j2r->j1j2', U3[i,:,:]*W3[k,i,:], V3[:,k,:])
        
        for i in range(n):
            for k in range(p):
                # Compute the m×m matrix
                # M[j1,j2] = sum_r U[i*m+j1,r] * V[j2*p+k,r] * W[k*n+i,r]
                u_block = U[i*m:(i+1)*m, :] * W[k*n+i, :]  # (m, r)
                v_block = V[np.arange(m)*p + k, :]  # (m, r)
                # M = u_block @ v_block.T  -- NO, that gives sum_r u[j1,r]*v[j2,r]
                # Actually: M[j1,j2] = sum_r u_block[j1,r] * v_block[j2,r] = (u_block @ v_block.T)[j1,j2]
                M = u_block @ v_block.T
                if not np.allclose(M, np.eye(m), atol=1e-6):
                    return False
        return True
    
    def _als_step(U, V, W, n_iters=50):
        """Run ALS optimization. Returns (U, V, W, error)."""
        for it in range(n_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Update W: for each (k,i)
            for k in range(p):
                for i in range(n):
                    # A[j1*m+j2, r] = U[i*m+j1, r] * V[j2*p+k, r]
                    U_block = U[i*m:(i+1)*m, :]  # (m, r)
                    V_block = V[np.arange(m)*p + k, :]  # (m, r)
                    # A = outer product structure: A[j1*m+j2, r] = U_block[j1,r] * V_block[j2,r]
                    A = (U_block[:, np.newaxis, :] * V_block[np.newaxis, :, :]).reshape(m*m, r)
                    w, *_ = np.linalg.lstsq(A, b_id, rcond=None)
                    W[k*n+i, :] = w
            
            # Update U: for each (i,j1)
            for i in range(n):
                for j1 in range(m):
                    # A[k*m+j2, r] = W[k*n+i, r] * V[j2*p+k, r]
                    A = np.zeros((p*m, r))
                    b = np.zeros(p*m)
                    for k in range(p):
                        for j2 in range(m):
                            A[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                            b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                    u, *_ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j1, :] = u
            
            # Update V: for each (j2,k)
            for j2 in range(m):
                for k in range(p):
                    # A[i*m+j1, r] = W[k*n+i, r] * U[i*m+j1, r]
                    A = np.zeros((n*m, r))
                    b = np.zeros(n*m)
                    for i in range(n):
                        for j1 in range(m):
                            A[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                            b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                    v, *_ = np.linalg.lstsq(A, b, rcond=None)
                    V[j2*p+k, :] = v
        
        # Compute final error
        err = 0.0
        for i in range(n):
            for k in range(p):
                u_block = U[i*m:(i+1)*m, :] * W[k*n+i, :]
                v_block = V[np.arange(m)*p + k, :]
                M = u_block @ v_block.T
                err += np.sum((M - np.eye(m))**2)
        
        return U, V, W, err
    
    # Strategy: try ranks from high to low, with more restarts at critical points
    # Based on literature and structure, try a focused set of ranks
    candidate_ranks = list(range(rank0 - 1, 50, -1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 20.0:
            break
        
        # More restarts for lower ranks
        n_restarts = 3 if r > 70 else 5 if r > 60 else 8
        
        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(9999 + r * 100 + restart)
            
            # Mixed initialization: some random, some structured
            if restart % 3 == 0:
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            elif restart % 3 == 1:
                # Sparse-ish initialization
                U = np.random.randn(n*m, r) * 0.3
                V = np.random.randn(m*p, r) * 0.3
                W = np.random.randn(p*n, r) * 0.3
                # Add some structure: make first few columns more prominent
                U[:, :min(5, r)] *= 1.5
                V[:, :min(5, r)] *= 1.5
                W[:, :min(5, r)] *= 1.5
            else:
                U = np.random.randn(n*m, r) * 0.4
                V = np.random.randn(m*p, r) * 0.4
                W = np.random.randn(p*n, r) * 0.4
            
            n_iters = 30 if r > 70 else 40 if r > 60 else 50
            U, V, W, err = _als_step(U, V, W, n_iters)
            
            if err < 1e-8:
                # Snap to half-integers and verify
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify_fast(U_r, V_r, W_r):
                    best_rank = r
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                    found = True
                    break
        
        if found:
            continue  # Try lower ranks
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
