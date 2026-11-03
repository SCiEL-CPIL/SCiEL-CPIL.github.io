"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def schoolbook():
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        # Vectorized check
        # Construct the tensor T_decomp
        # T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient check using einsum or loops
        # Given small sizes, loops are fine, but let's try to be safe.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def als_step(U, V, W, n, m, p, r):
        """
        Perform one ALS iteration.
        Updates U, V, W to minimize the squared error of the tensor decomposition.
        """
        # Update U
        # For each (i, j), we solve for u = U[i*m+j]
        # Constraint: sum_r u_r * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1 ? No.
        # The equation is: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
        # Let A be a p x r matrix where A[k, r] = V[j*p+k, r] * W[k*n+i, r].
        # We want A @ u = ones(p).
        # Solve least squares: u = lstsq(A, ones(p))
        
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build A (p x r)
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # V slice for j: V[j*p : (j+1)*p, :] shape (p, r)
                # W slice for i: W[:, i] shape (p*n, r) -> we need rows k*n+i for k in 0..p-1
                # W rows are k*n + i.
                # W is (p*n, r).
                # We can extract W[k*n+i, :] for all k.
                # Indices: [0*n+i, 1*n+i, ..., (p-1)*n+i]
                w_idx = np.arange(p) * n + i
                w_slice = W[w_idx, :] # shape (p, r)
                v_slice = V[j*p : (j+1)*p, :] # shape (p, r)
                
                A = v_slice * w_slice # element-wise product, shape (p, r)
                b = np.ones(p)
                
                # Solve A u = b
                # Use lstsq for robustness
                try:
                    u_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_U[i*m+j, :] = u_sol
                except np.linalg.LinAlgError:
                    # Fallback: keep old value or set to zero
                    new_U[i*m+j, :] = U[i*m+j, :]
        
        # Update V
        # For each (j, k), solve for v = V[j*p+k]
        # Constraint: for each i, sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1.
        # Let A be n x r matrix where A[i, r] = U[i*m+j, r] * W[k*n+i, r].
        # We want A @ v = ones(n).
        
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # U rows: i*m + j for i in 0..n-1
                u_idx = np.arange(n) * m + j
                u_slice = U[u_idx, :] # shape (n, r)
                
                # W rows: k*n + i for i in 0..n-1
                w_idx = k * n + np.arange(n)
                w_slice = W[w_idx, :] # shape (n, r)
                
                A = u_slice * w_slice # shape (n, r)
                b = np.ones(n)
                
                try:
                    v_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_V[j*p+k, :] = v_sol
                except np.linalg.LinAlgError:
                    new_V[j*p+k, :] = V[j*p+k, :]
        
        # Update W
        # For each (k, i), solve for w = W[k*n+i]
        # Constraint: for each j, sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1.
        # Let A be m x r matrix where A[j, r] = U[i*m+j, r] * V[j*p+k, r].
        # We want A @ w = ones(m).
        
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # U rows: i*m + j for j in 0..m-1
                u_idx = i * m + np.arange(m)
                u_slice = U[u_idx, :] # shape (m, r)
                
                # V rows: j*p + k for j in 0..m-1
                v_idx = np.arange(m) * p + k
                v_slice = V[v_idx, :] # shape (m, r)
                
                A = u_slice * v_slice # shape (m, r)
                b = np.ones(m)
                
                try:
                    w_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_W[k*n+i, :] = w_sol
                except np.linalg.LinAlgError:
                    new_W[k*n+i, :] = W[k*n+i, :]
        
        return new_U, new_V, new_W

    def try_rank_als(r, max_iters=50):
        if time.time() > deadline:
            return None
        
        # Initialize randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U, V, W = als_step(U, V, W, n, m, p, r)
            
            # Check convergence or just proceed
            # Rounding and verification is expensive, do it periodically or at end
            if it % 10 == 0 or it == max_iters - 1:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, n, m, p):
                    return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
        
        # Final check
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if verify(U_rounded, V_rounded, W_rounded, n, m, p):
            return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
            
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to best_rank - 1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random restarts for each rank
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            res = try_rank_als(r)
            if res is not None:
                best_result = res
                best_rank = r
                found = True
                break
        
        if found:
            break # Found a valid decomposition for this rank, stop (since we go up from LB)
            
    return best_result
# EVOLVE-BLOCK-END
