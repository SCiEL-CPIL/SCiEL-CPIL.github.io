"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 115  # 115s budget, 5s safety margin
    
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
        return U, V, W, rank
    
    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact using vectorized operations."""
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]  # shape (m, rank)
                V_cols = np.array([V[j*p + k, :] for j in range(m)])  # shape (m, rank)
                W_val = W[k*n + i, :]  # shape (rank,)
                temp = U_slice * W_val  # shape (m, rank)
                result = temp @ V_cols.T  # shape (m, m)
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True
    
    def try_rank(r, num_restarts=30):
        """Try to find a valid decomposition with rank r."""
        # Precompute index arrays
        u_idx = np.arange(n * m)
        v_idx = np.arange(m * p)
        i_u = u_idx // m
        j_u = u_idx % m
        j_v = v_idx // p
        k_v = v_idx % p
        
        # Precompute b vectors for all w values
        # b[w, (u,v)] = 1 iff j_u[u] == j_v[v] and i_u[u] == w%n and k_v[v] == w//n
        b_all = np.zeros((p * n, n * m, m * p))
        for w in range(p * n):
            i_out = w % n
            k_out = w // n
            # Vectorized: b_2d[u,v] = 1 iff j_u[u]==j_v[v] and i_u[u]==i_out and k_v[v]==k_out
            mask = (j_u[:, None] == j_v[None, :]) & (i_u[:, None] == i_out) & (k_v[None, :] == k_out)
            b_all[w] = mask.astype(float)
        b_flat = b_all.reshape(p * n, n * m * m * p)  # shape (36, 576)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Build A once: shape (576, r), A[(u,v), c] = U[u,c] * V[v,c]
            A = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
            
            # Solve for all W rows at once using least squares
            # We need: A @ W[w,:] = b_flat[w,:] for each w
            # This is equivalent to: for each w, solve A x = b_flat[w]
            # We can do this by stacking: but each w has different b
            # So we solve 36 separate systems, but A is shared
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                b = b_flat[w]
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None
    
    # Get schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try lower ranks
    # Lower bound: max(n*m, m*p, p*n) = 36
    # Try from 36 upward
    for r in range(36, 144):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_restarts=15)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
