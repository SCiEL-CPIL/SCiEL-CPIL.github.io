"""Matrix multiplication tensor decomposition 3x5x6 - rank minimization."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    best_rank = n * m * p
    best_result = None
    
    # Generate the target tensor T[i,j,k] = 1 (it's the multiplication tensor)
    # T has shape (n, m, p), all entries are 1 for the "multiplication" part
    # Actually T[i,j,k] = 1 for all i,j,k - it's the all-ones tensor
    # Wait - the multiplication tensor T_{ijk} = 1 for all i,j,k
    # Because C[i,k] = sum_j A[i,j] * B[j,k]
    
    T = np.ones((n, m, p))
    
    def verify(U, V, W):
        """Check if U @ outer decomposition equals T."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Decomposition: sum_c U[:,c] * V[:,c] * W[:,c] should give T
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # Contract: sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum
        result = np.einsum('ijc,jkc,kic->ik', U3, V3, W3)
        # But we need to check against T reshaped appropriately
        # T[i,j,k] = 1, so we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let's verify differently: for each (i,j,k), the product should be 1
        # Actually the tensor T has T[i,j,k]=1 for all i,j,k
        # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    error += abs(s - 1)
        return error < 1e-6
    
    def round_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2.0
    
    def try_rank(r, max_iters=200, n_restarts=5):
        """Try to find rank-r decomposition via ALS."""
        nonlocal best_rank, best_result
        if r >= best_rank:
            return
        if time.time() > deadline:
            return
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            
            # Initialize
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each row i*m+j, minimize ||sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1||
                # This is: for fixed (i,j), we want sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # Let M_k = V[j*p+k] * W[k*n+i] (element-wise product, shape r)
                # We want U[i*m+j] @ M_k^T = 1 for all k
                # So U[i*m+j] is in the row space of {M_k}
                # Solve least squares: U[i*m+j] * M^T = ones(p) where M is (p, r)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k] = V[j*p+k] * W[k*n+i]
                        # Solve U[i*m+j] @ M.T = ones(p)
                        # i.e., M @ U[i*m+j].T = ones(p)
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U[i*m+j] = sol
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i] = U[i*m+j] * W[k*n+i]
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V[j*p+k] = sol
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j] = U[i*m+j] * V[j*p+k]
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W[k*n+i] = sol
            
            if time.time() > deadline:
                break
            
            # Try rounding to half-integers
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            
            if verify(U_r, V_r, W_r):
                best_rank = r
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return
            
            # Also try rounding to integers
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify(U_i, V_i, W_i):
                best_rank = r
                best_result = {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                return
        
        # Try structured initializations
        for restart in range(3):
            if time.time() > deadline:
                return
            # Sparse random init
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            for c in range(r):
                idx_u = np.random.choice(n*m, size=2, replace=False)
                idx_v = np.random.choice(m*p, size=2, replace=False)
                idx_w = np.random.choice(p*n, size=2, replace=False)
                U[idx_u[0], c] = 1
                U[idx_u[1], c] = 1
                V[idx_v[0], c] = 1
                V[idx_v[1], c] = 1
                W[idx_w[0], c] = 1
                W[idx_w[1], c] = 1
            
            # Quick ALS
            for it in range(50):
                if time.time() > deadline:
                    break
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k] = V[j*p+k] * W[k*n+i]
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U[i*m+j] = sol
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i] = U[i*m+j] * W[k*n+i]
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V[j*p+k] = sol
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j] = U[i*m+j] * V[j*p+k]
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W[k*n+i] = sol
            
            if time.time() > deadline:
                break
            
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            if verify(U_r, V_r, W_r):
                best_rank = r
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also rank >= max(n, m, p) = 6
    # Try a range of ranks
    for r in range(30, best_rank + 1):
        if time.time() > deadline:
            break
        try_rank(r, max_iters=100, n_restarts=3)
        if best_rank <= r:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
