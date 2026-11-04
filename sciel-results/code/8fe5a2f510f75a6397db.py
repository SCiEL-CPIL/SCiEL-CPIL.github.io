"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] -> actually T[i,j,k] for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Target tensor as 3D array
    T = np.ones((n, m, p), dtype=float)
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T (within tol)."""
        r = U.shape[1]
        # Compute sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape and use einsum
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.max(np.abs(result - T)) < tol
    
    def snap_to_half_int(A):
        """Snap each entry to nearest half-integer."""
        return np.round(2 * A) / 2.0
    
    def try_rank(r, n_restarts=50, max_iter=5000):
        """Try to find a rank-r decomposition using random initialization and optimization."""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Use coordinate descent / simple gradient steps
            # Objective: ||sum_c U_c ⊗ V_c ⊗ W_c - T||_F^2
            best_err = np.inf
            best_U, best_V, best_W = None, None, None
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                current = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.sum((current - T) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                # Gradient descent step
                lr = 0.01 / (1 + it * 0.001)
                
                # Gradient w.r.t. U: for each (i,j), grad_U[i,j] = sum_{k,c} (current[i,j,k]-T[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
                # Simplify: fix V, W, update U
                for c in range(r):
                    # d/dU[i*m+j,c] = sum_k (current[i,j,k]-T[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
                    # Vectorize over i,j
                    # For fixed c, the contribution to U[:,c]
                    # grad_U[i*m+j, c] = sum_k (current[i,j,k]-T[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
                    
                    # Build the gradient for this column
                    grad_U_col = np.zeros(n * m)
                    for i in range(n):
                        for j in range(m):
                            s = 0.0
                            for k in range(p):
                                s += (current[i, j, k] - T[i, j, k]) * V[j * p + k, c] * W[k * n + i, c]
                            grad_U_col[i * m + j] = s
                    U[:, c] -= lr * grad_U_col
                    
                    # Similarly for V
                    grad_V_col = np.zeros(m * p)
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for i in range(n):
                                s += (current[i, j, k] - T[i, j, k]) * U[i * m + j, c] * W[k * n + i, c]
                            grad_V_col[j * p + k] = s
                    V[:, c] -= lr * grad_V_col
                    
                    # Similarly for W
                    grad_W_col = np.zeros(p * n)
                    for k in range(p):
                        for i in range(n):
                            s = 0.0
                            for j in range(m):
                                s += (current[i, j, k] - T[i, j, k]) * U[i * m + j, c] * V[j * p + k, c]
                            grad_W_col[k * n + i] = s
                    W[:, c] -= lr * grad_W_col
                
                # Recompute for next iteration
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                current = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            # Snap to half-integers and verify
            U_snap = snap_to_half_int(best_U)
            V_snap = snap_to_half_int(best_V)
            W_snap = snap_to_half_int(best_W)
            
            if verify(U_snap, V_snap, W_snap):
                return (U_snap, V_snap, W_snap)
            
            # Also try without snapping in case it's already exact
            if verify(best_U, best_V, best_W):
                return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from 19 upward
    for r in range(19, 25):
        if time.time() > deadline:
            break
        result = try_rank(r, n_restarts=30, max_iter=3000)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
