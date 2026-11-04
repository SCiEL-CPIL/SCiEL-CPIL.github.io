import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank
    
    # Search for lower ranks
    # Lower bound is max(nm, mp, pn) = 15
    # Try from 15 up to schoolbook_rank
    for target_rank in range(15, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 50 if target_rank < 20 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Alternating Least Squares (ALS)
            # We want to minimize || T - sum_r U[:,r] V[:,r] W[:,r] ||_F^2
            # This is equivalent to minimizing || reshape(T) - M ||_F^2 where M is the matricization?
            # No, it's a tensor.
            # Let's use the property:
            # Fix V, W. Update U.
            # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Let M_r = outer(V[:, r], W[:, r]) of shape (m*p, p*n)
            # Then T_approx[a] = sum_r U[a, r] * M_r (flattened)
            # This is a linear least squares problem for each row a of U.
            # Specifically, let T_mat be shape (n*m, m*p * p*n) where T_mat[a, idx] = T[a, b, c]
            # Let A be shape (target_rank, m*p * p*n) where A[r, idx] = V[b, r] * W[c, r]
            # Then we want U @ A^T approx T_mat.
            # U is (n*m, target_rank). A is (target_rank, m*p * p*n).
            # T_mat = U @ A^T + E
            # So U = T_mat @ (A^T @ A)^{-1} A^T ? No, U is on the left.
            # T_mat (n*m, D) = U (n*m, r) @ A^T (r, D)
            # So U = T_mat @ pinv(A^T)
            
            D = m * p * p * n
            # Precompute indices for flattening T
            # T is (n*m, m*p, p*n)
            # T_mat[a, b, c] -> T_mat[a, b * (p*n) + c]
            
            for iteration in range(100):
                # Update U
                # Construct A_U of shape (target_rank, D)
                # A_U[r, b * (p*n) + c] = V[b, r] * W[c, r]
                # V is (m*p, r), W is (p*n, r)
                # We can compute this via einsum or broadcasting
                # V_flat: (m*p, r) -> (1, m*p, r)
                # W_flat: (p*n, r) -> (p*n, 1, r)
                # Product: (m*p, p*n, r)
                # Reshape to (m*p*p*n, r)
                # Transpose to (r, m*p*p*n)
                
                # Efficiently:
                # V is (mp, r), W is (pn, r)
                # We need A_U[r, b, c] = V[b, r] * W[c, r]
                # A_U is (r, mp, pn)
                # A_U = V.T[:, :, None] * W.T[:, None, :] -> (r, mp, pn)
                A_U = V.T[:, :, None] * W.T[:, None, :] # shape (r, mp, pn)
                A_U_flat = A_U.reshape(target_rank, D)
                
                # T_mat is (n*m, D)
                T_mat = T.reshape(n*m, D)
                
                # Solve U @ A_U_flat.T = T_mat
                # U = T_mat @ pinv(A_U_flat.T)
                # A_U_flat.T is (D, r)
                # pinv(A_U_flat.T) is (r, D)
                # U = T_mat @ (A_U_flat.T) @ pinv(A_U_flat.T @ A_U_flat.T)?
                # Least squares: min || T_mat - U @ A_U_flat.T ||
                # U = T_mat @ (A_U_flat.T) @ inv(A_U_flat.T @ A_U_flat.T)
                # Or just use lstsq
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_U_flat.T, T_mat.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V
                # Fix U, W. Update V.
                # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed b, we want to find V[b, :]
                # Let's reshape T to (m*p, n*m * p*n)
                # T_mat_V[b, a * (p*n) + c] = T[a, b, c]
                # A_V[r, a * (p*n) + c] = U[a, r] * W[c, r]
                
                # A_V is (r, nm, pn)
                A_V = U.T[:, :, None] * W.T[:, None, :] # shape (r, nm, pn)
                A_V_flat = A_V.reshape(target_rank, D)
                
                T_mat_V = np.transpose(T, (1, 0, 2)).reshape(m*p, D)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_V_flat.T, T_mat_V.T, rcond=None)
                    V = V_new.T
                except:
                    break
                    
                # Update W
                # Fix U, V. Update W.
                # A_W[r, a * (m*p) + b] = U[a, r] * V[b, r]
                # T_mat_W[c, a * (m*p) + b] = T[a, b, c]
                
                A_W = U.T[:, :, None] * V.T[:, None, :] # shape (r, nm, mp)
                A_W_flat = A_W.reshape(target_rank, D)
                
                T_mat_W = np.transpose(T, (2, 0, 1)).reshape(p*n, D)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_W_flat.T, T_mat_W.T, rcond=None)
                    W = W_new.T
                except:
                    break
                    
                # Check convergence / exactness
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T - T_approx)
                if error < 1e-8:
                    # Try rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    T_approx_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                    if np.allclose(T, T_approx_int, atol=1e-10):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    T_approx_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                    if np.allclose(T, T_approx_half, atol=1e-10):
                        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                        
                    # If close but not exact, maybe we can refine?
                    # For now, break to try next restart or rank
                    break

    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
