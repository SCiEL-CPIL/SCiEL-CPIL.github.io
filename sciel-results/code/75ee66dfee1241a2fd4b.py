import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Note: The problem statement says W[k*n+i, r].
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Construct target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
                
    T_flat = T.flatten()
    
    best_rank = n*m*p
    best_decomp = None
    
    # Try ranks from lower bound to schoolbook
    # Lower bound heuristic: max(n*m, m*p, p*n) is 9.
    # Known rank for 2x3x3 is likely 16.
    # Let's try 14, 15, 16, 17.
    
    for r in range(14, n*m*p + 1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            # Use small random values to avoid large numbers initially
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U: T_flat = sum_c V_c (x) W_c (x) U_c
                # This is a linear system for U.
                # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
                # For fixed V, W, U_{ic} are unknowns.
                # We can reshape T to (n*m, m*p*p*n) ?
                # T_{i, (j,k)} = sum_c U_{ic} (V_{jc} W_{kc})
                # Let M_U be a matrix of size (m*p*p*n, r) where M_U[(j,k), c] = V_{jc} W_{kc}
                # Then T_{i, (j,k)} = sum_c U_{ic} M_U[(j,k), c]
                # So T_reshaped (n*m, m*p*p*n) = U (n*m, r) @ M_U.T (r, m*p*p*n)
                
                # Construct M_U
                # V shape (m*p, r), W shape (p*n, r)
                # We need outer product of V and W for each c?
                # M_U[(j,k), c] = V[j, c] * W[k, c]
                # V_flat = V.flatten() (m*p*r)
                # W_flat = W.flatten() (p*n*r)
                # This is a bit slow to construct explicitly each time.
                # Better: T_reshaped = U @ (V[:, :, None] * W[None, :, :]).reshape(m*p, p*n, r).transpose(2, 0, 1).reshape(r, m*p*p*n).T
                # Actually, let's just use the tensor contraction.
                
                # Reshape T to (n*m, -1)
                T_reshaped = T.reshape(n*m, -1)
                
                # Construct matrix A_U such that T_reshaped = U @ A_U.T
                # A_U.T has shape (r, m*p*p*n)
                # A_U.T[c, j*p*n + k] = V[j, c] * W[k, c]
                
                # Vectorized construction of A_U
                # V[:, :, None] is (m*p, r, 1)
                # W[None, :, :] is (1, p*n, r)
                # Product is (m*p, p*n, r)
                # We want to reshape this to (r, m*p*p*n)
                # prod = V[:, :, None] * W[None, :, :]  # (m*p, p*n, r)
                # A_U_T = prod.transpose(2, 0, 1).reshape(r, m*p*p*n)
                
                prod_U = V[:, :, None] * W[None, :, :]
                A_U_T = prod_U.transpose(2, 0, 1).reshape(r, -1)
                
                # Solve U = T_reshaped @ A_U_T.T @ inv(A_U_T @ A_U_T.T)
                # Or use lstsq
                try:
                    U, _, _, _ = np.linalg.lstsq(A_U_T, T_reshaped.T, rcond=None)
                    U = U.T
                except:
                    break
                    
                # Solve for V
                # T_{ijk} = sum_c V_{jc} U_{ic} W_{kc}
                # T_{j, (i,k)} = sum_c V_{jc} (U_{ic} W_{kc})
                # A_V.T[c, i*p*n + k] = U[i, c] * W[k, c]
                prod_V = U[:, :, None] * W[None, :, :]
                A_V_T = prod_V.transpose(2, 0, 1).reshape(r, -1)
                T_reshaped_V = T.reshape(m*p, -1)
                try:
                    V, _, _, _ = np.linalg.lstsq(A_V_T, T_reshaped_V.T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Solve for W
                # T_{ijk} = sum_c W_{kc} U_{ic} V_{jc}
                # T_{k, (i,j)} = sum_c W_{kc} (U_{ic} V_{jc})
                # A_W.T[c, i*m*p + j] = U[i, c] * V[j, c]
                prod_W = U[:, :, None] * V[None, :, :]
                A_W_T = prod_W.transpose(2, 0, 1).reshape(r, -1)
                T_reshaped_W = T.reshape(p*n, -1)
                try:
                    W, _, _, _ = np.linalg.lstsq(A_W_T, T_reshaped_W.T, rcond=None)
                    W = W.T
                except:
                    break
                
                # Check convergence / error
                # Reconstruct T_approx
                # T_approx = sum_c U_c (x) V_c (x) W_c
                # Vectorized:
                # T_approx_flat = np.einsum('ic,jc,kc->ijk', U, V, W).flatten()
                # err = np.linalg.norm(T_flat - T_approx_flat)
                
                # Faster check:
                # If we just solved for U, we can check the residual of the last step
                # But let's do a full check every few steps or at the end.
                if it % 10 == 0 or it == 99:
                    T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
                    err = np.linalg.norm(T_flat - T_approx.flatten())
                    if err < 1e-6:
                        converged = True
                        break
                        # Try to snap to grid
                        # Round to nearest 0.5
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        T_snap = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
                        if np.allclose(T_flat, T_snap.flatten(), atol=1e-9):
                            # Success!
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            if converged:
                # If converged but didn't snap, maybe it's not exact.
                # Try snapping again with tighter tolerance or different rounding
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                T_snap = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_flat, T_snap.flatten(), atol=1e-9):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
                # Try rounding to integers
                U_snap = np.round(U)
                V_snap = np.round(V)
                W_snap = np.round(W)
                T_snap = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_flat, T_snap.flatten(), atol=1e-9):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    # If no better rank found, return schoolbook
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
