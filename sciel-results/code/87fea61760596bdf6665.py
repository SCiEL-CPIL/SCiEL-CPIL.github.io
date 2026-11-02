import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Create the target tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, target_rank))
    V_sb = np.zeros((m*p, target_rank))
    W_sb = np.zeros((p*n, target_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
                
    best_rank = target_rank
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = 120.0
    
    # Try to find lower rank
    # We will try ranks from target_rank - 1 down to target_rank - 10
    for R in range(target_rank - 1, max(target_rank - 10, 0), -1):
        if time.time() - start_time > deadline:
            break
            
        found = False
        # Multiple restarts for this R
        for restart in range(10):
            if time.time() - start_time > deadline:
                break
                
            # Initialize U, V, W
            # Strategy: Take R random schoolbook terms
            indices = np.random.choice(target_rank, R, replace=False)
            U_curr = U_sb[:, indices].copy()
            V_curr = V_sb[:, indices].copy()
            W_curr = W_sb[:, indices].copy()
            
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > deadline:
                    break
                    
                # Solve for W
                A_W = np.zeros((n*m*p, p*n*R))
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_t = i * m * p + j * p + k
                                idx_w = (k * n + i) * R + r
                                A_W[idx_t, idx_w] = U_curr[i*m + j, r] * V_curr[j*p + k, r]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
                W_curr = W_flat.reshape((p*n, R))
                
                # Solve for V
                A_V = np.zeros((n*m*p, m*p*R))
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_t = i * m * p + j * p + k
                                idx_v = (j * p + k) * R + r
                                A_V[idx_t, idx_v] = U_curr[i*m + j, r] * W_curr[k*n + i, r]
                                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
                V_curr = V_flat.reshape((m*p, R))
                
                # Solve for U
                A_U = np.zeros((n*m*p, n*m*R))
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_t = i * m * p + j * p + k
                                idx_u = (i * m + j) * R + r
                                A_U[idx_t, idx_u] = V_curr[j*p + k, r] * W_curr[k*n + i, r]
                                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
                U_curr = U_flat.reshape((n*m, R))
                
                # Check error
                T_recon = np.zeros((n, m, p))
                for r in range(R):
                    T_recon += np.einsum('i,j,k->ijk', U_curr[:, r], V_curr[:, r], W_curr[:, r])
                error = np.linalg.norm(T_recon.flatten() - T_flat)
                
                if error < 1e-6:
                    # Found a solution!
                    # Snap to integers/half-integers
                    U_snap = np.round(U_curr * 2) / 2
                    V_snap = np.round(V_curr * 2) / 2
                    W_snap = np.round(W_curr * 2) / 2
                    
                    # Verify exactness with snapped values
                    T_recon_snap = np.zeros((n, m, p))
                    for r in range(R):
                        T_recon_snap += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                    if np.allclose(T_recon_snap.flatten(), T_flat, atol=1e-9):
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        found = True
                        break
                        
        if found:
            break
            
    if best_rank < target_rank and best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
