import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Build the target tensor T flattened
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We represent T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Trivial solution as fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, idx] = 1.0
                V_triv[j * p + k, idx] = 1.0
                W_triv[k * n + i, idx] = 1.0
                idx += 1

    deadline = time.time() + 110  # Safety margin
    
    # Try to find a lower rank decomposition
    # We will try ranks from 35 down to 30
    # For each rank, we run a few iterations of ALS
    
    for r in range(35, 25, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Initialize with small integers
            U = np.random.randint(-1, 2, (n * m, r)).astype(float)
            V = np.random.randint(-1, 2, (m * p, r)).astype(float)
            W = np.random.randint(-1, 2, (p * n, r)).astype(float)
            
            # ALS Iterations
            for it in range(20):
                if time.time() > deadline:
                    break
                    
                # Update U:
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed b, c, T[a, b, c] = sum_r U[a, r] (V[b, r] W[c, r])
                # Let M_U[b, c, r] = V[b, r] * W[c, r]
                # Then T[:, b, c] = U @ M_U[b, c, :]
                # We can solve for U by flattening.
                # T_flat (n*m, m*p*p*n) = U (n*m, r) @ M_U_flat (r, m*p*p*n)
                # This is a large system.
                
                # Efficient ALS step for U:
                # We want to minimize || T - sum_r U[:,r] outer(V[:,r], W[:,r]) ||_F
                # This is equivalent to solving for each column r? No, U is coupled.
                # Actually, we can solve for U given V, W by forming the design matrix.
                
                # Let's reshape T to (n*m, m*p * p*n)
                T_flat = T.reshape(n * m, m * p * p * n)
                
                # Construct design matrix D_U of shape (r, m*p * p*n)
                # D_U[r, b_idx * p*n + c_idx] = V[b_idx, r] * W[c_idx, r]
                D_U = np.zeros((r, m * p * p * n))
                for b in range(m * p):
                    for c in range(p * n):
                        D_U[:, b * p * n + c] = V[b, :] * W[c, :]
                
                # Solve U @ D_U.T = T_flat.T ?
                # T_flat = U @ D_U.T
                # [T_flat.T] = D_U @ U.T
                # U.T = pinv(D_U) @ T_flat.T
                try:
                    U_new = np.linalg.lstsq(D_U, T_flat.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue
                    
                # Update V
                # T[a, b, c] = sum_r V[b, r] (U[a, r] W[c, r])
                T_flat_V = T.reshape(m * p, n * m * p * n)
                D_V = np.zeros((r, n * m * p * n))
                for a in range(n * m):
                    for c in range(p * n):
                        D_V[:, a * p * n + c] = U[a, :] * W[c, :]
                try:
                    V_new = np.linalg.lstsq(D_V, T_flat_V.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue

                # Update W
                T_flat_W = T.reshape(p * n, n * m * m * p)
                D_W = np.zeros((r, n * m * m * p))
                for a in range(n * m):
                    for b in range(m * p):
                        D_W[:, a * m * p + b] = U[a, :] * V[b, :]
                try:
                    W_new = np.linalg.lstsq(D_W, T_flat_W.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue

                U, V, W = U_new, V_new, W_new
                
                # Check convergence and validity
                # Compute residual
                # T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
                # Residual norm
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(T - T_approx)
                
                if residual < 1e-8:
                    # Found a solution! Now round to half-integers
                    # The solution from lstsq is real.
                    # We need to check if it's close to half-integers.
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    residual_rounded = np.linalg.norm(T - T_approx_rounded)
                    
                    if residual_rounded < 1e-8:
                        # Valid integer/half-integer solution found
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    else:
                        # Not exact after rounding, but close. 
                        # We could try to repair, but for now, if we can't get exact, 
                        # we might have to continue searching or fallback.
                        # However, lstsq gives the minimum norm solution.
                        # If the true solution is half-integer, the real solution should be very close.
                        # If rounding fails, the rank might not be achievable with half-integers 
                        # or we are in a local minimum.
                        pass

    # Fallback to trivial
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }
