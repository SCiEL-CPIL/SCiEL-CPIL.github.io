import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=3i+j, v=3j+k, w=3k+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # We want to find U, V, W such that sum_r U[:,r] outer V[:,r] outer W[:,r] == T
    # This is equivalent to minimizing || T - sum_r u_r v_r^T w_r ||^2
    
    # Try ranks from 23 up to 27
    # Rank 27 is always possible (schoolbook)
    
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep 30s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook initialization for fallback
    U_fb = np.zeros((n*m, best_rank))
    V_fb = np.zeros((m*p, best_rank))
    W_fb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank
    for R in range(23, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few random restarts
        max_restarts = 5 if R < 27 else 1
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Use a simple gradient descent or least squares approach
            # Flatten U, V, W? Or update one matrix at a time?
            # Alternating Least Squares (ALS) is better for tensor decomposition.
            # But we have a fixed target T, not a data tensor.
            # We want T ≈ sum u_r v_r w_r.
            # This is a linear system in U if V, W are fixed.
            # T_{uvw} = sum_r U_{ur} V_{vr} W_{wr}
            # Let's fix V and W, solve for U.
            # For each u, T_{u,:,:} = sum_r U_{ur} (V_{:,r} outer W_{:,r})
            # This is a linear system: A U_u = T_u
            # Where A is (m*p*p*n) x R? No.
            
            # Let's use a simple iterative refinement:
            # 1. Fix V, W. Solve for U.
            # 2. Fix U, W. Solve for V.
            # 3. Fix U, V. Solve for W.
            
            # Construct the linear system for U:
            # T_{u, v, w} = sum_r U_{u, r} V_{v, r} W_{w, r}
            # Let M_r = V_{:, r} outer W_{:, r} (shape m*p x p*n)
            # T_{u, :, :} = sum_r U_{u, r} M_r
            # This is a linear system for U_u (vector of length R).
            # [M_0, M_1, ..., M_{R-1}] U_u = T_u
            # The matrix is (m*p*p*n) x R.
            # m*p = 9, p*n = 9. Size 81 x R.
            
            def solve_linear(X, Y, Z, T_target, X_name):
                # Solve for X given Y, Z
                # T_target is (n*m, m*p, p*n)
                # We solve for X (n*m, R)
                # Reshape T_target to (n*m, m*p*p*n)
                T_flat = T_target.reshape(n*m, m*p*p*n)
                
                # Construct A: (m*p*p*n, R)
                # A[:, r] = Y[:, r] outer Z[:, r] flattened
                A = np.zeros((m*p*p*n, R))
                for r in range(R):
                    M_r = np.outer(Y[:, r], Z[:, r])
                    A[:, r] = M_r.flatten()
                
                # Solve A X_u = T_flat_u for each u
                # Use least squares
                # X = np.linalg.lstsq(A, T_flat.T, rcond=None)[0].T
                # This might be slow if done in a loop, but R is small.
                try:
                    # T_flat is (9, 81). A is (81, R).
                    # We want X (9, R) such that X @ A.T ≈ T_flat.T (R, 9)?
                    # No. T_flat[u, :] = sum_r X[u, r] A[:, r].T ?
                    # T_flat[u, v] = sum_r X[u, r] A[v, r]
                    # So T_flat = X @ A.T
                    # X = T_flat @ A.T.pinv() ?
                    # Or use lstsq.
                    # A.T is (R, 81). T_flat.T is (81, 9).
                    # We want X (9, R).
                    # T_flat.T = A.T @ X.T ?
                    # (81, R) @ (R, 9) = (81, 9). Yes.
                    # So X.T = lstsq(A.T, T_flat.T)
                    X_T, residuals, rank, s = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    return X_T.T
                except np.linalg.LinAlgError:
                    return None

            # Iteration
            error = np.inf
            for it in range(50):
                # Solve for U
                U_new = solve_linear(U, V, W, T, 'U')
                if U_new is None: break
                # Solve for V
                # T_{u, v, w} = sum_r U_{u, r} V_{v, r} W_{w, r}
                # Fix U, W. Solve for V.
                # T_{:, v, :} = sum_r V_{v, r} (U_{:, r} outer W_{:, r})
                # M_r = U_{:, r} outer W_{:, r} (shape n*m x p*n)
                # A_V[:, r] = M_r.flatten() (shape n*m*p*n x R)
                # T_flat_V = T.reshape(m*p, n*m*p*n) ?
                # T is (n*m, m*p, p*n). We want to group by v (index 1).
                # T_v = T[:, v, :] (shape n*m x p*n)
                # T_v = sum_r V[v, r] M_r
                # So for each v, T_v = V_v @ A_V.T
                # V_v = T_v @ A_V.T.pinv()
                
                A_V = np.zeros((n*m*p*n, R))
                for r in range(R):
                    M_r = np.outer(U[:, r], W[:, r])
                    A_V[:, r] = M_r.flatten()
                
                V_new = np.zeros((m*p, R))
                for v in range(m*p):
                    T_v = T[:, v, :].flatten() # shape n*m*p*n
                    # T_v = sum_r V[v, r] A_V[:, r]
                    # T_v = V_v @ A_V.T
                    # V_v.T = lstsq(A_V.T, T_v.T)
                    V_v_T, _, _, _ = np.linalg.lstsq(A_V.T, T_v.T, rcond=None)
                    V_new[v, :] = V_v_T.T
                    
                # Solve for W
                A_W = np.zeros((n*m*m*p, R))
                for r in range(R):
                    M_r = np.outer(U[:, r], V[:, r])
                    A_W[:, r] = M_r.flatten()
                    
                W_new = np.zeros((p*n, R))
                for w in range(p*n):
                    T_w = T[:, :, w].flatten() # shape n*m*m*p
                    # T_w = W_w @ A_W.T
                    W_w_T, _, _, _ = np.linalg.lstsq(A_W.T, T_w.T, rcond=None)
                    W_new[w, :] = W_w_T.T
                    
                U, V, W = U_new, V_new, W_new
                
                # Calculate error
                # T_recon = sum_r outer(outer(U[:,r], V[:,r]), W[:,r])
                # This is expensive.
                # error = np.linalg.norm(T - np.tensordot(np.tensordot(U, V, axes=([1],[1])), W, axes=([1],[1])))
                # Let's compute it.
                T_recon = np.zeros_like(T)
                for r in range(R):
                    T_recon += np.tensordot(np.tensordot(U[:, r], V[:, r], axes=0), W[:, r], axes=0)
                error = np.linalg.norm(T - T_recon)
                
                if error < 1e-6:
                    found = True
                    break
                
                if time.time() > deadline - 10:
                    break
                    
            if found:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
                
        if found:
            break
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
