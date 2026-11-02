import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook solution as fallback
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = get_schoolbook()
    
    # Try to find lower rank solutions
    # Lower bound estimation: maybe start from 24?
    # We know 32 is possible.
    # Let's try ranks in descending order of likelihood or ascending?
    # Ascending allows finding the minimum if we check all.
    # But if 24 is the min, we should find it.
    
    # We'll try ranks from 20 to 31
    for r in range(20, 32):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this rank
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(100):
                # Solve for W
                # Eq: sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
                # Let's build the linear system for W
                # Variables: W_flat of size (p*n*r)
                # Equations: for each (i, j, k), sum_c ... = 1
                
                # We can vectorize this.
                # Let's create a matrix A_W of shape (n*m*p, p*n*r)
                # But building it every time is slow.
                # We can update it incrementally or just build it.
                # n*m*p = 32. p*n*r = 8*r.
                # For r=24, 192. 32x192 matrix. Very fast to build.
                
                A_W = np.zeros((n * m * p, p * n * r))
                b_W = np.ones(n * m * p)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                # W index: k*n + i, c -> flat: (k*n+i)*r + c
                                w_idx = (k * n + i) * r + c
                                # Coeff: U[i*m+j, c] * V[j*p+k, c]
                                A_W[idx, w_idx] = U[i * m + j, c] * V[j * p + k, c]
                            idx += 1
                
                # Solve A_W @ W_flat = b_W
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(p * n, r)
                
                # Check error
                err = np.linalg.norm(A_W @ W_flat - b_W)
                if err < 1e-5:
                    # Check if we can snap
                    # We should also update U and V to see if error stays low
                    pass
                
                # Solve for V
                # Eq: sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1
                A_V = np.zeros((n * m * p, m * p * r))
                b_V = np.ones(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                v_idx = (j * p + k) * r + c
                                A_V[idx, v_idx] = U[i * m + j, c] * W[k * n + i, c]
                            idx += 1
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_flat.reshape(m * p, r)
                
                # Solve for U
                # Eq: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                A_U = np.zeros((n * m * p, n * m * r))
                b_U = np.ones(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                u_idx = (i * m + j) * r + c
                                A_U[idx, u_idx] = V[j * p + k, c] * W[k * n + i, c]
                            idx += 1
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(n * m, r)
                
                # Check final error
                # Rebuild A_W with new U, V
                # Or just compute the tensor contraction
                # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                
                err = np.linalg.norm(T - 1.0)
                if err < 1e-6:
                    # Try to snap to integers/half-integers
                    # Round to nearest 0.5
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    
                    # Verify snapped solution
                    T_snapped = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_snapped[i, j, k] = np.sum(U_snapped[i * m + j, :] * V_snapped[j * p + k, :] * W_snapped[k * n + i, :])
                    
                    if np.allclose(T_snapped, 1.0, atol=1e-9):
                        # Found a valid solution
                        return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
            
            # If not converged in this restart, try next
            pass

    return best_result
