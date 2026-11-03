import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Target tensor T[i, j, k] = 1
    # Decomposition: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] == 1
    
    # We will try ranks from lower bound up to 40
    # Lower bound is max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    for R in range(20, 41):
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts for this rank
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W with random small integers
            # U: (n*m, R)
            # V: (m*p, R)
            # W: (p*n, R)
            
            U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
            
            # Avoid all zeros
            U[0, 0] = 1
            V[0, 0] = 1
            W[0, 0] = 1
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                    
                # Solve for U
                # Eq: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed i, j:
                # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r]) = 1 ?
                # NO. The sum is over r. The k is an index.
                # The equation is for each (i, j, k):
                # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                # This is a linear system in U[i*m+j, :] for each (i, j, k)?
                # No, U[i*m+j, r] appears in multiple equations (for different k).
                # Let's group by (i, j).
                # For a fixed (i, j), we have variables U[i*m+j, 0..R-1].
                # The equations involving these variables are for all k.
                # For each k, we have:
                # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let M_k[r] = V[j*p+k, r] * W[k*n+i, r]
                # Then sum_r U[i*m+j, r] * M_k[r] = 1 for all k.
                # This is a system of p equations (for k=0..p-1) with R unknowns.
                # We can solve this in least squares sense.
                
                # We do this for all (i, j).
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        idx_u = i*m + j
                        # Build matrix A (p x R) and vector b (p)
                        A = np.zeros((p, R))
                        b = np.ones(p)
                        for k in range(p):
                            idx_v = j*p + k
                            idx_w = k*n + i
                            A[k, :] = V[idx_v, :] * W[idx_w, :]
                        # Solve A @ U[idx_u, :] = b
                        # Use least squares
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_U[idx_u, :] = sol
                U = new_U
                
                # Solve for V
                # Similar, group by (j, k)
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        idx_v = j*p + k
                        A = np.zeros((n, R))
                        b = np.ones(n)
                        for i in range(n):
                            idx_u = i*m + j
                            idx_w = k*n + i
                            A[i, :] = U[idx_u, :] * W[idx_w, :]
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_V[idx_v, :] = sol
                V = new_V
                
                # Solve for W
                # Group by (k, i)
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        idx_w = k*n + i
                        A = np.zeros((m, R))
                        b = np.ones(m)
                        for j in range(m):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            A[j, :] = U[idx_u, :] * V[idx_v, :]
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_W[idx_w, :] = sol
                W = new_W
                
                # Check residual
                # Compute T_decomp[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # We can do this efficiently
                # T_decomp is n x m x p
                T_decomp = np.zeros((n, m, p))
                for r in range(R):
                    # u_r is n x m
                    u_r = U[:, r].reshape(n, m)
                    # v_r is m x p
                    v_r = V[:, r].reshape(m, p)
                    # w_r is p x n
                    w_r = W[:, r].reshape(p, n)
                    # Contribution to T[i, j, k] is u_r[i, j] * v_r[j, k] * w_r[k, i]
                    # This is not a simple outer product.
                    # We can iterate
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_decomp[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                
                residual = np.max(np.abs(T_decomp - 1))
                if residual < 1e-9:
                    # Found a solution!
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounded solution is exact
                    T_rounded = np.zeros((n, m, p))
                    for r in range(R):
                        u_r = U_rounded[:, r].reshape(n, m)
                        v_r = V_rounded[:, r].reshape(m, p)
                        w_r = W_rounded[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_rounded[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                    
                    if np.max(np.abs(T_rounded - 1)) < 1e-9:
                        # Convert to lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Return the floating point solution? 
                        # The task requires integer/half-integer entries.
                        # If rounding fails, we might need to continue searching.
                        pass

    # Fallback to schoolbook if no better found
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
