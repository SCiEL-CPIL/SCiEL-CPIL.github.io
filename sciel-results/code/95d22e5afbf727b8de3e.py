import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_solution = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Precompute indices for fast tensor access
    # We need to evaluate S[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def evaluate_error(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over c
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = val - 1.0
                    error += diff * diff
        return error

    def snap_and_verify(U, V, W):
        # Snap to half-integers
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check error
        err = evaluate_error(U_s, V_s, W_s)
        if err < 1e-6:
            return U_s, V_s, W_s, True
        return None, None, None, False

    def optimize_rank(r, num_restarts=10):
        if time.time() > deadline:
            return None, None, None, False
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    return None, None, None, False
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # We can vectorize the gradient computation
                # S[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # dE/dU[i*m+j, c] = 2 * (S[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
                
                # Iterate over i, j, k
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # S_val = dot(U[u_idx], V[v_idx] * W[w_idx])
                            S_val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            diff = 2 * (S_val - 1.0)
                            
                            grad_U[u_idx, :] += diff * V[v_idx, :] * W[w_idx, :]
                            grad_V[v_idx, :] += diff * U[u_idx, :] * W[w_idx, :]
                            grad_W[w_idx, :] += diff * U[u_idx, :] * V[v_idx, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if step % 50 == 0:
                    err = evaluate_error(U, V, W)
                    if err < 1e-8:
                        U_s, V_s, W_s, valid = snap_and_verify(U, V, W)
                        if valid:
                            return U_s, V_s, W_s, True
                            
        return None, None, None, False

    # Try ranks from 41 down to 30
    for r in range(rank_sb - 1, 29, -1):
        if time.time() > deadline:
            break
            
        U_opt, V_opt, W_opt, valid = optimize_rank(r, num_restarts=5)
        if valid:
            best_solution = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            best_rank = r
            # We can break if we find a very good rank, or continue to find even lower?
            # The score is rank. Lower is better.
            # We should try to find the lowest possible.
            # But time is limited.
            # If we find 41, we should try 40.
            # So we don't break, we just update best.
            # However, if we find 35, it's a big jump.
            # We will continue searching lower ranks.

    return best_solution
