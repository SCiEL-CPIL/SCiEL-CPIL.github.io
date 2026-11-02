"""Rewritten program for 3x3x3 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 3]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_nm = n * m  # 9
    rank_mp = m * p  # 9
    rank_pn = p * n  # 9
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition (Rank 27)
    U_sb = np.zeros((rank_nm, 27), dtype=np.float64)
    V_sb = np.zeros((rank_mp, 27), dtype=np.float64)
    W_sb = np.zeros((rank_pn, 27), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = 27
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify decomposition with random matrices."""
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # C_comp[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(U_mat.shape[1]):
                            C_comp[i, k] += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
            
            if np.max(np.abs(C_comp - C_true)) > tol:
                return False
        return True

    def optimize_rank(rank_target, max_iters=2000, lr=0.05, seed=0):
        """
        Optimize for a specific rank using vectorized Adam.
        Returns (U, V, W) if successful, else None.
        """
        if rank_target >= best_rank:
            return None
            
        np.random.seed(seed)
        U = np.random.randn(rank_nm, rank_target) * 0.1
        V = np.random.randn(rank_mp, rank_target) * 0.1
        W = np.random.random((rank_pn, rank_target)) * 0.1 # Smaller init for W
        
        # Pre-compute index maps for vectorization? 
        # It's easier to use einsum or matrix multiplication tricks.
        
        # For speed, we will use a simple gradient step with Adam.
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        # Use a fixed set of test matrices for speed during optimization
        np.random.seed(123)
        A_tests = np.random.randn(5, n, m)
        B_tests = np.random.randn(5, m, p)
        C_tests = A_tests @ B_tests
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            t += 1
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for idx in range(5):
                A = A_tests[idx]
                B = B_tests[idx]
                C_true = C_tests[idx]
                
                # Compute C_comp
                # C_comp[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
                C_comp = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            # Vectorize over r
                            u_vec = U[i*m+j, :]
                            v_vec = V[j*p+k, :]
                            w_vec = W[k*n+i, :]
                            C_comp[i, k] = np.dot(u_vec * v_vec, w_vec)
                
                err = C_comp - C_true
                loss += np.sum(err**2)
                
                # Gradients
                # dW[k*n+i, r] += 2 * err[i,k] * sum_j U[i*m+j,r] V[j*p+k,r]
                # dU[i*m+j, r] += 2 * sum_k err[i,k] V[j*p+k,r] W[k*n+i,r]
                # dV[j*p+k, r] += 2 * sum_i err[i,k] U[i*m+j,r] W[k*n+i,r]
                
                for i in range(n):
                    for k in range(p):
                        e = err[i, k]
                        if e == 0: continue
                        for j in range(m):
                            u_val = U[i*m+j, :]
                            v_val = V[j*p+k, :]
                            w_val = W[k*n+i, :]
                            
                            # dW
                            dW[k*n+i, :] += 2 * e * (u_val * v_val)
                            
                            # dU (accumulate over k)
                            dU[i*m+j, :] += 2 * e * (v_val * w_val)
                            
                            # dV (accumulate over i)
                            dV[j*p+k, :] += 2 * e * (u_val * w_val)
            
            # Adam Update
            m_U[:] = beta1 * m_U + (1 - beta1) * dU
            v_U[:] = beta2 * v_U + (1 - beta2) * (dU**2)
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U[:] -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V[:] = beta1 * m_V + (1 - beta1) * dV
            v_V[:] = beta2 * v_V + (1 - beta2) * (dV**2)
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V[:] -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W[:] = beta1 * m_W + (1 - beta1) * dW
            v_W[:] = beta2 * v_W + (1 - beta2) * (dW**2)
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W[:] -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check convergence
            if loss < 1e-12:
                if verify_decomposition(U, V, W, tol=1e-8):
                    return U, V, W
        
        return None

    # Try to find lower ranks
    # Start from 26 down to 23
    for rank_target in [26, 25, 24, 23]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            res = optimize_rank(rank_target, max_iters=500, lr=0.05, seed=seed*100+1)
            if res is not None:
                U, V, W = res
                best_rank = rank_target
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            # Try to go lower immediately if we have time
            continue
        else:
            # If we failed at this rank, it's unlikely we'll succeed at lower ranks with this method
            # But we might have just bad luck. Try one more seed for the next lower rank?
            # For now, break if we fail at 26, as 25/24/23 are harder.
            # Actually, let's try the next rank with fewer seeds.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
