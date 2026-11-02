"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target = np.ones((n, m, p))
    
    # Time guard
    t0 = time.time()
    deadline = 135.0  # 135s out of 150s limit, safety margin
    
    # Helper functions
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def reconstruct(U, V, W, R):
        """Reconstruct tensor from factor matrices."""
        Ur = U.reshape((n, m, R))
        Vr = V.reshape((m, p, R))
        Wr = W.reshape((p, n, R))
        return np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    # Build rank-26 decomposition (Strassen + standard)
    R26 = 26
    U26 = np.zeros((n * m, R26))
    V26 = np.zeros((m * p, R26))
    W26 = np.zeros((p * n, R26))
    
    # Strassen 7 terms for 2x2 core
    r = 0
    U26[idx_u(0,0), r] = 1; U26[idx_u(1,1), r] = 1
    V26[idx_v(0,0), r] = 1; V26[idx_v(1,1), r] = 1
    W26[idx_w(0,0), r] = 1; W26[idx_w(1,1), r] = 1
    
    r = 1
    U26[idx_u(1,0), r] = 1; U26[idx_u(1,1), r] = 1
    V26[idx_v(0,0), r] = 1
    W26[idx_w(0,1), r] = 1; W26[idx_w(1,1), r] = -1
    
    r = 2
    U26[idx_u(0,0), r] = 1
    V26[idx_v(0,1), r] = 1; V26[idx_v(1,1), r] = -1
    W26[idx_w(1,0), r] = 1; W26[idx_w(1,1), r] = 1
    
    r = 3
    U26[idx_u(1,1), r] = 1
    V26[idx_v(1,0), r] = 1; V26[idx_v(0,0), r] = -1
    W26[idx_w(0,0), r] = 1; W26[idx_w(0,1), r] = 1
    
    r = 4
    U26[idx_u(0,0), r] = 1; U26[idx_u(0,1), r] = 1
    V26[idx_v(1,1), r] = 1
    W26[idx_w(0,0), r] = -1; W26[idx_w(1,0), r] = 1
    
    r = 5
    U26[idx_u(1,0), r] = 1; U26[idx_u(0,0), r] = -1
    V26[idx_v(0,0), r] = 1; V26[idx_v(0,1), r] = 1
    W26[idx_w(1,1), r] = 1
    
    r = 6
    U26[idx_u(0,1), r] = 1; U26[idx_u(1,1), r] = -1
    V26[idx_v(1,0), r] = 1; V26[idx_v(1,1), r] = 1
    W26[idx_w(0,0), r] = 1
    
    # Standard 19 terms for border
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
    
    # Verify rank-26 is correct
    T_recon_26 = reconstruct(U26, V26, W26, R26)
    if not np.allclose(T_recon_26, target, atol=1e-9):
        # Fallback: schoolbook rank 27
        R = 27
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i,j), col] = 1
                    V[idx_v(j,k), col] = 1
                    W[idx_w(k,i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_reduce_rank(R, max_time=30.0):
        """Try to find a rank-R decomposition via optimization."""
        start_time = time.time()
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Try multiple random initializations
        for seed in range(20):
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 12345 + 678)
            
            # Initialize near rank-26 but with R columns
            if R <= 26:
                U_init = U26[:, :R].copy()
                V_init = V26[:, :R].copy()
                W_init = W26[:, :R].copy()
            else:
                U_init = np.random.randn(size_u, R) * 0.5
                V_init = np.random.randn(size_v, R) * 0.5
                W_init = np.random.randn(size_w, R) * 0.5
            
            # Add perturbation
            U_init += np.random.randn(size_u, R) * 0.3
            V_init += np.random.randn(size_v, R) * 0.3
            W_init += np.random.randn(size_w, R) * 0.3
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Adam optimizer
            lr = 0.005
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_t = np.zeros_like(params)
            v_t = np.zeros_like(params)
            t = 0
            
            for it in range(5000):
                if time.time() - start_time > max_time:
                    break
                if time.time() > deadline:
                    break
                
                t += 1
                
                U = params[:size_u*R].reshape((size_u, R))
                V = params[size_u*R : size_u*R + size_v*R].reshape((size_v, R))
                W = params[size_u*R + size_v*R :].reshape((size_w, R))
                
                Ur = U.reshape((n, m, R))
                Vr = V.reshape((m, p, R))
                Wr = W.reshape((p, n, R))
                
                T_recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
                diff = T_recon - target
                
                # Compute gradients
                dU = np.einsum('ijk,jkr,kir->ijr', diff, Vr, Wr)
                dV = np.einsum('ijk,ijr,kir->jkr', diff, Ur, Wr)
                dW = np.einsum('ijk,ijr,jkr->kir', diff, Ur, Vr)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                
                # Adam update
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)
                m_hat = m_t / (1 - beta1 ** t)
                v_hat = v_t / (1 - beta2 ** t)
                
                params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
                
                # Check convergence every 500 iterations
                if it % 500 == 0:
                    U_chk = params[:size_u*R].reshape((size_u, R))
                    V_chk = params[size_u*R : size_u*R + size_v*R].reshape((size_v, R))
                    W_chk = params[size_u*R + size_v*R :].reshape((size_w, R))
                    T_chk = reconstruct(U_chk, V_chk, W_chk, R)
                    err = np.sum((T_chk - target) ** 2)
                    if err < 1e-12:
                        break
            
            # Extract and try to round
            U_opt = params[:size_u*R].reshape((size_u, R))
            V_opt = params[size_u*R : size_u*R + size_v*R].reshape((size_v, R))
            W_opt = params[size_u*R + size_v*R :].reshape((size_w, R))
            
            # Try rounding to half-integers
            U_r = np.round(U_opt * 2) / 2
            V_r = np.round(V_opt * 2) / 2
            W_r = np.round(W_opt * 2) / 2
            
            T_r = reconstruct(U_r, V_r, W_r, R)
            if np.allclose(T_r, target, atol=1e-9):
                return U_r, V_r, W_r, True
        
        return None, None, None, False
    
    # Try to reduce rank below 26
    best_result = None
    
    for R in [23, 24, 25]:
        if time.time() > deadline - 10:
            break
        
        result = try_reduce_rank(R, max_time=35.0)
        if result[3]:
            best_result = (R, result[0], result[1], result[2])
            break
    
    # If no reduction found, use rank 26
    if best_result is None:
        R = 26
        U = U26
        V = V26
        W = W26
    else:
        R, U, V, W = best_result
    
    # Final verification
    T_recon = reconstruct(U, V, W, R)
    if not np.allclose(T_recon, target, atol=1e-9):
        # Ultimate fallback: schoolbook rank 27
        R = 27
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i,j), col] = 1
                    V[idx_v(j,k), col] = 1
                    W[idx_w(k,i), col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
