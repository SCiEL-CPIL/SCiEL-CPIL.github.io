import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r] == 1 for all i,j,k
    # Note: W index is k*n + i? 
    # Task says: W[k*n+i, r].
    # My previous thought: W[i, k].
    # Let's check the task description: "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is $k \cdot n + i$.
    # In my equation: W[i, k] -> index $i \cdot n + k$? No.
    # If W index is $k \cdot n + i$, then for fixed $k, i$, the value is $W[k \cdot n + i, r]$.
    # So the term is $U[i \cdot m + j, r] \cdot V[j \cdot p + k, r] \cdot W[k \cdot n + i, r]$.
    
    start_time = time.time()
    timeout = 110.0 # 10s margin
    
    def verify(U, V, W, R):
        # Check if sum is 1 for all i,j,k
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape U to (n, m, R)
        # Reshape V to (m, p, R)
        # Reshape W to (p, n, R)
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # Compute sum for all i,j,k
        # Sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W_r is (p, n, R). Index is (k, i).
        # So W_r[k, i, r]
        
        # Use einsum or broadcasting
        # U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :]
        # Sum over last axis.
        
        # Let's iterate for simplicity and clarity, or use vectorization
        # Vectorized:
        # U_r: (n, m, R)
        # V_r: (m, p, R)
        # W_r: (p, n, R)
        
        # We want S[i, j, k] = sum_r U_r[i, j, r] V_r[j, k, r] W_r[k, i, r]
        
        # This is tricky to do in one einsum because indices are permuted.
        # Let's just loop over i, j, k (27 iterations)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_r[i, j, :]
                    v_val = V_r[j, k, :]
                    w_val = W_r[k, i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(R):
        # Initialize U, V, W
        # Use random initialization
        # Or use schoolbook-like initialization if R >= 27
        # For R < 27, random is better
        
        np.random.seed(42) # Fixed seed for reproducibility in test? No, random is fine.
        
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale to make initial error smaller?
        # The target sum is 1.
        # If U, V, W are ~ N(0, 1), sum of 23 products is ~ N(0, sqrt(23)*1) ~ 4.8.
        # So error is large.
        
        # Simple gradient descent
        # Error E = sum (S[i,j,k] - 1)^2
        # dE/dU = 2 * (S-1) * V * W
        # dE/dV = 2 * (S-1) * U * W
        # dE/dW = 2 * (S-1) * U * V
        
        lr = 0.01
        for step in range(10000):
            if time.time() - start_time > timeout:
                return None
            
            # Compute S
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            S = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S[i, j, k] = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
            
            error = np.sum((S - 1.0) ** 2)
            if error < 1e-12:
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                if verify(U_int, V_int, W_int, R):
                    return U_int, V_int, W_int
                # If rounding fails, try to fix?
                # For now, return None and try another restart
            
            # Gradient
            # dU[i,j,r] = 2 * sum_k (S[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
            # dV[j,k,r] = 2 * sum_i (S[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
            # dW[k,i,r] = 2 * sum_j (S[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2 * (S[i, j, k] - 1.0)
                        # dU[i,j,:] += err * V[j,k,:] * W[k,i,:]
                        dU[i * m + j, :] += err * V_r[j, k, :] * W_r[k, i, :]
                        # dV[j,k,:] += err * U[i,j,:] * W[k,i,:]
                        dV[j * p + k, :] += err * U_r[i, j, :] * W_r[k, i, :]
                        # dW[k,i,:] += err * U[i,j,:] * V[j,k,:]
                        dW[k * n + i, :] += err * U_r[i, j, :] * V_r[j, k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            if step % 100 == 0:
                lr *= 0.95
        
        return None

    # Try ranks from 23 up to 27
    for R in range(23, 28):
        if time.time() - start_time > timeout:
            break
        
        # Multiple restarts for each rank
        for attempt in range(50):
            if time.time() - start_time > timeout:
                break
            
            np.random.seed(attempt * 1000 + R)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale
            U /= np.sqrt(R)
            V /= np.sqrt(R)
            W /= np.sqrt(R)
            
            # Run local search
            # Re-implementing the loop inside try_rank is better for isolation
            # But for speed, I'll just call a helper
            
            res = try_rank(R)
            if res:
                U_f, V_f, W_f = res
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        
    # Fallback to schoolbook
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
