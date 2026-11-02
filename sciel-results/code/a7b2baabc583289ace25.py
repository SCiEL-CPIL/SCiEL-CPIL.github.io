import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    
    # Target Tensor T
    # T[I, J, K] = 1 if I = i*m+j, J = j*p+k, K = k*n+i
    # Flattened T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    T_flat = T.flatten()
    
    def check_decomposition(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reconstruct T
        # T_recon[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
        # This is a tensor contraction.
        # Efficient way:
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        R = U.shape[1]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon.flatten(), T_flat, atol=1e-9)

    def round_to_half(x):
        return np.round(x * 2) / 2

    # Try to find a better rank
    # Start from rank_sb - 1 down to 36
    # Use random restarts and gradient descent
    
    best_rank = rank_sb
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
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
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    start_time = time.time()
    timeout = 100 # Leave margin
    
    for target_rank in range(rank_sb - 1, 35, -1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > timeout:
                break
                
            # Random init
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Gradient Descent
            lr = 0.01
            for step in range(100):
                if time.time() - start_time > timeout:
                    break
                    
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                error = T_recon - T
                err_flat = error.flatten()
                loss = np.sum(err_flat ** 2)
                
                if loss < 1e-10:
                    break
                    
                # Gradients
                # dL/dU[I, r] = sum_{J, K} error[I, J, K] V[J, r] W[K, r]
                # dL/dV[J, r] = sum_{I, K} error[I, J, K] U[I, r] W[K, r]
                # dL/dW[K, r] = sum_{I, J} error[I, J, K] U[I, r] V[J, r]
                
                dU = np.einsum('ijk,jr,kr->ir', error, V, W)
                dV = np.einsum('ijk,ir,kr->jr', error, U, W)
                dW = np.einsum('ijk,ir,jr->kr', error, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # Check
            if check_decomposition(U, V, W):
                # Round to half-integers
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                
                if check_decomposition(U_r, V_r, W_r):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        if found:
            break
            
    # Convert to lists for output
    U_list = [[float(val) for val in row] for row in best_U]
    V_list = [[float(val) for val in row] for row in best_V]
    W_list = [[float(val) for val in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
