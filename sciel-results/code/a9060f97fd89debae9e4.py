import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Trivial decomposition as fallback
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_trivial[a_idx, col] = 1
                V_trivial[b_idx, col] = 1
                W_trivial[c_idx, col] = 1
                col += 1
    
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()
    best_rank = trivial_rank
    
    # Precompute index mappings
    idx_a = np.zeros((n, m, p), dtype=int)  # a_idx = i*m + j
    idx_b = np.zeros((n, m, p), dtype=int)  # b_idx = j*p + k  
    idx_c = np.zeros((n, m, p), dtype=int)  # c_idx = k*n + i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a[i,j,k] = i * m + j
                idx_b[i,j,k] = j * p + k
                idx_c[i,j,k] = k * n + i
    
    def compute_residual(U, V, W, rank):
        """Compute residual efficiently using einsum-like operations."""
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        residual = 0.0
        for i in range(n):
            for j in range(m):
                a_idx = i * m + j
                for k in range(p):
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    residual += (val - 1.0) ** 2
        return residual
    
    def optimize_decomposition(rank, U_init, V_init, W_init, max_iters=500, lr=0.01):
        """Simple gradient descent for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        best_res = compute_residual(U, V, W, rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    a_idx = i * m + j
                    for k in range(p):
                        b_idx = j * p + k
                        c_idx = k * n + i
                        # T[i,j,k] = sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,r]
                        # dT/du = V[b_idx,:] * W[c_idx,:]
                        # dT/dv = U[a_idx,:] * W[c_idx,:]
                        # dT/dw = U[a_idx,:] * V[b_idx,:]
                        val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                        err = val - 1.0
                        dU[a_idx, :] += err * (V[b_idx, :] * W[c_idx, :])
                        dV[b_idx, :] += err * (U[a_idx, :] * W[c_idx, :])
                        dW[c_idx, :] += err * (U[a_idx, :] * V[b_idx, :])
            
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            res = compute_residual(U, V, W, rank)
            if res < best_res:
                best_res = res
                best_U_local = U.copy()
                best_V_local = V.copy()
                best_W_local = W.copy()
            
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                lr *= 0.95
        
        return best_U_local, best_V_local, best_W_local, best_res
    
    # Try different ranks from low to high
    np.random.seed(42)
    
    # Strategy: try to find decomposition with rank < 63
    # Start with reasonable initializations
    
    for target_rank in [55, 50, 45, 40, 35]:
        if time.time() > deadline - 40:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + restart * 100 + target_rank)
            
            # Initialize with small random values
            scale = 0.5 / np.sqrt(target_rank)
            U_init = np.random.randn(n * m, target_rank) * scale
            V_init = np.random.randn(m * p, target_rank) * scale
            W_init = np.random.randn(p * n, target_rank) * scale
            
            # Add some structure: initialize with partial trivial decomposition
            # Use first min(target_rank, trivial_rank) columns from trivial
            num_trivial = min(target_rank, trivial_rank)
            for col in range(num_trivial):
                i = col // (m * p)
                rem = col % (m * p)
                j = rem // p
                k = rem % p
                if i < n and j < m and k < p:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_init[a_idx, col] = 1.0
                    V_init[b_idx, col] = 1.0
                    W_init[c_idx, col] = 1.0
            
            U_opt, V_opt, W_opt, res = optimize_decomposition(
                target_rank, U_init, V_init, W_init, 
                max_iters=300, lr=0.05
            )
            
            if res < 1e-10 and target_rank < best_rank:
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                best_rank = target_rank
                print(f"Found rank {target_rank} with residual {res:.2e}")
                break  # This rank works, try lower
        
        if best_rank <= target_rank:
            continue  # Found a good decomposition at this rank or lower
    
    # Final polish: try to improve the best found
    if best_rank < trivial_rank:
        if time.time() < deadline - 10:
            U_polish, V_polish, W_polish, res_polish = optimize_decomposition(
                best_rank, best_U, best_V, best_W,
                max_iters=200, lr=0.01
            )
            if res_polish < 1e-10:
                best_U = U_polish
                best_V = V_polish
                best_W = W_polish
    
    # Convert to lists and return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
