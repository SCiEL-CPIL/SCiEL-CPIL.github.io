"""Program for matrix multiplication tensor decomposition 3x5x5."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s safety margin within 150s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k"""
        rank = U.shape[1]
        reconstructed = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        target = np.ones((n, m, p))
        return np.max(np.abs(reconstructed - target)) < tol
    
    def objective(params, n, m, p, target_rank):
        """Minimize ||sum_c U_c ⊗ V_c ⊗ W_c - T||_F^2"""
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank:2*n*m*target_rank+m*p*target_rank]
        W_flat = params[2*n*m*target_rank+m*p*target_rank:]
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        reconstructed = np.zeros((n, m, p))
        for c in range(target_rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        diff = reconstructed - np.ones((n, m, p))
        return np.sum(diff ** 2)
    
    def try_rank(rank, n, m, p, deadline, seed=42):
        """Try to find a valid decomposition at the given rank"""
        if time.time() > deadline - 30:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random matrices
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use gradient descent with Adam-like update
        lr = 0.01
        best_loss = float('inf')
        best_params = params.copy()
        
        # Simple gradient descent with momentum
        velocity = np.zeros_like(params)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        m_v = np.zeros_like(params)
        v_v = np.zeros_like(params)
        t = 0
        
        for iteration in range(5000):
            if time.time() > deadline - 15:
                break
            
            loss = objective(params, n, m, p, rank)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if loss < 1e-10:
                break
            
            # Numerical gradient
            grad = np.zeros_like(params)
            step = 1e-5
            for idx in range(0, len(params), 100):  # Sample some indices for speed
                if time.time() > deadline - 10:
                    break
                for i in range(idx, min(idx + 100, len(params))):
                    params[i] += step
                    loss_plus = objective(params, n, m, p, rank)
                    params[i] -= 2 * step
                    loss_minus = objective(params, n, m, p, rank)
                    params[i] += step
                    grad[i] = (loss_plus - loss_minus) / (2 * step)
            
            # Adam update
            t += 1
            m_v = beta1 * m_v + (1 - beta1) * grad
            v_v = beta2 * v_v + (1 - beta2) * grad ** 2
            m_hat = m_v / (1 - beta1 ** t)
            v_hat = v_v / (1 - beta2 ** t)
            params -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Verify best solution
        U_best = best_params[:n*m*rank].reshape(n * m, rank)
        V_best = best_params[n*m*rank:2*n*m*rank+m*p*rank].reshape(m * p, rank)
        W_best = best_params[2*n*m*rank+m*p*rank:].reshape(p * n, rank)
        
        if verify_decomposition(U_best, V_best, W_best, n, m, p, tol=1e-6):
            return {"U": U_best, "V": V_best, "W": W_best}
        
        return None
    
    # First, establish the schoolbook rank-75 solution as baseline
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known results suggest rank around 69-72 might be achievable
    for target_rank in [74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(target_rank, n, m, p, deadline, seed=42 + seed)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        
        if found:
            # Found a valid decomposition, try lower rank
            continue
    
    # Convert to lists for return
    return {
        "U": best_result["U"].tolist(),
        "V": best_result["V"].tolist(),
        "W": best_result["W"].tolist()
    }
# EVOLVE-BLOCK-END
