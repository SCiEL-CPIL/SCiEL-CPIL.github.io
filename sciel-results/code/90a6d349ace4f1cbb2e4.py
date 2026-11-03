"""Matrix multiplication 4x5 x 5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n + i, r] convention
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Best known: schoolbook rank 100
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        total = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
        
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        return True
    
    def verify_exact(U, V, W, rank):
        """Exact verification with integer arithmetic."""
        U = np.array(U, dtype=int)
        V = np.array(V, dtype=int)
        W = np.array(W, dtype=int)
        
        total = np.zeros((n, m, p), dtype=int)
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
        
        if not np.all(total == 1):
            return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Get schoolbook as initial best
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Strategy: SGD with l1 regularization to find sparse low-rank decompositions
    # We optimize over real-valued matrices and then try to round to integers
    
    def sgd_search(target_rank, num_iterations, lr, l1_lambda, seed):
        """Try to find a low-rank decomposition using SGD with l1 regularization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Precompute index arrays for efficient tensor computation
        # We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Target: T[i,j,k] = 1 for all i,j,k
        
        for it in range(num_iterations):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient of loss = 0.5 * ||T - 1||^2 + l1_lambda * (||U||_1 + ||V||_1 + ||W||_1)
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Compute T
            T = np.zeros((n, m, p), dtype=float)
            for r in range(target_rank):
                u_vec = U[:, r].reshape(n, m)
                v_vec = V[:, r].reshape(m, p)
                w_vec = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
            
            # Gradient of data term
            loss_data = 0.5 * np.sum((T - 1.0) ** 2)
            
            # Gradient w.r.t. U[i*m+j, r]: sum_{k} (T[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = T[i, j, k] - 1.0
                        for r in range(target_rank):
                            grad_U[i * m + j, r] += diff * V[j * p + k, r] * W[k * n + i, r]
                            grad_V[j * p + k, r] += diff * U[i * m + j, r] * W[k * n + i, r]
                            grad_W[k * n + i, r] += diff * U[i * m + j, r] * V[j * p + k, r]
            
            # Add l1 gradient
            grad_U += l1_lambda * np.sign(U)
            grad_V += l1_lambda * np.sign(V)
            grad_W += l1_lambda * np.sign(W)
            
            # SGD update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Every 100 iterations, try rounding and checking
            if (it + 1) % 100 == 0:
                # Try rounding to nearest integer
                U_int = np.round(U).astype(int)
                V_int = np.round(V).astype(int)
                W_int = np.round(W).astype(int)
                
                if verify_exact(U_int.tolist(), V_int.tolist(), W_int.tolist(), target_rank):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist(), True
        
        return None, None, None, False
    
    # Try different target ranks with SGD
    for target_rank in range(95, 79, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds and learning rates
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            for lr in [0.01, 0.005, 0.001]:
                if time.time() > deadline - 30:
                    break
                
                U_result, V_result, W_result, success = sgd_search(
                    target_rank, 500, lr, 0.01, seed
                )
                
                if success:
                    best_rank = target_rank
                    best_U = U_result
                    best_V = V_result
                    best_W = W_result
                    break
            
            if best_rank < n * m * p:
                break
        
        if best_rank < n * m * p:
            continue
    
    # If we found an improvement, return it; otherwise return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
