"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook fallback
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
        return {"U": U, "V": V, "W": W}
    
    def check_decomposition(U, V, W, tol=1e-6):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank using random search + local refinement
    # Start from rank 47 down
    for target_rank in range(47, 39, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(1000 + target_rank * 100 + attempt)
            # Initialize with schoolbook-like structure but with some merging
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Start with schoolbook for first min(target_rank, n*m*p) terms
            for r in range(min(target_rank, n*m*p)):
                i = r // (m*p)
                j = (r // p) % m
                k = r % p
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            
            # Add some random perturbations to allow merging
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j,r] of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                            for r in range(target_rank):
                                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]) - 1.0
                                grad_U[i*m+j, r] += 2 * val * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += 2 * val * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += 2 * val * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                err = sum((np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]) - 1.0)**2 
                         for i in range(n) for j in range(m) for k in range(p))
                if err < 1e-10:
                    break
            
            # Check if we found a valid decomposition
            U_list = [[round(float(U[i][r]), 10) for r in range(target_rank)] for i in range(n*m)]
            V_list = [[round(float(V[i][r]), 10) for r in range(target_rank)] for i in range(m*p)]
            W_list = [[round(float(W[i][r]), 10) for r in range(target_rank)] for i in range(p*n)]
            
            if check_decomposition(U_list, V_list, W_list, tol=1e-6):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
