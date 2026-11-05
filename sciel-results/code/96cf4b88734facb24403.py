"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Search for a low-rank decomposition using random initialization and local optimization.
    # We aim for a rank R < 72. The tensor is T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We will try ranks from 40 up to 71.
    # For each rank, we try several random initializations and use a simple gradient descent step.
    
    def random_search(rank, iterations, seed):
        np.random.seed(seed)
        
        # Initialize U, V, W with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Target tensor T
        T = np.zeros((n * m, m * p, p * n)) # This is too large, we need to be careful.
        # Actually, the tensor is T[i,j,k]. Let's represent it as a 3D array of size n x m x p.
        T_tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_tensor[i, j, k] = 1.0
        
        # We want to minimize || T_tensor - sum_c U_c V_c W_c ||_F^2
        # where U_c is a vector of size n*m, V_c is size m*p, W_c is size p*n.
        # The product is a tensor of size n x m x p.
        
        # Gradient descent
        lr = 0.01
        for it in range(iterations):
            # Compute current tensor
            current = np.zeros((n, m, p))
            for c in range(rank):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                # The term is u[i,j] * v[j,k] * w[k,i]
                # We can compute this efficiently using einsum or loops
                # current[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            # Residual
            residual = T_tensor - current
            
            # Check if close enough
            if np.linalg.norm(residual) < 1e-6:
                # Convert to list of lists with integers/half-integers
                U_list = [[0]*rank for _ in range(n*m)]
                V_list = [[0]*rank for _ in range(m*p)]
                W_list = [[0]*rank for _ in range(p*n)]
                for c in range(rank):
                    for idx in range(n*m):
                        U_list[idx][c] = round(U[idx, c] * 2) / 2
                    for idx in range(m*p):
                        V_list[idx][c] = round(V[idx, c] * 2) / 2
                    for idx in range(p*n):
                        W_list[idx][c] = round(W[idx, c] * 2) / 2
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Compute gradients
            # dL/dU[i,j,c] = -2 * residual[i,j,k] * V[j*p+k, c] * W[k*n+i, c] (summed over k)
            # dL/dV[j,k,c] = -2 * residual[i,j,k] * U[i*m+j, c] * W[k*n+i, c] (summed over i)
            # dL/dW[k,i,c] = -2 * residual[i,j,k] * U[i*m+j, c] * V[j*p+k, c] (summed over j)
            
            dU = np.zeros((n * m, rank))
            dV = np.zeros((m * p, rank))
            dW = np.zeros((p * n, rank))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = residual[i, j, k]
                        if res == 0: continue
                        for c in range(rank):
                            u_val = U[i*m+j, c]
                            v_val = V[j*p+k, c]
                            w_val = W[k*n+i, c]
                            dU[i*m+j, c] -= 2 * res * v_val * w_val
                            dV[j*p+k, c] -= 2 * res * u_val * w_val
                            dW[k*n+i, c] -= 2 * res * u_val * v_val
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if time.time() > deadline:
                return None
        
        return None
    
    # Try a few ranks
    for rank in range(40, 72):
        if time.time() > deadline:
            break
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            result = random_search(rank, 100, seed)
            if result:
                # Verify
                U = np.array(result['U'])
                V = np.array(result['V'])
                W = np.array(result['W'])
                T_tensor = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_tensor[i, j, k] = 1.0
                current = np.zeros((n, m, p))
                for c in range(rank):
                    u = U[:, c].reshape(n, m)
                    v = V[:, c].reshape(m, p)
                    w = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                current[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                if np.allclose(current, T_tensor, atol=1e-5):
                    best_result = result
                    best_rank = rank
                    break
        if best_rank < n * m * p:
            break
    
    return best_result
# EVOLVE-BLOCK-END
