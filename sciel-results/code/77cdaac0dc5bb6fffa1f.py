import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    M = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for c in range(m):
            for r_prime in range(p):
                i = r * m + c
                j = c * p + r_prime
                k = r * n + r_prime
                M[i, j, k] = 1.0

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Try to find rank 23
    R = 23
    # Initialize with random vectors
    # Use a fixed seed for reproducibility? No, random restarts are better.
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Scale
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Gradient Descent
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Compute tensor product
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = T - M
            loss = np.sum(error**2)
            
            if loss < 1e-6:
                # Found a solution
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
                
            # Gradient
            # dL/dU_ir = 2 * sum_jk error_ijk V_jr W_kr
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', error, V, W)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', error, U, W)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', error, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Normalize to prevent scale drift
            for r in range(R):
                if np.linalg.norm(U[:, r]) > 1e-10:
                    U[:, r] /= np.linalg.norm(U[:, r])
                if np.linalg.norm(V[:, r]) > 1e-10:
                    V[:, r] /= np.linalg.norm(V[:, r])
                if np.linalg.norm(W[:, r]) > 1e-10:
                    W[:, r] /= np.linalg.norm(W[:, r])
                    
        if best_rank == R:
            break
            
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
