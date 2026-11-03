"""Matrix multiplication 3,3,6 rank-minimal decomposition."""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Vectorized approach
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is a trilinear form. We can compute it as:
        # For each (i,j), the contribution to T[i,j,:] is Uc[i,j] * Vc[j,:] * Wc[:,i]
        # So T[i,j,:] += Uc[i,j] * (Vc[j,:] * Wc[:,i])
        for i in range(n):
            for j in range(m):
                T[i,j,:] += Uc[i,j] * (Vc[j,:] * Wc[:,i])
    
    if np.allclose(T, 1.0, atol=tol):
        return True
    return False

def _compute_loss(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i,j,:] += Uc[i,j] * (Vc[j,:] * Wc[:,i])
    err = T - 1.0
    return np.sum(err**2)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    schoolbook = _schoolbook()
    
    # Try to find a lower rank decomposition
    # Start with rank 53 and go down
    for rank in range(53, 0, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(9, rank) * 0.1
            V = np.random.randn(18, rank) * 0.1
            W = np.random.randn(18, rank) * 0.1
            
            # Simple gradient descent
            lr = 0.05
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Compute loss and gradients
                T = np.zeros((3, 3, 6), dtype=float)
                for c in range(rank):
                    Uc = U[:, c].reshape(3, 3)
                    Vc = V[:, c].reshape(3, 6)
                    Wc = W[:, c].reshape(6, 3)
                    for i in range(3):
                        for j in range(3):
                            T[i,j,:] += Uc[i,j] * (Vc[j,:] * Wc[:,i])
                
                err = T - 1.0
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    if _verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(3):
                    for j in range(3):
                        for k in range(6):
                            for c in range(rank):
                                Uc = U[i*3+j, c]
                                Vc = V[j*6+k, c]
                                Wc = W[k*3+i, c]
                                dU[i*3+j, c] += 2 * err[i,j,k] * Vc * Wc
                                dV[j*6+k, c] += 2 * err[i,j,k] * Uc * Wc
                                dW[k*3+i, c] += 2 * err[i,j,k] * Uc * Vc
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
        
        # After trying this rank, check if we found a valid decomposition
        if time.time() > deadline - 5:
            break
    
    return schoolbook
