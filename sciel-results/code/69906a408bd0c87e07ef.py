"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    def compute_T(U, V, W):
        """Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        r = U.shape[1]
        T = np.zeros((n, m, p, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k,:] = U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]
        return np.sum(T, axis=3)
    
    def verify(U, V, W):
        T = compute_T(U, V, W)
        return np.max(np.abs(T - 1.0)) < 1e-9
    
    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.3
        V = np.random.randn(m*p, r) * 0.3
        W = np.random.randn(p*n, r) * 0.3
        
        lr = 0.01
        for step in range(300):
            if time.time() > deadline:
                return None
            
            T = compute_T(U, V, W)
            err = T - 1.0
            loss = np.sum(err**2)
            if loss < 1e-16:
                return (U, V, W)
            
            # Vectorized gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = 2 * err[i,j,k]
                        dU[i*m+j,:] += e * (V[j*p+k,:] * W[k*n+i,:])
                        dV[j*p+k,:] += e * (U[i*m+j,:] * W[k*n+i,:])
                        dW[k*n+i,:] += e * (U[i*m+j,:] * V[j*p+k,:])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if verify(U, V, W):
            return (U, V, W)
        return None
    
    for r in [30, 36, 40, 45, 50, 55, 60]:
        if time.time() > deadline:
            break
        for seed in [42, 123]:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
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
# EVOLVE-BLOCK-END
