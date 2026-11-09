import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 110  # safety margin
    
    # Target: all-ones tensor of size (n, m, p)
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try increasing ranks
    for rank in range(3, 30):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        np.random.seed(rank * 42)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Use iterative optimization
        # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        for iteration in range(500):
            if time.time() > deadline - 15:
                break
            
            # Compute gradient
            # For each (i,j,k): residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
            # dU[i*m+j,r] = residual * V[j*p+k,r] * W[k*n+i,r]
            # dV[j*p+k,r] = residual * U[i*m+j,r] * W[k*n+i,r]
            # dW[k*n+i,r] = residual * U[i*m+j,r] * V[j*p+k,r]
            
            dU = np.zeros((n*m, rank))
            dV = np.zeros((m*p, rank))
            dW = np.zeros((p*n, rank))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute residual
                        res = 0.0
                        for r in range(rank):
                            res += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        res -= 1.0
                        
                        for r in range(rank):
                            dU[i*m+j, r] += res * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += res * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += res * U[i*m+j, r] * V[j*p+k, r]
            
            # Update with learning rate
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded solution is exact
        exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        exact = False
                        break
                if not exact:
                    break
            if not exact:
                break
        
        if exact:
            # Convert to lists
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: naive decomposition
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
