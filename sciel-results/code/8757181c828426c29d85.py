"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
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
    
    # Build target tensor
    S = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_result = schoolbook()
    
    # Try to find lower rank decompositions
    lb = max(n * m, m * p, n * p)
    
    for rank in range(lb, n * m * p):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(rank * 100000 + attempt * 13)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Gradient descent
            lr = 0.01
            for step in range(3000):
                if time.time() > deadline - 20:
                    break
                
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_approx - S
                err = np.linalg.norm(diff)
                
                if err < 1e-10:
                    break
                
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,ar,br->cr', diff, U, V)
                
                gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if gnorm < 1e-15:
                    break
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 500 == 0:
                    lr *= 0.95
            
            T_final = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_final - S)
            
            if err < 1e-8:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.linalg.norm(T_r - S) < 1e-6:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
