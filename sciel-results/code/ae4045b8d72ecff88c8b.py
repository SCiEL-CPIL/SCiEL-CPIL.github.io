import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    for rank in range(14, 19):
        if time.time() > deadline - 20:
            break
        
        for seed in range(300):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize to avoid scaling issues
            for r in range(rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            lr = 0.1
            for iteration in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Compute error
                E = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            E[i,j,k] = 1.0 - sum(U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] for r in range(rank))
                
                err = np.sum(E**2)
                if err < 1e-12:
                    break
                
                # Gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = E[i,j,k]
                            for r in range(rank):
                                dU[i*m+j, r] += -e * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += -e * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += -e * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if verify(to_list(U), to_list(V), to_list(W), rank):
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback
    rank = n*m*p
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
