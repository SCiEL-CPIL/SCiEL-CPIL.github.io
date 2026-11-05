import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 9, 21, 21
    
    idx_u = np.array([i*m+j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j*p+k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k*n+i for i in range(n) for j in range(m) for k in range(p)])
    
    best = None
    
    for rank in [30, 35, 40, 45, 50, 55, 60, 62]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 7919 + rank)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            lr = 0.05
            for it in range(3000):
                if time.time() > deadline - 15:
                    break
                
                vals = np.einsum('tr,sr,ur->t', U[idx_u], V[idx_v], W[idx_w])
                errs = 1.0 - vals
                loss = np.dot(errs, errs)
                
                if loss < 1e-14:
                    break
                
                dU = np.zeros((nm, rank))
                dV = np.zeros((mp, rank))
                dW = np.zeros((pn, rank))
                
                np.add.at(dU, (idx_u, slice(None)), errs[:, None] * V[idx_v] * W[idx_w])
                np.add.at(dV, (idx_v, slice(None)), errs[:, None] * U[idx_u] * W[idx_w])
                np.add.at(dW, (idx_w, slice(None)), errs[:, None] * U[idx_u] * V[idx_v])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 200 == 0:
                    lr *= 0.8
            
            vals = np.einsum('tr,sr,ur->t', U[idx_u], V[idx_v], W[idx_w])
            if np.max(np.abs(1.0 - vals)) < 1e-6:
                best = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best is None:
        rank = 63
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
