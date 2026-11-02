import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 4, 5
    nm, mp, pn = 8, 20, 10
    
    # Build schoolbook
    rank = n * m * p  # 40
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W, r, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    if abs(np.dot(U[a,:r], V[b,:r]*W[c,:r]) - 1.0) > tol:
                        return False
        for _ in range(3000):
            a, b, c = np.random.randint(nm), np.random.randint(mp), np.random.randint(pn)
            if T[a,b,c] == 0:
                if abs(np.dot(U[a,:r], V[b,:r]*W[c,:r])) > tol:
                    return False
        return True
    
    # Compute flattening ranks for lower bound
    F1 = T.reshape(nm * mp, pn)
    F2 = T.reshape(nm * pn, mp)
    F3 = T.reshape(mp * pn, nm)
    
    r1 = np.linalg.matrix_rank(F1, tol=1e-10)
    r2 = np.linalg.matrix_rank(F2, tol=1e-10)
    r3 = np.linalg.matrix_rank(F3, tol=1e-10)
    lower_bound = max(r1, r2, r3)
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Search for minimal rank decomposition
    for target_rank in range(lower_bound, rank + 1):
        if time.time() > deadline - 40:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(100 + restart * 11)
            U_t = np.random.randn(nm, target_rank) * 0.3
            V_t = np.random.randn(mp, target_rank) * 0.3
            W_t = np.random.randn(pn, target_rank) * 0.3
            
            lr = 0.08
            prev_err = float('inf')
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                
                grad_U = np.zeros_like(U_t)
                grad_V = np.zeros_like(V_t)
                grad_W = np.zeros_like(W_t)
                error = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a, b, c = i*m+j, j*p+k, k*n+i
                            val = np.dot(U_t[a,:], V_t[b,:]*W_t[c,:])
                            diff = val - 1.0
                            error += diff*diff
                            vw = V_t[b,:]*W_t[c,:]
                            grad_U[a,:] += 2*diff*vw
                            grad_V[b,:] += 2*diff*(U_t[a,:]*W_t[c,:])
                            grad_W[c,:] += 2*diff*(U_t[a,:]*V_t[b,:])
                
                for _ in range(300):
                    a, b, c = np.random.randint(nm), np.random.randint(mp), np.random.randint(pn)
                    if T[a,b,c] == 0:
                        val = np.dot(U_t[a,:], V_t[b,:]*W_t[c,:])
                        error += val*val
                        vw = V_t[b,:]*W_t[c,:]
                        grad_U[a,:] += 2*val*vw
                        grad_V[b,:] += 2*val*(U_t[a,:]*W_t[c,:])
                        grad_W[c,:] += 2*val*(U_t[a,:]*V_t[b,:])
                
                if error < 1e-14:
                    U_r = np.round(U_t * 2) / 2
                    V_r = np.round(V_t * 2) / 2
                    W_r = np.round(W_t * 2) / 2
                    if verify(U_r, V_r, W_r, target_rank):
                        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, target_rank
                        found = True
                        break
                
                # Adaptive learning rate
                U_t -= lr * grad_U
                V_t -= lr * grad_V
                W_t -= lr * grad_W
                lr *= 0.995
                if error > prev_err * 1.05:
                    lr *= 0.5
                prev_err = error
            
            if found:
                break
        
        if found:
            break
    
    result = {
        "U": [[best_U[i, j] for j in range(best_rank)] for i in range(nm)],
        "V": [[best_V[i, j] for j in range(best_rank)] for i in range(mp)],
        "W": [[best_W[i, j] for j in range(best_rank)] for i in range(pn)]
    }
    
    if not verify(best_U, best_V, best_W, best_rank):
        result = {
            "U": [[U[i, j] for j in range(rank)] for i in range(nm)],
            "V": [[V[i, j] for j in range(rank)] for i in range(mp)],
            "W": [[W[i, j] for j in range(rank)] for i in range(pn)]
        }
    
    return result
