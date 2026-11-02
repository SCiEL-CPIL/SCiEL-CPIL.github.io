import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 110.0
    
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
        
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def gradient_search(rank, max_iters=1000, lr=0.01):
        # Initialize with random values
        U = np.random.rand(n*m, rank)
        V = np.random.rand(m*p, rank)
        W = np.random.rand(p*n, rank)
        
        # Scale to avoid large gradients
        U *= 2
        V *= 2
        W *= 2
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute errors
            # err[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
            # We can compute this efficiently using tensor contractions or loops.
            # For small sizes, loops are fine.
            err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err[i,j,k] = s - 1.0
            
            current_err = np.sum(err**2)
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if current_err < 1e-12:
                # Check if it's close to half-integers
                # If so, snap and verify
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                return None # Found a solution but not snap-able? Unlikely if error is 0.
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = err[i,j,k]
                        for r in range(rank):
                            dU[i*m+j, r] += 2 * e * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += 2 * e * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += 2 * e * U[i*m+j, r] * V[j*p+k, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic snapping and check
            if it % 50 == 0:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            
        return None

    # Try rank 22
    for attempt in range(10):
        if time.time() > deadline:
            break
        res = gradient_search(22, max_iters=200)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Try rank 23
    for attempt in range(10):
        if time.time() > deadline:
            break
        res = gradient_search(23, max_iters=200)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}

    return schoolbook()
