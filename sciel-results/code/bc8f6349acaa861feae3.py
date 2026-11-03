import numpy as np
import time
import random
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check if a solution is valid (integer/half-integer entries and exact)
    def check_solution(U, V, W):
        # Check entries are close to integer or half-integer
        for matrix in [U, V, W]:
            arr = np.array(matrix)
            # Check if all entries are close to x/2 for integer x
            doubled = arr * 2
            rounded = np.round(doubled)
            if np.max(np.abs(doubled - rounded)) > 1e-6:
                return False, float('inf')
        
        # Check tensor decomposition
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        max_err = 0.0
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.sum(U_arr[u, :] * V_arr[v, :] * W_arr[w, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        # Check some zero entries
        for _ in range(50):
            u = random.randint(0, n*m-1)
            v = random.randint(0, m*p-1)
            w = random.randint(0, p*n-1)
            # Check if this is a valid (i,j,k)
            i1 = u // m
            j1 = u % m
            j2 = v // p
            k1 = v % p
            k2 = w // n
            i2 = w % n
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue  # Skip non-zero entries
            val = np.sum(U_arr[u, :] * V_arr[v, :] * W_arr[w, :])
            err = abs(val)
            if err > max_err:
                max_err = err
        
        if max_err < 1e-9:
            return True, max_err
        return False, max_err
    
    # ALS step
    def als_step(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Update U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                B = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    for r in range(rank):
                        B[k, r] = V[v, r] * W[w, r]
                U[u, :], residuals, rk, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update V
        for j in range(m):
            for k in range(p):
                v = j * p + k
                B = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    for r in range(rank):
                        B[i, r] = U[u, r] * W[w, r]
                V[v, :], residuals, rk, s = np.linalg.lstsq(B, b, rcond=None)
        
        # Update W
        for k in range(p):
            for i in range(n):
                w = k * n + i
                B = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    for r in range(rank):
                        B[j, r] = U[u, r] * V[v, r]
                W[w, :], residuals, rk, s = np.linalg.lstsq(B, b, rcond=None)
        
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try to find a solution with given rank
    def try_rank(rank, max_restarts=10):
        if time.time() > deadline:
            return None
        
        # Try random initialization
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            for iter in range(50):
                if time.time() > deadline:
                    break
                U, V, W = als_step(U.tolist(), V.tolist(), W.tolist(), rank)
                U = np.array(U)
                V = np.array(V)
                W = np.array(W)
                
                # Snap and check
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                valid, err = check_solution(U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                if valid:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        # Try pruning from schoolbook
        if rank < n * m * p:
            U0 = np.zeros((n*m, n*m*p))
            V0 = np.zeros((m*p, n*m*p))
            W0 = np.zeros((p*n, n*m*p))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        U0[u, col] = 1.0
                        V0[v, col] = 1.0
                        W0[w, col] = 1.0
                        col += 1
            
            # Try different subsets of columns
            indices = list(range(n*m*p))
            random.shuffle(indices)
            for trial in range(3):
                if time.time() > deadline:
                    break
                selected = indices[:rank]
                U = U0[:, selected]
                V = V0[:, selected]
                W = W0[:, selected]
                
                for iter in range(30):
                    if time.time() > deadline:
                        break
                    U, V, W = als_step(U.tolist(), V.tolist(), W.tolist(), rank)
                    U = np.array(U)
                    V = np.array(V)
                    W = np.array(W)
                    
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    valid, err = check_solution(U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                    if valid:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        return None
    
    # Search from low rank to high rank
    for rank in range(20, n*m*p + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
