"""Matrix multiplication 3x4x4: rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def _validate(U, V, W):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if abs(s - 1) > 1e-8:
                    return False
    return True

def _try_rank(r, time_limit=8, seed=0):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    start = time.time()
    best_err = float('inf')
    best = None
    
    while time.time() - start < time_limit:
        # Initialize with structured random
        U = np_rng.randn(n*m, r) * 0.5
        V = np_rng.randn(m*p, r) * 0.5
        W = np_rng.randn(p*n, r) * 0.5
        
        # Alternating minimization style
        for it in range(300):
            if time.time() - start > time_limit:
                break
            i, j, k = rng.randint(0,n-1), rng.randint(0,m-1), rng.randint(0,p-1)
            a, b, ci = i*m+j, j*p+k, k*n+i
            Ua, Vb, Wci = U[a].copy(), V[b].copy(), W[ci].copy()
            val = np.dot(Ua, Vb*Wci)
            err = 1.0 - val
            lr = 0.03 * (0.995**it)
            U[a] += lr * err * Vb * Wci
            V[b] += lr * err * Ua * Wci
            W[ci] += lr * err * Ua * Vb
        
        # Compute error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, ci = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b]*W[ci])
                    err_sum += (val-1)**2
        if err_sum < best_err:
            best_err = err_sum
            best = (U.copy(), V.copy(), W.copy())
            if best_err < 1e-14:
                U_list = best[0].tolist()
                V_list = best[1].tolist()
                W_list = best[2].tolist()
                if _validate(U_list, V_list, W_list):
                    return U_list, V_list, W_list
    return None

def construct():
    t0 = time.time()
    deadline = 130
    
    for r in range(30, 49):
        if time.time() - t0 > deadline:
            break
        rem = deadline - (time.time() - t0)
        if rem < 3:
            break
        for seed in range(4):
            if time.time() - t0 > deadline:
                break
            tl = min(rem/4 - 0.5, 10)
            if tl < 1:
                break
            res = _try_rank(r, time_limit=tl, seed=seed)
            if res is not None:
                U, V, W = res
                if _validate(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
