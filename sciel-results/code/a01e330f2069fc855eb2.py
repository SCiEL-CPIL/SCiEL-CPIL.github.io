import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 140
    
    min_rank = max(nm, mp, pn)  # 16
    
    for rank in range(min_rank, 49):
        if time.time() > deadline:
            break
        
        for trial in range(300):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(trial * 104729 + rank * 7919)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            U = np.zeros((nm, rank))
            valid = True
            
            for j in range(m):
                num_eq = n * p
                num_unk = n * rank
                A = np.zeros((num_eq, num_unk))
                b = np.ones(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for k in range(p):
                        for ii in range(n):
                            for c in range(rank):
                                A[eq_idx, ii * rank + c] = V[j * p + k, c] * W[k * n + i, c]
                        eq_idx += 1
                
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                    if np.max(np.abs(A @ x - b)) > 1e-8:
                        valid = False
                        break
                    for i in range(n):
                        U[i * m + j, :] = x[i * rank:(i+1) * rank]
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(nm)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(mp)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
