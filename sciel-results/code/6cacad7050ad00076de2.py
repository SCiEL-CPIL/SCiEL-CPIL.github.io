import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find a better decomposition
    for rank in [55, 50, 45, 40]:
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            break
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    for attempt in range(2000):
        if time.time() > deadline:
            return None
        
        U = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(m*p)]
        
        W = [[0]*rank for _ in range(p*n)]
        valid = True
        
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i*m+j][c] * V[j*p+k][c] for c in range(rank)]
                    A.append(row)
                
                try:
                    A_np = np.array(A, dtype=float)
                    b = np.ones(m)
                    W_vec, _, _, _ = np.linalg.lstsq(A_np, b, rcond=None)
                    if not np.allclose(A_np @ W_vec, b, atol=1e-9):
                        valid = False
                        break
                    W[k*n+i] = [int(round(x)) for x in W_vec]
                except:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if total != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U, "V": V, "W": W}
    
    return None
