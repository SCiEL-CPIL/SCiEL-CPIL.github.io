import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 140
    
    n, m, p = 2, 3, 9
    rank_school = n * m * p  # 54
    
    # Build tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try lower ranks
    for r in range(rank_school - 1, 40, -1):
        if time.time() > deadline - 20:
            break
        # Try a few random inits
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            # Random init with small values
            Ur = np.random.randn(n*m, r) * 0.5
            Vr = np.random.randn(m*p, r) * 0.5
            Wr = np.random.randn(p*n, r) * 0.5
            # Check error
            # Compute sum_t Ur[:,t] outer Vr[:,t] outer Wr[:,t]
            # This is expensive; skip for now
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
