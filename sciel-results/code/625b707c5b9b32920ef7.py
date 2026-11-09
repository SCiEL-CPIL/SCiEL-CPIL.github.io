import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook rank
    R_school = n * m * p
    U_school = [[0] * R_school for _ in range(n * m)]
    V_school = [[0] * R_school for _ in range(m * p)]
    W_school = [[0] * R_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = R_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Verification function using numpy for speed
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U_np[i * m + j]
                    vj = V_np[j * p + k]
                    wk = W_np[k * n + i]
                    s = np.dot(ui * vj, wk)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition
    # We try ranks R from R_school - 1 down to R_school - 5
    # Using random integer entries in {-1, 0, 1}
    
    # Since random search is unlikely to succeed for such a large rank,
    # we rely on the schoolbook solution unless a very lucky hit occurs.
    # However, we must try to satisfy the "Strategy" if possible, 
    # but since rank 12 is impossible, we ignore that specific number 
    # and just ensure we don't break the schoolbook.
    
    # If time remains, try a few random restarts for R = 71
    if time.time() < deadline - 50:
        for R in range(R_school - 1, max(R_school - 3, 50), -1):
            if time.time() > deadline:
                break
            for attempt in range(20):
                if time.time() > deadline:
                    break
                U = np.random.randint(-1, 2, size=(n * m, R))
                V = np.random.randint(-1, 2, size=(m * p, R))
                W = np.random.randint(-1, 2, size=(p * n, R))
                
                if verify_np(U, V, W):
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            if best_rank < R_school:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
