import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    def check_solution(U, V, W):
        # Check if U, V, W form a valid decomposition
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        if np.allclose(S, T, atol=1e-5):
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            S_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(S_r, T, atol=1e-9):
                return True, U_r, V_r, W_r
        return False, None, None

    def try_rank(R, time_limit):
        deadline = time.time() + time_limit
        restarts = 10
        for _ in range(restarts):
            if time.time() > deadline:
                break
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            lr = 0.05
            for it in range(5000):
                if time.time() > deadline:
                    break
                S = np.einsum('ur,vr,wr->uvw', U, V, W)
                E = T - S
                
                grad_U = np.einsum('uvw,vr,wr->ur', E, V, W)
                grad_V = np.einsum('uvw,ur,wr->vr', E, U, W)
                grad_W = np.einsum('uvw,ur,vr->wr', E, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 50 == 0:
                    valid, Ur, Vr, Wr = check_solution(U, V, W)
                    if valid:
                        return Ur, Vr, Wr
        return None

    # Try ranks from 23 down to 12
    # Allocate more time for lower ranks
    total_time = 120
    elapsed = time.time() - start_time
    remaining = total_time - elapsed - 10 # 10s safety margin
    
    # Simple heuristic: try 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12
    ranks_to_try = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    time_per_rank = remaining / len(ranks_to_try)
    
    for R in ranks_to_try:
        if time.time() - start_time > total_time - 15:
            break
        res = try_rank(R, time_per_rank)
        if res is not None:
            Ur, Vr, Wr = res
            return {
                "U": Ur.tolist(),
                "V": Vr.tolist(),
                "W": Wr.tolist()
            }
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
