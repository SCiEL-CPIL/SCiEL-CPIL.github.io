import time
import numpy as np
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_fast(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        result = np.einsum('ijr,klr,mnr->ijklmn', U_r, V_r, W_r)
        
        expected = np.zeros((n, m, m, p, p, n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        expected[i, j, j, k, k, i2] = 1
        
        return np.allclose(result, expected, atol=1e-10)
    
    rng = np.random.RandomState(42)
    
    # Try a few random searches at various ranks
    for rank in range(28, min(best_rank, 50)):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            U = rng.randint(-2, 3, size=(nm, rank)).tolist()
            V = rng.randint(-2, 3, size=(mp, rank)).tolist()
            W = rng.randint(-2, 3, size=(pn, rank)).tolist()
            if verify_fast(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                break
        if best_rank == rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
