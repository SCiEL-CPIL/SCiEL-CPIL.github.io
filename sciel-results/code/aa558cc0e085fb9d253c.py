import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank
    deadline = time.time() + 100  # 100 second budget
    
    for target_rank in [24, 21, 18, 15, 12]:
        if time.time() > deadline:
            break
        # Try random search
        found = False
        for trial in range(5000):
            if time.time() > deadline:
                break
            U_r = np.random.randint(-2, 3, size=(n*m, target_rank))
            V_r = np.random.randint(-2, 3, size=(m*p, target_rank))
            W_r = np.random.randint(-2, 3, size=(p*n, target_rank))
            # Check if decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return {"U": U, "V": V, "W": W}
