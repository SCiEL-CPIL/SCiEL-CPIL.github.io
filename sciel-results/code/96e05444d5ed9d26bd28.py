import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(nm)]
    V = [[0] * schoolbook_rank for _ in range(mp)]
    W = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank

    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank using random initialization and local search
    # This is a structural change: using optimization search
    for target_rank in range(schoolbook_rank - 1, schoolbook_rank - 5, -1):
        if time.time() > deadline:
            break
        # Random search
        for _ in range(10):
            if time.time() > deadline:
                break
            U_new = [[np.random.randint(-2, 3) * 0.5 for _ in range(target_rank)] for _ in range(nm)]
            V_new = [[np.random.randint(-2, 3) * 0.5 for _ in range(target_rank)] for _ in range(mp)]
            W_new = [[np.random.randint(-2, 3) * 0.5 for _ in range(target_rank)] for _ in range(pn)]
            if verify(U_new, V_new, W_new, target_rank):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break

    return best_result
