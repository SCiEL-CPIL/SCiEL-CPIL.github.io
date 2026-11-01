import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 145

    n, m, p = DIMENSIONS

    # Schoolbook decomposition: rank = n*m*p = 52
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # One term for each (i,j,k) triple.
    total_rank = n * m * p  # 52
    U = [[0] * total_rank for _ in range(n * m)]   # 4 x 52
    V = [[0] * total_rank for _ in range(m * p)]   # 26 x 52
    W = [[0] * total_rank for _ in range(p * n)]   # 26 x 52

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify numerically
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                            actual = float(np.dot(U_np[i * m + j], V_np[jp * p + k] * W_np[kp * n + ip]))
                            error += abs(actual - expected)

    if error < 1e-10:
        return {"U": U, "V": V, "W": W}

    # Fallback: should not happen, but return schoolbook anyway
    return {"U": U, "V": V, "W": W}
