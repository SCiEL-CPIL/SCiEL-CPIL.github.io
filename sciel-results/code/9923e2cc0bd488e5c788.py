import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = n * m * p  # 63 for 3x3x7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Full verification using NumPy for correctness
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For the trivial decomposition, T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # i.e., the indices are consistent: a%m == b//p and b%p == c//n and c%n == a//m

    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Compute the decomposition tensor
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This can be computed as: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
    # More efficiently: T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    if not np.allclose(T_comp, T_target, atol=1e-10):
        raise ValueError("Verification failed")

    return {"U": U, "V": V, "W": W}
