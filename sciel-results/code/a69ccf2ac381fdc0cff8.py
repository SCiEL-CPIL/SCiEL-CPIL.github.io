"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = n * m * p  # 63 for 3x3x7

    # Pre-allocate as numpy arrays (int8 is sufficient for 0/1 entries)
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        if time.time() > deadline:
            break

    # Verify the decomposition is correct using efficient tensor contraction
    U_arr = U.astype(np.float64)
    V_arr = V.astype(np.float64)
    W_arr = W.astype(np.float64)

    # Build the target tensor: T[a,b,c] = 1 iff indices are consistent
    # a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Compute the decomposition tensor efficiently
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    if not np.allclose(T_comp, T_target, atol=1e-10):
        raise ValueError("Verification failed")

    # Convert to list-of-lists to match expected interface exactly
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
