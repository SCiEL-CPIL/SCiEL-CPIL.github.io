import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0

    # Build the rank-28 Strassen decomposition
    U28 = np.zeros((n * m, 28), dtype=float)
    V28 = np.zeros((m * p, 28), dtype=float)
    W28 = np.zeros((p * n, 28), dtype=float)

    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    w_map_L = [0, 2, 1, 3]
    w_map_R = [4, 6, 5, 7]

    col = 0
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U28[u_map_L[idx], col] += val
        for idx, val in v_t:
            V28[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W28[w_map_L[idx], col] += val
        col += 1
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U28[u_map_R[idx], col] += val
        for idx, val in v_t:
            V28[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W28[w_map_L[idx], col] += val
        col += 1
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U28[u_map_L[idx], col] += val
        for idx, val in v_t:
            V28[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W28[w_map_R[idx], col] += val
        col += 1
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U28[u_map_R[idx], col] += val
        for idx, val in v_t:
            V28[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W28[w_map_R[idx], col] += val
        col += 1

    # Attempt rank reduction via linear dependency search on flattened tensors
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    flat_dim = dim_u * dim_v * dim_w
    T_flat = np.zeros((28, flat_dim))
    for c in range(28):
        T_flat[c] = np.kron(np.kron(U28[:, c], V28[:, c]), W28[:, c])

    # Find a minimal spanning subset using QR
    A = T_flat.T  # flat_dim x 28
    Q, R = np.linalg.qr(A)
    diag_R = np.abs(np.diag(R))
    tol = 1e-10 * max(diag_R) if len(diag_R) > 0 else 1e-10
    independent = list(np.where(diag_R > tol)[0])

    best_rank = len(independent)
    best_indices = independent

    # Verify the reduced decomposition
    if best_rank < 28:
        U_best = U28[:, best_indices]
        V_best = V28[:, best_indices]
        W_best = W28[:, best_indices]
        
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_actual = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(best_rank):
                    for jj in range(m):
                        acc += U_best[ii * m + jj, cc] * V_best[jj * p + kk, cc] * W_best[kk * n + ii, cc]
                C_actual[ii, kk] = acc
        
        if not np.allclose(C_expected, C_actual, atol=1e-10):
            best_rank = 28
            best_indices = list(range(28))
            U_best, V_best, W_best = U28, V28, W28
    else:
        U_best, V_best, W_best = U28, V28, W28

    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
        
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

def _verify_decomposition(U, V, W, n=2, m=4, p=4, seed=42):
    """Verify that the tensor decomposition correctly computes matrix multiplication."""
    np.random.seed(seed)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(len(U[0])):
                for jj in range(m):
                    acc += U[ii * m + jj][cc] * V[jj * p + kk][cc] * W[kk * n + ii][cc]
            Cf[ii, kk] = acc
            
    return np.allclose(C, Cf, atol=1e-10)

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = res["U"]
    V = res["V"]
    W = res["W"]
    
    if _verify_decomposition(U, V, W, n, m, p):
        print("Verification Passed")
    else:
        print("Verification Failed")
    print("Rank:", len(U[0]))
