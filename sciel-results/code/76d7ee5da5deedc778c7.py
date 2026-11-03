import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank, num_trials=5):
    """Verify the tensor decomposition on random inputs."""
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j
        # More precisely: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    global _START_TIME, _DEADLINE

    n, m, p = 3, 4, 8

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Part 1: i=0 (1x4x8 schoolbook)
    for j in range(m):
        for k in range(p):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1

    # Part 2: i=1,2 (2x4x8 Strassen)
    # Strassen matrices for 2x2x2
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                for l in range(4):
                    coeff = S_A[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_start + loc_j
                        U[g_i * m + g_j, col] += coeff

                for l in range(4):
                    coeff = S_B[t, l]
                    if coeff != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_start + loc_j
                        g_k = k_start + loc_k
                        V[g_j * p + g_k, col] += coeff

                for l in range(4):
                    coeff = S_C[t, l]
                    if coeff != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_start + loc_k
                        W[g_k * n + g_i, col] += coeff

                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    # Verify correctness before returning
    if not _verify(U, V, W, n, m, p, rank, num_trials=3):
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
