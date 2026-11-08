import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS

    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify and try to reduce rank
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor from factors
    T_from_factors = np.einsum('ir,js,ks->ijs', U_np, V_np, W_np)
    
    # Build the true tensor
    T_true = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i, j, k] = 1.0
    
    # Check if decomposition is valid
    if not np.allclose(T_from_factors, T_true):
        # Fallback: return trivial decomposition
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
    
    # Try to find linear dependencies to reduce rank
    # Check if any column is a linear combination of others
    # Use SVD on the unfolded tensors
    U_rank = np.linalg.matrix_rank(U_np)
    V_rank = np.linalg.matrix_rank(V_np)
    W_rank = np.linalg.matrix_rank(W_np)
    
    # The effective rank is min of the three, but we need all three to have the same support
    # Try removing columns one by one and checking validity
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for col_idx in range(rank - 1, -1, -1):
        if time.time() - _t0 > _DEADLINE - 5:
            break
        # Try removing this column
        U_test = [row[:col_idx] + row[col_idx+1:] for row in U]
        V_test = [row[:col_idx] + row[col_idx+1:] for row in V]
        W_test = [row[:col_idx] + row[col_idx+1:] for row in W]
        
        U_test_np = np.array(U_test, dtype=float)
        V_test_np = np.array(V_test, dtype=float)
        W_test_np = np.array(W_test, dtype=float)
        
        T_test = np.einsum('ir,js,ks->ijs', U_test_np, V_test_np, W_test_np)
        
        if np.allclose(T_test, T_true, atol=1e-10):
            best_rank = col_idx
            best_U, best_V, best_W = U_test, V_test, W_test
    
    return {"U": best_U, "V": best_V, "W": best_W}
