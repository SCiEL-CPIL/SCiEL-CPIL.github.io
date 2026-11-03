"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 7

    # Schoolbook decomposition (rank = n*m*p = 63)
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Build the matrix multiplication tensor
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor(n, m, p)

    # Verify a decomposition
    def verify(U, V, W, tol=1e-8):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check all (i, j, k) triples that should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val - 1.0) > tol:
                        return False

        # Check zero entries (sample for efficiency)
        import random
        random.seed(42)
        checked = 0
        for _ in range(200):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j1 != j2 or i1 != i2 or k1 != k2:
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val) > tol:
                    return False
                checked += 1
        return True

    # Alternating Least Squares for tensor decomposition
    def als_decompose(target_rank, n_iter=200, tol=1e-10, seed=42):
        """Find U, V, W such that T[a,b,c] = sum_col U[a,col]*V[b,col]*W[c,col]"""
        rng = np.random.RandomState(seed)

        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5

        prev_cost = float('inf')

        for it in range(n_iter):
            if time.time() > deadline - 15:
                break

            # Update U: for each row a, solve least squares
            # T[a, b, c] = sum_col U[a, col] * V[b, col] * W[c, col]
            # For fixed a: T[a, :, :] flattened = U[a, :] @ M where M[b, c] = sum_col V[b,col]*W[c,col]... 
            # Actually, for fixed a, the equation is:
            # T[a, b, c] = sum_col U[a, col] * (V[b, col] * W[c, col])
            # Let's form: for each (b,c), we have a linear equation in U[a, :]
            # Stack all (b,c) pairs: (m*p*p*n) equations, target_rank unknowns

            for a in range(n*m):
                # Build design matrix: rows indexed by (b, c), columns by rank
                # M[b, c, col] = V[b, col] * W[c, col]
                M = np.outer(V, W).reshape(m*p, p*n, target_rank)
                M_flat = M.reshape(m*p*p*n, target_rank)
                rhs = T[a, :, :].flatten()
                U[a, :], residuals, rank_u, sv = np.linalg.lstsq(M_flat, rhs, rcond=None)

            # Update V
            for b in range(m*p):
                M = np.outer(U, W).reshape(n*m, p*n, target_rank)
                M_flat = M.reshape(n*m*p*n, target_rank)
                rhs = T[:, b, :].flatten()
                V[b, :], residuals, rank_v, sv = np.linalg.lstsq(M_flat, rhs, rcond=None)

            # Update W
            for c in range(p*n):
                M = np.outer(U, V).reshape(n*m, m*p, target_rank)
                M_flat = M.reshape(n*m*m*p, target_rank)
                rhs = T[:, :, c].flatten()
                W[c, :], residuals, rank_w, sv = np.linalg.lstsq(M_flat, rhs, rcond=None)

            # Compute cost
            cost = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        cost += (val - 1.0) ** 2

            if abs(prev_cost - cost) < tol and it > 10:
                break
            prev_cost = cost

        # Convert to lists
        U_list = [[float(x) for x in U[a, :]] for a in range(n*m)]
        V_list = [[float(x) for x in V[b, :]] for b in range(m*p)]
        W_list = [[float(x) for x in W[c, :]] for c in range(p*n)]

        if verify(U_list, V_list, W_list, tol=1e-7):
            return U_list, V_list, W_list, target_rank, cost
        return None

    # Try block-based decomposition: split into 3x3x3 + 3x3x4 blocks
    # For 3x3x3, known minimum rank is 21 (but schoolbook is 27)
    # Let's try to find good decompositions for sub-blocks

    # Try various target ranks with multiple random seeds
    best_result = None
    best_rank = n * m * p  # 63

    # First, get schoolbook as baseline
    U_school, V_school, W_school, rank_school = schoolbook(n, m, p)

    # Try ranks from high to low
    target_ranks = [56, 50, 45, 42, 40, 36, 35, 30, 28, 25, 21, 20]

    for target_rank in target_ranks:
        if time.time() > deadline - 25:
            break

        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            if found:
                break

            result = als_decompose(target_rank, n_iter=150, tol=1e-10, seed=seed + 100)
            if result is not None:
                U_list, V_list, W_list, r, cost = result
                if r < best_rank:
                    best_rank = r
                    best_result = (U_list, V_list, W_list)
                    found = True
                    print(f"Found rank {r} decomposition with seed {seed}, cost {cost:.2e}")
                    break

        if found:
            # Try to go lower
            continue
        else:
            # If we can't find this rank, stop trying lower
            break

    # If we found a better decomposition, return it
    if best_result is not None and best_rank < rank_school:
        U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}

    # Otherwise, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
