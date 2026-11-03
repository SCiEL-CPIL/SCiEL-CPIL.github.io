import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _build_strassen55(n, m, p):
    """Build the rank-55 Strassen-based decomposition."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_11][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1; V[v_22][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1; V[v_11][col] += -1
        W[w_00][col] += 1; W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1; U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1; W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1; U[u_11][col] += -1
        V[v_11][col] += 1; V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1; U[u_22][col] += -1
        V[v_21][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank
    return U, V, W


def _build_schoolbook(n, m, p):
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False

    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    # Check all zero positions (small enough to check all)
    total = (n * m) * (m * p) * (p * n)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in nonzero_set:
                    if abs(T[a, b, c]) > tol:
                        return False
    return True


def _is_valid_entries(U, V, W):
    """Check all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _als_optimize(U, V, W, n, m, p, max_iters=50, deadline=None):
    """
    ALS optimization for tensor decomposition.
    Given current U, V, W, try to find a lower-rank solution.
    """
    U_arr = np.array(U, dtype=np.float64).copy()
    V_arr = np.array(V, dtype=np.float64).copy()
    W_arr = np.array(W, dtype=np.float64).copy()

    for it in range(max_iters):
        if deadline and time.time() > deadline - 5:
            break

        # Update U: for each row a, solve (U[a,:] @ V.T @ W.T) = target
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Fix V, W: for each a, U[a,:] should satisfy:
        # sum_r U[a,r] * (V[b,r] * W[c,r]) = T_target[a,b,c]
        # This is: U[a,:] @ (V * W)^T = T_target[a,:,:].flatten()
        # where (V*W)[r,:] = V[:,r] * W[:,r] element-wise, shape (r, m*p*p*n)

        # More efficiently:
        # For fixed V, W: the tensor contribution from row a of U is:
        # T_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = U[a,:] @ (V^T * W^T)  -- no, let me think more carefully.
        # T[b,c] for fixed a: sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r U[a,r] * (V[b,r] * W[c,r])
        # Let M[a,b,c] = V[b,r]*W[c,r] summed over r weighted by U[a,r]
        # For each a: we need U[a,:] such that the resulting T[a,:,:] matches target.
        # T_target[a,b,c] = 1 if (a,b,c) is a valid mult index, else 0.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (U[a,:] @ V^T) * W^T ... no.
        # Let's define: for each r, the outer product V[:,r] ⊗ W[:,r] gives a (m*p, p*n) matrix.
        # T[a] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
        # So T[a] is a linear combination of the rank-1 matrices V[:,r]⊗W[:,r].
        # We want T[a] = T_target[a].
        # This is a linear system: U[a,:] @ A = T_target[a].flatten()
        # where A[r, (b,c)] = V[b,r] * W[c,r], shape (rank, m*p * p*n)

        # Build A once (it doesn't depend on a)
        # A shape: (rank, m*p, p*n) -> reshape to (rank, m*p*p*n)
        # A[r, b*p*n + c] = V[b,r] * W[c,r]
        A = np.einsum('br,cr->rbc', V_arr, W_arr)  # (rank, m*p, p*n)
        A_flat = A.reshape(A.shape[0], -1)  # (rank, m*p*p*n)

        # Target for each a
        for a in range(n * m):
            if deadline and time.time() > deadline - 5:
                break
            # Determine target T_target[a, b, c]
            # a = i*m + j, b = j'*p + k, c = k'*n + i'
            # Target = 1 if i=i', j=j', k=k' AND a = i*m+j, b=j*p+k, c=k*n+i
            # So for given a = i*m+j: target is 1 when b=j*p+k and c=k*n+i for same k.
            target = np.zeros(A_flat.shape[1])
            i = a // m
            j = a % m
            for k in range(p):
                b = j * p + k
                c = k * n + i
                idx = b * (p * n) + c
                target[idx] = 1.0

            # Solve U[a,:] @ A_flat = target
            # U[a,:] is (1, rank), A_flat is (rank, total), target is (total,)
            # This is underdetermined if rank < total. Use least squares.
            # Actually we want: A_flat.T @ U[a,:] = target
            # A_flat.T is (total, rank), U[a,:] is (rank,)
            sol, _, _, _ = np.linalg.lstsq(A_flat.T, target, rcond=None)
            U_arr[a, :] = sol

        # Update V similarly
        B = np.einsum('ar,cr->rac', U_arr, W_arr)  # (n*m, rank, p*n)
        B_flat = B.reshape(B.shape[0] * B.shape[1], -1)  # (n*m*rank, p*n)
        # For each b: V[b,:] @ (U ⊗ W)_b = target_b
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed b: T[:,b,:] = V[b,:] @ (U^T * W^T)
        # (U^T * W^T)[r, (a,c)] = U[a,r] * W[c,r]
        C_mat = np.einsum('ar,cr->rac', U_arr, W_arr)  # (n*m, rank, p*n)
        C_flat = C_mat.reshape(-1, C_mat.shape[1])  # (n*m*p*n, rank)
        # For each b: C_flat[:, b] ... no, let me restructure.
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For fixed b: T[:,b,:] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
        # Let D[r, a, c] = U[a,r] * W[c,r]
        D = np.einsum('ar,cr->arc', U_arr, W_arr)  # (rank, n*m, p*n)
        D_flat = D.reshape(D.shape[0], -1)  # (rank, n*m*p*n)
        for b in range(m * p):
            if deadline and time.time() > deadline - 5:
                break
            target = np.zeros(D_flat.shape[1])
            j = b // p
            k = b % p
            for i in range(n):
                a = i * m + j
                c = k * n + i
                idx = a * (p * n) + c
                target[idx] = 1.0
            sol, _, _, _ = np.linalg.lstsq(D_flat.T, target, rcond=None)
            V_arr[b, :] = sol

        # Update W
        E = np.einsum('ar,br->rab', U_arr, V_arr)  # (rank, n*m, m*p)
        E_flat = E.reshape(E.shape[0], -1)  # (rank, n*m*m*p)
        for c in range(p * n):
            if deadline and time.time() > deadline - 5:
                break
            target = np.zeros(E_flat.shape[1])
            k = c // n
            i = c % n
            for j in range(m):
                a = i * m + j
                b = j * p + k
                idx = a * (m * p) + b
                target[idx] = 1.0
            sol, _, _, _ = np.linalg.lstsq(E_flat.T, target, rcond=None)
            W_arr[c, :] = sol

    return U_arr, V_arr, W_arr


def _project_to_half_integers(U, V, W):
    """Project to nearest half-integers."""
    U_p = np.round(np.array(U) * 2) / 2.0
    V_p = np.round(np.array(V) * 2) / 2.0
    W_p = np.round(np.array(W) * 2) / 2.0
    return U_p, V_p, W_p


def _try_als_for_rank(n, m, p, target_rank, U55, V55, W55, deadline):
    """Try ALS from rank-55 init, truncating to target_rank."""
    U55_arr = np.array(U55, dtype=np.float64)
    V55_arr = np.array(V55, dtype=np.float64)
    W55_arr = np.array(W55, dtype=np.float64)

    # Truncate to target_rank
    U_init = U55_arr[:, :target_rank].copy()
    V_init = V55_arr[:, :target_rank].copy()
    W_init = W55_arr[:, :target_rank].copy()

    # Run ALS
    U_opt, V_opt, W_opt = _als_optimize(
        U_init.tolist(), V_init.tolist(), W_init.tolist(),
        n, m, p, max_iters=30, deadline=deadline
    )

    # Project to half-integers
    U_p, V_p, W_p = _project_to_half_integers(U_opt, V_opt, W_opt)

    # Verify
    if _verify(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p):
        if _is_valid_entries(U_p.tolist(), V_p.tolist(), W_p.tolist()):
            return U_p.tolist(), V_p.tolist(), W_p.tolist()

    # Also try without projection (in case exact solution exists)
    if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
        if _is_valid_entries(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()

    return None


def _random_rank_search(n, m, p, target_rank, deadline, max_trials=100):
    """Random search with ALS refinement."""
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break

        # Random initialization with small values
        values = [0, 0.5, 1.0, -0.5, -1.0]
        U = np.array([[random.choice(values) for _ in range(target_rank)]
                      for _ in range(n * m)], dtype=np.float64)
        V = np.array([[random.choice(values) for _ in range(target_rank)]
                      for _ in range(m * p)], dtype=np.float64)
        W = np.array([[random.choice(values) for _ in range(target_rank)]
                      for _ in range(p * n)], dtype=np.float64)

        # ALS refine
        U_opt, V_opt, W_opt = _als_optimize(
            U.tolist(), V.tolist(), W.tolist(),
            n, m, p, max_iters=20, deadline=deadline
        )

        # Project and verify
        U_p, V_p, W_p = _project_to_half_integers(U_opt, V_opt, W_opt)
        if _verify(U_p.tolist(), V_p.tolist(), W_p.tolist(), n, m, p):
            if _is_valid_entries(U_p.tolist(), V_p.tolist(), W_p.tolist()):
                return U_p.tolist(), V_p.tolist(), W_p.tolist()

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # Start with rank-55
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify baseline
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find rank 54 or 53 via ALS
    for target_rank in [54, 53]:
        if time.time() > deadline - 20:
            break

        # Try ALS from truncated rank-55
        result = _try_als_for_rank(n, m, p, target_rank, U_best, V_best, W_best, deadline)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = target_rank
            break

        # Try random search with ALS refinement
        if time.time() > deadline - 15:
            break
        result = _random_rank_search(n, m, p, target_rank, deadline, max_trials=50)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = target_rank
            break

    return {"U": U_best, "V": V_best, "W": W_best}
