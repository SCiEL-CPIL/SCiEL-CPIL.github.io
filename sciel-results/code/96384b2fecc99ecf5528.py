# EVOLVE-BLOCK-START

def _build_tensor_indices(n, m, p):
    """Return list of (u_idx, v_idx, w_idx) for each (i,j,k)."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    return triples

def _solve_linear_block(A, b):
    """Solve A x = b for min-norm x. A is (M, R), b is (M,)."""
    # Use lstsq for min-norm solution
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x

def _exact_rank_solution(n, m, p, rank, rng):
    """Attempt to find an exact integer decomposition of given rank.
    Returns (U,V,W) as numpy int arrays or None.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    max_iters = 500
    deadline = time.time() + 10.0  # strict internal deadline for one rank attempt

    for attempt in range(max_iters):
        if time.time() >= deadline:
            break

        # Random initialisation with small integer entries
        U = rng.integers(-1, 2, size=(N, rank))
        V = rng.integers(-1, 2, size=(P, rank))
        W = rng.integers(-1, 2, size=(Q, rank))

        for _iter in range(100):
            if time.time() >= deadline:
                break

            # 1. Fix V, W. Solve for U.
            # For each (i, j), we have equations for k=0..p-1:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A[i,j] be a p x rank matrix where A[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            # We solve A[i,j] @ U[i*m+j, :] = 1
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    u_sol = _solve_linear_block(A, b)
                    # Round to nearest integer
                    U_new[i * m + j, :] = np.round(u_sol)
            
            U = U_new

            # 2. Fix U, W. Solve for V.
            # For each (j, k), we have equations for i=0..n-1:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A[j,k] be an n x rank matrix where A[j,k][i, r] = U[i*m+j, r] * W[k*n+i, r]
            # We solve A[j,k] @ V[j*p+k, :] = 1
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    v_sol = _solve_linear_block(A, b)
                    # Round to nearest integer
                    V_new[j * p + k, :] = np.round(v_sol)
            
            V = V_new

            # 3. Fix U, V. Solve for W.
            # For each (k, i), we have equations for j=0..m-1:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A[k,i] be an m x rank matrix where A[k,i][j, r] = U[i*m+j, r] * V[j*p+k, r]
            # We solve A[k,i] @ W[k*n+i, :] = 1
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    w_sol = _solve_linear_block(A, b)
                    # Round to nearest integer
                    W_new[k * n + i, :] = np.round(w_sol)
            
            W = W_new

            # Check exactness
            R = np.ones((n, m, p), dtype=float)
            for r in range(rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                for i in range(n):
                    for j in range(m):
                        u_val = u_col[i * m + j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            R[i, j, k] -= u_val * v_col[j * p + k] * w_col[k * n + i]
            
            if np.max(np.abs(R)) < 1e-9:
                return (U.astype(int), V.astype(int), W.astype(int))

    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition, searching for lower
    ranks with a hard wall-clock deadline."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # 10s safety margin

    rng = np.random.default_rng(12345)

    # Try ranks from 44 down to 41
    for rank in [44, 43, 42, 41]:
        if time.time() >= deadline:
            break
        sol = _exact_rank_solution(n, m, p, rank, rng)
        if sol is not None:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
            }

    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
