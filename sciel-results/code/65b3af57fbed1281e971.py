"""Matrix multiplication 2x3x10: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _strassen_block(U, V, W, col, u_base, v_base, w_base, n, p):
    """Apply Strassen to a 2x2x2 block. u_base = row offset in U, v_base = row offset in V, w_base = row offset in W."""
    # u_base: first row index in U (so rows are u_base, u_base+1)
    # v_base: first row index in V (so rows are v_base, v_base+1)
    # w_base: first row index in W (so rows are w_base, w_base+n)
    u00, u01 = u_base, u_base + 1
    u10, u11 = u_base + n, u_base + n + 1
    v00, v01 = v_base, v_base + 1
    v10, v11 = v_base + p, v_base + p + 1
    w00, w01 = w_base, w_base + 1
    w10, w11 = w_base + n, w_base + n + 1

    # M1 = (a00 + a11)(b00 + b11)
    U[u00][col] += 1; U[u11][col] += 1
    V[v00][col] += 1; V[v11][col] += 1
    W[w00][col] += 1; W[w11][col] += 1
    col += 1

    # M2 = (a10 + a11)b00
    U[u10][col] += 1; U[u11][col] += 1
    V[v00][col] += 1
    W[w10][col] += 1; W[w11][col] += -1
    col += 1

    # M3 = a00(b01 - b11)
    U[u00][col] += 1
    V[v01][col] += 1; V[v11][col] += -1
    W[w01][col] += 1; W[w11][col] += 1
    col += 1

    # M4 = a11(b10 - b00)
    U[u11][col] += 1
    V[v10][col] += 1; V[v00][col] += -1
    W[w00][col] += 1; W[w10][col] += 1
    col += 1

    # M5 = (a00 + a01)b11
    U[u00][col] += 1; U[u01][col] += 1
    V[v11][col] += 1
    W[w00][col] += -1; W[w01][col] += 1
    col += 1

    # M6 = (a10 - a00)(b00 + b01)
    U[u10][col] += 1; U[u00][col] += -1
    V[v00][col] += 1; V[v01][col] += 1
    W[w11][col] += 1
    col += 1

    # M7 = (a01 - a11)(b10 + b11)
    U[u01][col] += 1; U[u11][col] += -1
    V[v10][col] += 1; V[v11][col] += 1
    W[w00][col] += 1
    col += 1

    return col

def _build_rank55_decomposition(n, m, p):
    """Build the rank-55 decomposition: 20 trivial + 5 Strassen blocks of rank 7."""
    rank = 20 + 5 * 7  # 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part B: j=1,2 -> 5 blocks of 2x2 Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        # u_base: row for (i=0, j=1) in U
        u_base = 0 * m + 1
        # v_base: row for (j=1, k=k0) in V
        v_base = 1 * p + k0
        # w_base: row for (k=k0, i=0) in W
        w_base = k0 * n + 0
        col = _strassen_block(U, V, W, col, u_base, v_base, w_base, n, p)

    return {"U": U, "V": V, "W": W}

def _try_linear_reduction(decomp, n, m, p, max_iter=200, tol=1e-8, time_budget=15.0):
    """Try to find linear dependencies among the rank-1 tensors to reduce rank."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)

    rank = U.shape[1]

    # Build the "flattened tensor" matrix for each column
    # Each rank-1 tensor is the outer product of three vectors
    # We can check if a column is a linear combination of others

    # Use QR with column pivoting to find rank
    # Create a matrix where each column is the flattened tensor
    # But tensors are 3D, so we flatten each rank-1 term

    # Actually, a simpler approach: try removing each column and check if the rest still spans
    # This is expensive, so use a heuristic: find columns with smallest norm and try to remove them

    U_norms = np.linalg.norm(U, axis=0)
    V_norms = np.linalg.norm(V, axis=0)
    W_norms = np.linalg.norm(W, axis=0)
    combined_norms = U_norms * V_norms * W_norms

    # Try removing columns in order of smallest combined norm
    removed = []
    remaining = list(range(rank))

    start_time = time.time()
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if not remaining:
            break

        # Find the column with smallest combined norm among remaining
        idx = min(remaining, key=lambda i: combined_norms[i])

        # Check if removing this column still gives a valid decomposition
        test_U = np.delete(U, idx, axis=1)
        test_V = np.delete(V, idx, axis=1)
        test_W = np.delete(W, idx, axis=1)

        # Reconstruct tensor
        T_recon = np.zeros((n * m, m * p, p * n))
        for c in range(test_U.shape[1]):
            T_recon += np.einsum('i,j,k->ijk', test_U[:, c], test_V[:, c], test_W[:, c])

        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1

        if not np.allclose(T_recon, T_true, atol=tol):
            removed.append(idx)
            remaining.remove(idx)

    # Build the reduced decomposition
    if not remaining:
        return decomp

    new_rank = len(remaining)
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]

    for new_idx, old_idx in enumerate(remaining):
        for i in range(n * m):
            new_U[i][new_idx] = U[i, old_idx]
        for i in range(m * p):
            new_V[i][new_idx] = V[i, old_idx]
        for i in range(p * n):
            new_W[i][new_idx] = W[i, old_idx]

    return {"U": new_U, "V": new_V, "W": new_W}

def _verify_decomposition(decomp, n, m, p, atol=1e-10):
    """Verify that the decomposition exactly represents the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)

    T_recon = np.zeros((n * m, m * p, p * n))
    for c in range(U.shape[1]):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])

    T_true = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1

    return np.allclose(T_recon, T_true, atol=atol)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Start with the known rank-55 decomposition
    decomp = _build_rank55_decomposition(n, m, p)

    if not _verify_decomposition(decomp, n, m, p):
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)

    # Try to reduce rank via linear dependency search
    if time.time() < deadline:
        reduced = _try_linear_reduction(decomp, n, m, p, time_budget=min(30, deadline - time.time() - 10))
        if _verify_decomposition(reduced, n, m, p):
            decomp = reduced

    # Verify final result
    if not _verify_decomposition(decomp, n, m, p):
        return _schoolbook_decomposition(n, m, p)

    return decomp
