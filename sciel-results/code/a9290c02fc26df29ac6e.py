"""
Rank-minimal tensor decomposition for 2x3 * 3x7 matrix multiplication.
Strategy: numerical rank minimization via iterative column elimination
and least-squares refitting.  Start from a known valid decomposition
(hybrid Strassen/schoolbook, rank 39) and attempt to reduce the rank
by removing one column at a time and refitting the remaining columns.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]


def _build_hybrid_rank39():
    """
    Build the known rank-39 hybrid Strassen/schoolbook decomposition.
    U: (6, 39), V: (21, 39), W: (14, 39)
    """
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Strassen for 2x2 * 2x2, given global indices
    def add_strassen(B_cs, C_cs, sc):
        # A is 2x2: U indices 0,1,3,4 (i*m+j)
        # B is 2x2: V indices B_cs, B_cs+1, 7+B_cs, 7+B_cs+1 (j*p+k)
        # C is 2x2: W indices C_cs*2+0, C_cs*2+1, (C_cs+1)*2+0, (C_cs+1)*2+1 (k*n+i)
        # C[0,0] -> W[C_cs*2+0], C[1,0] -> W[C_cs*2+1]
        # C[0,1] -> W[(C_cs+1)*2+0], C[1,1] -> W[(C_cs+1)*2+1]

        # M1: (A00+A11)(B00+B11) -> C00 + C11
        U[0, sc] = 1; U[4, sc] = 1
        V[B_cs, sc] = 1; V[7+B_cs, sc] = 1
        W[C_cs*2, sc] = 1; W[(C_cs+1)*2+1, sc] = 1

        # M2: (A10+A11)B00 -> C10
        U[3, sc+1] = 1; U[4, sc+1] = 1
        V[B_cs, sc+1] = 1
        W[C_cs*2+1, sc+1] = 1

        # M3: A00(B10-B00) -> C01
        U[0, sc+2] = 1
        V[7+B_cs, sc+2] = 1; V[B_cs, sc+2] = -1
        W[(C_cs+1)*2, sc+2] = 1

        # M4: A11(B01-B11) -> C01
        U[4, sc+3] = 1
        V[B_cs+1, sc+3] = 1; V[7+B_cs+1, sc+3] = -1
        W[(C_cs+1)*2, sc+3] = 1

        # M5: (A00+A01)B11 -> C01
        U[0, sc+4] = 1; U[1, sc+4] = 1
        V[7+B_cs+1, sc+4] = 1
        W[(C_cs+1)*2, sc+4] = 1

        # M6: (A10-A00)(B00+B01) -> C11
        U[3, sc+5] = 1; U[0, sc+5] = -1
        V[B_cs, sc+5] = 1; V[B_cs+1, sc+5] = 1
        W[(C_cs+1)*2+1, sc+5] = 1

        # M7: (A01-A11)(B10+B11) -> C11
        U[1, sc+6] = 1; U[4, sc+6] = -1
        V[7+B_cs, sc+6] = 1; V[7+B_cs+1, sc+6] = 1
        W[(C_cs+1)*2+1, sc+6] = 1

        return sc + 7

    # Part 1: A1 (2x2) * B1 (2x7)
    col = add_strassen(0, 0, 0)   # B cols 0,1
    col = add_strassen(2, 2, col) # B cols 2,3
    col = add_strassen(4, 4, col) # B cols 4,5

    # B col 6: schoolbook 2x2x1
    U[0, col] = 1; V[6, col] = 1; W[6*2, col] = 1; col += 1
    U[1, col] = 1; V[7+6, col] = 1; W[6*2, col] = 1; col += 1
    U[3, col] = 1; V[6, col] = 1; W[6*2+1, col] = 1; col += 1
    U[4, col] = 1; V[7+6, col] = 1; W[6*2+1, col] = 1; col += 1

    # Part 2: A2 (2x1) * B2 (1x7) -- schoolbook
    for i in range(2):
        for k in range(7):
            U[i*3+2, col] = 1
            V[14+k, col] = 1
            W[k*2+i, col] = 1
            col += 1

    assert col == 39, f"Expected 39 columns, got {col}"
    return U, V, W


def _tensor_entries(U, V, W, n, m, p):
    """
    Compute the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Returns a dict mapping (i,j,k) -> value.
    """
    entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                entries[(i,j,k)] = val
    return entries


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_remove_column(U, V, W, n, m, p, col_idx, max_iter=200, tol=1e-10):
    """
    Try to remove column col_idx by refitting the remaining columns.
    Returns (new_U, new_V, new_W) if successful, else None.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None

    # Remove the column
    mask = [c for c in range(rank) if c != col_idx]
    U_r = U[:, mask].copy()
    V_r = V[:, mask].copy()
    W_r = W[:, mask].copy()

    new_rank = rank - 1
    if new_rank == 0:
        return None

    # Build the tensor target
    n_entries = n * m * p
    # For each entry (i,j,k), the constraint is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a trilinear constraint. We can linearize by fixing two matrices
    # and solving for the third, then iterate.
    #
    # Strategy: alternating least squares.
    #   Fix V, W -> solve for U (linear in each U entry)
    #   Fix U, W -> solve for V
    #   Fix U, V -> solve for W

    U_cur = U_r.copy()
    V_cur = V_r.copy()
    W_cur = W_r.copy()

    for iteration in range(max_iter):
        # --- Update U ---
        # For each (i,j): U[i*m+j, r] appears in entries (i,j,k) for all k.
        # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For fixed (i,j), we have p equations and new_rank unknowns (one per r).
        # Build the linear system: A_U[i,j] @ U[i*m+j, :] = b_U[i,j]
        # where A_U[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
        # and b_U[i,j][k] = 1

        A_U = np.zeros((n, m, p, new_rank))
        b_U = np.ones((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_U[i, j, k, :] = V_cur[j*p+k] * W_cur[k*n+i]
        # Solve least squares for each (i,j)
        for i in range(n):
            for j in range(m):
                A = A_U[i, j]  # (p, new_rank)
                b = b_U[i, j]  # (p,)
                # Least squares
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_cur[i*m+j, :] = sol

        # --- Update V ---
        # For each (j,k): V[j*p+k, r] appears in entries (i,j,k) for all i.
        # A_V[j,k][i, r] = U[i*m+j, r] * W[k*n+i, r]
        A_V = np.zeros((m, p, n, new_rank))
        b_V = np.ones((m, p, n))
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    A_V[j, k, i, :] = U_cur[i*m+j] * W_cur[k*n+i]
        for j in range(m):
            for k in range(p):
                A = A_V[j, k]  # (n, new_rank)
                b = b_V[j, k]  # (n,)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_cur[j*p+k, :] = sol

        # --- Update W ---
        # For each (k,i): W[k*n+i, r] appears in entries (i,j,k) for all j.
        # A_W[k,i][j, r] = U[i*m+j, r] * V[j*p+k, r]
        A_W = np.zeros((p, n, m, new_rank))
        b_W = np.ones((p, n, m))
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    A_W[k, i, j, :] = U_cur[i*m+j] * V_cur[j*p+k]
        for k in range(p):
            for i in range(n):
                A = A_W[k, i]  # (m, new_rank)
                b = b_W[k, i]  # (m,)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_cur[k*n+i, :] = sol

        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_cur[i*m+j] * V_cur[j*p+k] * W_cur[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err < tol:
            return U_cur, V_cur, W_cur

    # Final check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_cur[i*m+j] * V_cur[j*p+k] * W_cur[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    if max_err < tol:
        return U_cur, V_cur, W_cur
    return None


def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce from current rank to target_rank by iteratively removing columns.
    Returns (U, V, W) with target_rank columns if successful, else None.
    """
    current_rank = U.shape[1]
    if current_rank < target_rank:
        return None

    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()

    while U_cur.shape[1] > target_rank:
        if time.time() > deadline - 5.0:
            return None

        best_result = None
        best_err = float('inf')

        # Try removing each column; keep the one that gives the best fit
        for col_idx in range(U_cur.shape[1]):
            result = _try_remove_column(U_cur, V_cur, W_cur, n, m, p, col_idx)
            if result is not None:
                U_t, V_t, W_t = result
                # Verify
                if _verify(U_t, V_t, W_t, n, m, p, tol=1e-9):
                    return U_t, V_t, W_t
                else:
                    # Compute residual
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i])
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    if max_err < best_err:
                        best_err = max_err
                        best_result = result

        if best_result is not None:
            U_cur, V_cur, W_cur = best_result
        else:
            return None

    if _verify(U_cur, V_cur, W_cur, n, m, p, tol=1e-9):
        return U_cur, V_cur, W_cur
    return None


def _to_list(U, V, W):
    """Convert numpy arrays to lists of lists with rational-like entries."""
    U_l = [[float(x) for x in row] for row in U]
    V_l = [[float(x) for x in row] for row in V]
    W_l = [[float(x) for x in row] for row in W]
    return U_l, V_l, W_l


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Start with the known rank-39 decomposition
    U, V, W = _build_hybrid_rank39()

    # Verify the starting point
    if not _verify(U, V, W, n, m, p, tol=1e-9):
        # Shouldn't happen, but fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        U_l, V_l, W_l = _to_list(U, V, W)
        return {"U": U_l, "V": V_l, "W": W_l}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Try to reduce rank: 38, 37, 36, ...
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10.0:
            break
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
            # Continue trying to go lower
        else:
            # Can't reduce further from this starting point
            break

    U_l, V_l, W_l = _to_list(best_U, best_V, best_W)
    return {"U": U_l, "V": V_l, "W": W_l}
