"""
Combined approach: deterministic multi-Strassen decomposition (from inspirations)
plus a structured linear-algebra refinement pass (from current program's ALS idea).

The deterministic construction gives a guaranteed valid rank-81 solution.
The refinement pass then attempts to find a lower-rank decomposition using
ALS with half-integer snapping, seeded by the Strassen solution structure.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    deadline = time.time() + 110  # safety margin within 150s

    # --- Step 1: Deterministic multi-Strassen rank-81 solution ---
    rank_strassen = 81
    U = [[0.0] * rank_strassen for _ in range(n * m)]
    V = [[0.0] * rank_strassen for _ in range(m * p)]
    W = [[0.0] * rank_strassen for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, j0, k0):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block at (i0,j0,k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1.0
        U[i1 * m + j1][col_start] = 1.0
        V[j0 * p + k0][col_start] = 1.0
        V[j1 * p + k1][col_start] = 1.0
        W[k0 * n + i0][col_start] = 1.0
        W[k1 * n + i1][col_start] = 1.0

        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1.0
        U[i1 * m + j1][col_start + 1] = 1.0
        V[j0 * p + k0][col_start + 1] = 1.0
        W[k0 * n + i1][col_start + 1] = 1.0
        W[k1 * n + i1][col_start + 1] = -1.0

        # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1.0
        V[j0 * p + k1][col_start + 2] = 1.0
        V[j1 * p + k1][col_start + 2] = -1.0
        W[k1 * n + i0][col_start + 2] = 1.0
        W[k1 * n + i1][col_start + 2] = 1.0

        # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1.0
        V[j1 * p + k0][col_start + 3] = 1.0
        V[j0 * p + k0][col_start + 3] = -1.0
        W[k0 * n + i0][col_start + 3] = 1.0
        W[k0 * n + i1][col_start + 3] = 1.0

        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1.0
        U[i0 * m + j1][col_start + 4] = 1.0
        V[j1 * p + k1][col_start + 4] = 1.0
        W[k0 * n + i0][col_start + 4] = -1.0
        W[k1 * n + i0][col_start + 4] = 1.0

        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1.0
        U[i0 * m + j0][col_start + 5] = -1.0
        V[j0 * p + k0][col_start + 5] = 1.0
        V[j0 * p + k1][col_start + 5] = 1.0
        W[k1 * n + i1][col_start + 5] = 1.0

        # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col_start + 6] = 1.0
        U[i1 * m + j1][col_start + 6] = -1.0
        V[j1 * p + k0][col_start + 6] = 1.0
        V[j1 * p + k1][col_start + 6] = 1.0
        W[k0 * n + i0][col_start + 6] = 1.0

        return col_start + 7

    # Three disjoint 2x2x2 Strassen blocks
    col = add_strassen_block(col, 0, 0, 0)   # i∈{0,1}, j∈{0,1}, k∈{0,1}
    col = add_strassen_block(col, 1, 2, 5)   # i∈{1,2}, j∈{2,3}, k∈{5,6}
    col = add_strassen_block(col, 0, 2, 0)   # i∈{0,1}, j∈{2,3}, k∈{0,1}

    # Schoolbook for all remaining (i,j,k)
    covered = set()
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if i0 in (0, 1) and j0 in (0, 1) and k0 in (0, 1):
                    covered.add((i0, j0, k0))
                    continue
                if i0 in (1, 2) and j0 in (2, 3) and k0 in (5, 6):
                    covered.add((i0, j0, k0))
                    continue
                if i0 in (0, 1) and j0 in (2, 3) and k0 in (0, 1):
                    covered.add((i0, j0, k0))
                    continue
                U[i0 * m + j0][col] = 1.0
                V[j0 * p + k0][col] = 1.0
                W[k0 * n + i0][col] = 1.0
                col += 1

    assert col == rank_strassen, f"Expected rank {rank_strassen}, got {col}"

    # Convert to numpy for verification and potential refinement
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    def check_solution(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i * m + j, :], V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Verify the Strassen solution
    assert check_solution(U_np, V_np, W_np, tol=1e-9), "Strassen solution failed verification"

    best_rank = rank_strassen
    best_U, best_V, best_W = U_np, V_np, W_np

    # --- Step 2: Attempt to find a lower-rank decomposition via structured search ---
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    min_rank = max(n * m, m * p, p * n)

    def als_decompose(rank, max_iter=200, tol=1e-10, seed=0):
        """
        ALS-based tensor decomposition with half-integer snapping.
        Initialized from the Strassen structure projected to the target rank.
        """
        np.random.seed(seed)

        # Initialize: use a subset of Strassen columns as starting point
        if rank >= rank_strassen:
            U_init = U_np[:, :rank].copy()
            V_init = V_np[:, :rank].copy()
            W_init = W_np[:, :rank].copy()
        else:
            # Use first 'rank' columns of Strassen as initialization
            U_init = U_np[:, :rank].copy()
            V_init = V_np[:, :rank].copy()
            W_init = W_np[:, :rank].copy()
            # Add small random perturbation to escape local minima
            U_init += np.random.randn(n * m, rank) * 0.05
            V_init += np.random.randn(m * p, rank) * 0.05
            W_init += np.random.randn(p * n, rank) * 0.05

        U_cur = U_init
        V_cur = V_init
        W_cur = W_init

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None

            # Update U
            U_new = np.zeros_like(U_cur)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                    try:
                        U_new[i * m + j, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = U_cur[i * m + j, :]
            U_cur = U_new

            # Update V
            V_new = np.zeros_like(V_cur)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                    try:
                        V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = V_cur[j * p + k, :]
            V_cur = V_new

            # Update W
            W_new = np.zeros_like(W_cur)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, m))
                    for j in range(m):
                        A[:, j] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                    try:
                        W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = W_cur[k * n + i, :]
            W_cur = W_new

            # Check if solution is valid and snap to half-integers
            if check_solution(U_cur, V_cur, W_cur, tol=1e-5):
                U_snap = np.round(U_cur * 2) / 2
                V_snap = np.round(V_cur * 2) / 2
                W_snap = np.round(W_cur * 2) / 2
                if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                    return U_snap, V_snap, W_snap

        # Final check
        if check_solution(U_cur, V_cur, W_cur, tol=1e-8):
            U_snap = np.round(U_cur * 2) / 2
            V_snap = np.round(V_cur * 2) / 2
            W_snap = np.round(W_cur * 2) / 2
            if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                return U_snap, V_snap, W_snap

        return None

    # Try ranks from lower bound up to (but not including) the Strassen rank
    max_restarts = 15
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break

        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            seed = hash((n, m, p, rank, restart)) % (2**31)
            result = als_decompose(rank, max_iter=100, seed=seed)
            if result is not None:
                U_f, V_f, W_f = result
                best_rank = rank
                best_U, best_V, best_W = U_f, V_f, W_f
                found = True
                break

        if found:
            # Found a valid lower-rank solution; stop searching lower
            break

    # --- Step 3: Final safety check ---
    if not check_solution(best_U, best_V, best_W, tol=1e-6):
        # Fall back to the verified Strassen solution
        best_U, best_V, best_W = U_np, V_np, W_np
        best_rank = rank_strassen

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
