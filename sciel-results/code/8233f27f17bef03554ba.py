"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115  # keep safety margin below 120s

    # ------------------------------------------------------------------
    # Build an initial rank-36 schoolbook decomposition, then attempt
    # to reduce rank by exploiting additive structure and symmetry.
    # ------------------------------------------------------------------

    def schoolbook_rank():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U, V, W = schoolbook_rank()
    rank = n * m * p

    # ------------------------------------------------------------------
    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # ------------------------------------------------------------------
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    r = min(U.shape[1], V.shape[1], W.shape[1])
                    for c in range(r):
                        actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(actual - expected) > tol:
                        return False
        return True

    # Verify initial
    if not verify(U, V, W):
        # fallback: return schoolbook
        def to_list(mat):
            return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # ------------------------------------------------------------------
    # Try to reduce rank by finding zero columns or combining.
    # Strategy: iterative random search for low-rank decompositions using
    # ALS-like optimization with exact constraint enforcement.
    # ------------------------------------------------------------------

    def try_reduce_rank(U, V, W, max_rank, iterations=2000, seed=42):
        """Attempt to find a valid decomposition with rank <= max_rank."""
        rng = np.random.RandomState(seed)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]

        # Try each target rank from max_rank down to some lower bound
        for target_rank in range(max_rank, max(15, best_rank - 20) - 1, -1):
            if time.time() > deadline - 5:
                break
            # Initialize with random projections of current U, V, W
            for init_attempt in range(3):
                U_t = rng.randn(n * m, target_rank)
                V_t = rng.randn(m * p, target_rank)
                W_t = rng.randn(p * n, target_rank)
                # Normalize
                U_t /= (np.linalg.norm(U_t, axis=1, keepdims=True) + 1e-12)
                V_t /= (np.linalg.norm(V_t, axis=1, keepdims=True) + 1e-12)
                W_t /= (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)

                # ALS optimization
                for it in range(iterations):
                    if time.time() > deadline - 5:
                        break
                    # Update W: for each (k,i), want sum_c W[k*n+i,c]*U[i*m+j,c]*V[j*p+k,c] = 1
                    # Group by (k,i): for each j, W_ki * (U_ij * V_jk) = 1
                    # This is a system: for fixed (k,i), we have m equations (one per j)
                    # W_ki · M_j = 1 for all j, where M_j[c] = U[i*m+j,c]*V[j*p+k,c]
                    # So we want W_ki to be a vector such that dot(M_j, W_ki) = 1 for all j
                    # Solve least squares: minimize ||M^T W_ki - 1||^2
                    # M is m x target_rank, solve M^T M w = M^T 1
                    solved_all = True
                    for k in range(p):
                        for i in range(n):
                            M = np.zeros((m, target_rank))
                            for j in range(m):
                                M[j, :] = U_t[i * m + j, :] * V_t[j * p + k, :]
                            # Solve M^T M w = M^T 1
                            AtA = M.T @ M
                            At1 = M.T @ np.ones(m)
                            try:
                                W_t[k * n + i, :] = np.linalg.solve(AtA + 1e-8 * np.eye(target_rank), At1)
                            except np.linalg.LinAlgError:
                                W_t[k * n + i, :] = np.linalg.lstsq(AtA + 1e-8 * np.eye(target_rank), At1, rcond=None)[0]
                            # Verify
                            for j in range(m):
                                val = np.dot(U_t[i * m + j, :], V_t[j * p + k, :] * W_t[k * n + i, :])
                                if abs(val - 1) > 1e-8:
                                    solved_all = False
                                    break
                        if time.time() > deadline - 5:
                            break
                    if not solved_all or time.time() > deadline - 5:
                        # Try updating U and V similarly
                        break

                    # Update U: for fixed (i,j), want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                    # For fixed (i,j): N_k[c] = V[j*p+k,c]*W[k*n+i,c], want U_ij · N_k = 1 for all k
                    solved_all = True
                    for i in range(n):
                        for j in range(m):
                            N = np.zeros((p, target_rank))
                            for k in range(p):
                                N[k, :] = V_t[j * p + k, :] * W_t[k * n + i, :]
                            AtA = N.T @ N
                            At1 = N.T @ np.ones(p)
                            try:
                                U_t[i * m + j, :] = np.linalg.solve(AtA + 1e-8 * np.eye(target_rank), At1)
                            except np.linalg.LinAlgError:
                                U_t[i * m + j, :] = np.linalg.lstsq(AtA + 1e-8 * np.eye(target_rank), At1, rcond=None)[0]
                            for k in range(p):
                                val = np.dot(U_t[i * m + j, :], V_t[j * p + k, :] * W_t[k * n + i, :])
                                if abs(val - 1) > 1e-8:
                                    solved_all = False
                                    break
                        if time.time() > deadline - 5:
                            break
                    if not solved_all or time.time() > deadline - 5:
                        break

                    # Update V: for fixed (j,k), want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
                    solved_all = True
                    for j in range(m):
                        for k in range(p):
                            P = np.zeros((n, target_rank))
                            for i in range(n):
                                P[i, :] = U_t[i * m + j, :] * W_t[k * n + i, :]
                            AtA = P.T @ P
                            At1 = P.T @ np.ones(n)
                            try:
                                V_t[j * p + k, :] = np.linalg.solve(AtA + 1e-8 * np.eye(target_rank), At1)
                            except np.linalg.LinAlgError:
                                V_t[j * p + k, :] = np.linalg.lstsq(AtA + 1e-8 * np.eye(target_rank), At1, rcond=None)[0]
                            for i in range(n):
                                val = np.dot(U_t[i * m + j, :], V_t[j * p + k, :] * W_t[k * n + i, :])
                                if abs(val - 1) > 1e-8:
                                    solved_all = False
                                    break
                        if time.time() > deadline - 5:
                            break

                    if solved_all:
                        # Converged
                        best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                        best_rank = target_rank
                        break

        return best_U, best_V, best_W, best_rank

    # Try rank reduction
    new_U, new_V, new_W, new_rank = try_reduce_rank(U, V, W, rank)
    if new_rank < rank:
        U, V, W = new_U, new_V, new_W
        rank = new_rank

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------
    if verify(U, V, W):
        def to_list(mat):
            return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook_rank()
        def to_list(mat):
            return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
