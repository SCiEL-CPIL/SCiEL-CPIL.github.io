"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find the minimal rank r such that this holds for all i,j,k.

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition.
        U, V, W are numpy arrays or lists.
        Returns True if valid, False otherwise.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def compute_loss(U_mat, V_mat, W_mat):
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_mat[i * m + j, :]
                    v = V_mat[j * p + k, :]
                    w = W_mat[k * n + i, :]
                    val = np.dot(u * v, w)
                    loss += (val - 1.0) ** 2
        return loss

    def als_step(U_mat, V_mat, W_mat, rank, iters=80, tol=1e-12):
        """
        Alternating least squares: fix two matrices, solve for the third.
        Returns (U_mat, V_mat, W_mat, loss).
        """
        for it in range(iters):
            if time.time() > deadline - 5:
                break

            # Solve for U: for each (i,j), U[i*m+j,:] is the least-squares
            # solution to (V[j*p+k,:] * W[k*n+i,:])^T * u = 1 for all k.
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    # Build the system: sum_k V[j*p+k,c] * W[k*n+i,c] * U[i*m+j,c] = 1
                    # This is a linear system in U[i*m+j,:]:
                    # M @ U[i*m+j,:] = 1 where M[c,c'] = sum_k V[j*p+k,c] * W[k*n+i,c']
                    M = np.zeros((rank, rank))
                    for k in range(p):
                        M += np.outer(V_mat[j * p + k, :], W_mat[k * n + i, :])
                    try:
                        U_new[i * m + j, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            U_mat = U_new

            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, rank))
                    for i in range(n):
                        M += np.outer(U_mat[i * m + j, :], W_mat[k * n + i, :])
                    try:
                        V_new[j * p + k, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            V_mat = V_new

            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, rank))
                    for j in range(m):
                        M += np.outer(U_mat[i * m + j, :], V_mat[j * p + k, :])
                    try:
                        W_new[k * n + i, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            W_mat = W_new

            loss = compute_loss(U_mat, V_mat, W_mat)
            if loss < tol:
                break

        loss = compute_loss(U_mat, V_mat, W_mat)
        return U_mat, V_mat, W_mat, loss

    def polish(U_mat, V_mat, W_mat, rank, iters=200):
        """
        High-precision polish: run ALS with tighter tolerance.
        """
        for it in range(iters):
            if time.time() > deadline - 3:
                break

            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, rank))
                    for k in range(p):
                        M += np.outer(V_mat[j * p + k, :], W_mat[k * n + i, :])
                    try:
                        U_new[i * m + j, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            U_mat = U_new

            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, rank))
                    for i in range(n):
                        M += np.outer(U_mat[i * m + j, :], W_mat[k * n + i, :])
                    try:
                        V_new[j * p + k, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            V_mat = V_new

            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, rank))
                    for j in range(m):
                        M += np.outer(U_mat[i * m + j, :], V_mat[j * p + k, :])
                    try:
                        W_new[k * n + i, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            W_mat = W_new

            loss = compute_loss(U_mat, V_mat, W_mat)
            if loss < 1e-14:
                break

        return U_mat, V_mat, W_mat

    def try_rank(rank, max_attempts=30):
        """
        Try to find a rank-`rank` decomposition.
        Uses structured initialization based on polynomial ring Z[x]/(x^3-1).
        """
        if time.time() > deadline - 10:
            return None

        # Structured initialization:
        # Group columns of B by index mod 3: B[:, 0], B[:, 3], B[:, 6] -> poly0
        # B[:, 1], B[:, 4], B[:, 7] -> poly1
        # B[:, 2], B[:, 5], B[:, 8] -> poly2
        # Similarly for A columns: A[:, 0], A[:, 1], A[:, 2]
        #
        # The idea: use 3 "polynomial" factors of degree < 3, giving 3*3 = 9
        # basis elements per (i,j) pair, but we only have n*m = 6 rows for U.
        # So we need rank = 6 * 3 = 18? Let's try rank 18.
        #
        # Actually, let's think differently. The tensor T_{i,j,k} has structure:
        # For fixed i, the map (j,k) -> T_{i,j,k} is a 3x9 matrix.
        # We can write T_{i,j,k} = sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # Let's try a structured initialization where:
        # - U[i*m+j, c] depends on (i, j, c mod something)
        # - V[j*p+k, c] depends on (j, k, c mod something)
        # - W[k*n+i, c] depends on (k, i, c mod something)

        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            # Random initialization with some structure
            np.random.seed(attempt * 1000 + rank)
            U_mat = np.random.randn(n * m, rank) * 0.5
            V_mat = np.random.randn(m * p, rank) * 0.5
            W_mat = np.random.randn(p * n, rank) * 0.5

            # Scale to make initial products closer to 1
            U_mat /= np.linalg.norm(U_mat, axis=0, keepdims=True) + 1e-10
            V_mat /= np.linalg.norm(V_mat, axis=0, keepdims=True) + 1e-10
            W_mat /= np.linalg.norm(W_mat, axis=0, keepdims=True) + 1e-10

            U_mat, V_mat, W_mat, loss = als_step(U_mat, V_mat, W_mat, rank, iters=100)

            if loss < 1e-6:
                # Polish
                U_mat, V_mat, W_mat = polish(U_mat, V_mat, W_mat, rank, iters=300)
                loss = compute_loss(U_mat, V_mat, W_mat)

                if loss < 1e-10:
                    U_snap = snap_to_half_int(U_mat.tolist())
                    V_snap = snap_to_half_int(V_mat.tolist())
                    W_snap = snap_to_half_int(W_mat.tolist())

                    if verify(U_snap, V_snap, W_snap):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}

                    # Try snapping with a small perturbation
                    for _ in range(5):
                        U_snap2 = snap_to_half_int(U_mat.tolist())
                        V_snap2 = snap_to_half_int(V_mat.tolist())
                        W_snap2 = snap_to_half_int(W_mat.tolist())
                        if verify(U_snap2, V_snap2, W_snap2):
                            return {"U": U_snap2, "V": V_snap2, "W": W_snap2}

        return None

    # Try ranks from lower bound upwards
    for rank in range(18, 55):
        if time.time() > deadline - 15:
            break

        attempts = max(3, 20 - (rank - 18) * 2)
        res = try_rank(rank, attempts)
        if res is not None:
            return res

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
