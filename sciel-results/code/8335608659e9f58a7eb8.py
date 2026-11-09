"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        r = U.shape[1]
        result = np.zeros_like(T)
        for c in range(r):
            result += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, T, atol=1e-8)

    def schoolbook():
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

    U0, V0, W0 = schoolbook()
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = n * m * p

    def als_solve(U, V, W, max_iters=80):
        """Alternating least squares for tensor CP decomposition."""
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Solve for W given U, V
            UtU = U.T @ U
            VtV = V.T @ V
            G = UtU * VtV
            UV = U[:, None, :] * V[None, :, :]
            RHS = np.einsum('abd,abc->dc', T, UV)
            try:
                W_new = np.linalg.solve(G, RHS.T).T
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(G, RHS.T, rcond=None)[0].T

            # Solve for V given U, W
            UtU = U.T @ U
            WtW = W_new.T @ W_new
            G2 = UtU * WtW
            RHS2 = np.einsum('abd,ac,dc->bc', T, U, W_new)
            try:
                V_new = np.linalg.solve(G2, RHS2.T).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(G2, RHS2.T, rcond=None)[0].T

            # Solve for U given V, W
            VtV = V_new.T @ V_new
            WtW = W_new.T @ W_new
            G3 = VtV * WtW
            RHS3 = np.einsum('abd,bc,dc->ac', T, V_new, W_new)
            try:
                U_new = np.linalg.solve(G3, RHS3.T).T
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(G3, RHS3.T, rcond=None)[0].T

            U, V, W = U_new, V_new, W_new

        return U, V, W

    # Search for lower rank decompositions
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 3:
            break

        found = False
        remaining = deadline - 3 - time.time()
        n_attempts = max(5, min(40, int(remaining / 1.5)))

        for attempt in range(n_attempts):
            if time.time() > deadline - 3:
                break

            # Diverse initializations
            if attempt % 4 == 0:
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            elif attempt % 4 == 1:
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
            elif attempt % 4 == 2:
                U = np.random.randn(n * m, target_rank) * 0.1
                V = np.random.randn(m * p, target_rank) * 0.1
                W = np.random.randn(p * n, target_rank) * 0.1
            else:
                # Sparse-like init
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for c in range(target_rank):
                    U[np.random.randint(0, n * m), c] = 1.0
                    V[np.random.randint(0, m * p), c] = 1.0
                    W[np.random.randint(0, p * n), c] = 1.0

            U, V, W = als_solve(U, V, W, max_iters=40)

            if verify(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break

        if not found:
            # No improvement at this rank; continue to next lower rank
            pass

    def to_list(mat):
        return mat.tolist()

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
