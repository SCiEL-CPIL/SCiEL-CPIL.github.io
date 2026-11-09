"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank(r):
        """Try to find rank-r decomposition using alternating least squares."""
        if time.time() > deadline:
            return None

        for attempt in range(100):
            if time.time() > deadline:
                return None

            np.random.seed(attempt * 7919 + 13)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5

            for it in range(2000):
                if time.time() > deadline:
                    return None

                # Update W
                W_new = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        if r >= m:
                            W_new[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        else:
                            W_new[k*n+i, :] = np.linalg.pinv(A) @ np.ones(m)
                W = W_new

                # Update U
                U_new = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        if r >= p:
                            U_new[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        else:
                            U_new[i*m+j, :] = np.linalg.pinv(A) @ np.ones(p)
                U = U_new

                # Update V
                V_new = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        if r >= n:
                            V_new[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        else:
                            V_new[j*p+k, :] = np.linalg.pinv(A) @ np.ones(n)
                V = V_new

                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-10:
                    break

            if verify(U, V, W):
                for scale in [2, 4, 1]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    if verify(U_r, V_r, W_r, tol=1e-9):
                        return (U_r, V_r, W_r)

        return None

    # Try ranks from 24 to 29
    for r in range(24, 30):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U_sb[i*m+j, c] = 1
                V_sb[j*p+k, c] = 1
                W_sb[k*n+i, c] = 1

    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
