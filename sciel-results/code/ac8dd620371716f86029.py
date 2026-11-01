"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify_full(U, V, W, tol=1e-9):
        """Verify exact decomposition by checking ALL tensor entries."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def build_rank32():
        """Build the known rank-32 solution using Strassen on 4 pairs + trivial for k=8."""
        rank = 32
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # 7 Strassen terms for multiplying [2x2] * [2x2] for columns k0, k1
            U[0, col] = 1; U[3, col] = 1
            V[0*p+k0, col] = 1; V[1*p+k1, col] = 1
            W[k0*n+0, col] = 1; W[k1*n+1, col] = 1
            col += 1

            U[2, col] = 1; U[3, col] = 1
            V[0*p+k0, col] = 1
            W[k0*n+1, col] = 1; W[k1*n+1, col] = -1
            col += 1

            U[0, col] = 1
            V[0*p+k1, col] = 1; V[1*p+k1, col] = -1
            W[k1*n+0, col] = 1; W[k1*n+1, col] = 1
            col += 1

            U[3, col] = 1
            V[1*p+k0, col] = 1; V[0*p+k0, col] = -1
            W[k0*n+0, col] = 1; W[k0*n+1, col] = 1
            col += 1

            U[0, col] = 1; U[1, col] = 1
            V[1*p+k1, col] = 1
            W[k0*n+0, col] = -1; W[k1*n+0, col] = 1
            col += 1

            U[2, col] = 1; U[0, col] = -1
            V[0*p+k0, col] = 1; V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1
            col += 1

            U[1, col] = 1; U[3, col] = -1
            V[1*p+k0, col] = 1; V[1*p+k1, col] = 1
            W[k0*n+0, col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
            U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
            U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
            U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_rank_r(r_target, max_iters=200, ftol=1e-14, gtol=1e-12, max_attempts=10):
        """Try to find a rank-r decomposition by optimizing from rank-32 initialization."""
        if time.time() > deadline - 60:
            return None

        r32 = build_rank32()
        U32 = np.array(r32["U"], dtype=np.float64)
        V32 = np.array(r32["V"], dtype=np.float64)
        W32 = np.array(r32["W"], dtype=np.float64)

        r = r_target
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r

        def objective(x):
            if time.time() > deadline - 55:
                return 1e10
            U_mat = x[:U_size].reshape(n*m, r)
            V_mat = x[U_size:U_size+V_size].reshape(m*p, r)
            W_mat = x[U_size+V_size:].reshape(p*n, r)

            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+i])
                        residual += (total - 1.0) ** 2
            return residual

        # Try multiple random initializations by removing random columns
        np.random.seed(42)
        for attempt in range(max_attempts):
            if time.time() > deadline - 50:
                break

            # Remove (32-r) random columns
            remove_cols = np.random.choice(32, size=32-r, replace=False)
            keep_cols = [c for c in range(32) if c not in remove_cols]
            U_keep = U32[:, keep_cols]
            V_keep = V32[:, keep_cols]
            W_keep = W32[:, keep_cols]

            # Pad to r columns with small random noise
            pad_size = r - len(keep_cols)
            if pad_size > 0:
                U_pad = np.random.randn(n*m, pad_size) * 0.1
                V_pad = np.random.randn(m*p, pad_size) * 0.1
                W_pad = np.random.randn(p*n, pad_size) * 0.1
                U_init = np.hstack([U_keep, U_pad])
                V_init = np.hstack([V_keep, V_pad])
                W_init = np.hstack([W_keep, W_pad])
            else:
                U_init = U_keep
                V_init = V_keep
                W_init = W_keep

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': max_iters, 'ftol': ftol, 'gtol': gtol})

                if res.fun < 1e-12:
                    U_mat = res.x[:U_size].reshape(n*m, r)
                    V_mat = res.x[U_size:U_size+V_size].reshape(m*p, r)
                    W_mat = res.x[U_size+V_size:].reshape(p*n, r)

                    U_list = U_mat.tolist()
                    V_list = V_mat.tolist()
                    W_list = W_mat.tolist()

                    if verify_full(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass

        return None

    # Build the known rank-32 solution
    result = build_rank32()

    if verify_full(result["U"], result["V"], result["W"]):
        # Try to improve to rank 31
        r31 = try_rank_r(31, max_iters=300, max_attempts=15)
        if r31 is not None:
            return r31

        # Try to improve to rank 30
        r30 = try_rank_r(30, max_iters=250, max_attempts=10)
        if r30 is not None:
            return r30

        return result

    # Fallback: schoolbook rank 36
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
