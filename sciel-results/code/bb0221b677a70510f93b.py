import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin

    n, m, p = DIMENSIONS

    # Build the target tensor T
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0

    T_flat = T.flatten()

    def compute_residual(params, R):
        """Compute residual for given parameters and rank."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R

        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]

        U = U_flat.reshape((n * m, R)).T  # (R, n*m)
        V = V_flat.reshape((m * p, R)).T  # (R, m*p)
        W = W_flat.reshape((p * n, R)).T  # (R, p*n)

        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))

        recon = np.zeros_like(T)
        for r in range(R):
            u = U_reshaped[r]  # (n, m)
            v = V_reshaped[r]  # (m, p)
            w = W_reshaped[r]  # (p, n)

            uv = np.einsum('ij,jk->ijjk', u, v)
            recon += np.einsum('ijjk,ki->ijkiki', uv, w)

        return (recon.flatten() - T_flat)

    def try_rank(R, n_restarts=8, max_nfev=150):
        """Try to find a valid decomposition at rank R."""
        total_dim = n * m * R + m * p * R + p * n * R

        # Multiple restarts with different initializations
        for attempt in range(n_restarts):
            if time.time() > deadline:
                return None

            # Initialize with small random values
            if attempt % 2 == 0:
                x0 = np.random.randn(total_dim) * 0.3
            else:
                # Try structured initialization
                x0 = np.zeros(total_dim)
                # Add some non-zero entries
                idx = np.random.choice(total_dim, size=min(50, total_dim), replace=False)
                x0[idx] = np.random.randn(len(idx)) * 0.5

            try:
                res = least_squares(
                    compute_residual, x0, args=(R,),
                    method='lm', max_nfev=max_nfev
                )
            except Exception:
                continue

            if res.cost < 1e-10:
                # Check if we can snap to half-integers
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R

                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]

                # Try snapping to nearest half-integer
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2

                x_snapped = np.concatenate([U_snap, V_snap, W_snap])
                res_snapped = compute_residual(x_snapped, R)

                if np.max(np.abs(res_snapped)) < 1e-9:
                    return {
                        "U": U_snap.reshape((n * m, R)).tolist(),
                        "V": V_snap.reshape((m * p, R)).tolist(),
                        "W": W_snap.reshape((p * n, R)).tolist()
                    }

        return None

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 60 to 83, starting from lower ranks
    for R in range(60, min(84, n * m * p)):
        if time.time() > deadline:
            break

        # Use more restarts for lower ranks (harder to find)
        n_restarts = 12 if R <= 65 else 8
        max_nfev = 200 if R <= 65 else 150

        sol = try_rank(R, n_restarts=n_restarts, max_nfev=max_nfev)
        if sol is not None:
            best_rank = R
            best_solution = sol
            break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook decomposition
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
