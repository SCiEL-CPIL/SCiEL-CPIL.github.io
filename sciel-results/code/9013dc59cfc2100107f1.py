import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0

    # Schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        # Vectorized verification using einsum
        # res[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(res, 1.0, atol=1e-9)

    def _compute_residuals_vec(x, n, m, p, r):
        """Vectorized residual computation."""
        U_flat = x[:n*m*r]
        V_flat = x[n*m*r : n*m*r + m*p*r]
        W_flat = x[n*m*r + m*p*r :]

        U_mat = U_flat.reshape(n*m, r)
        V_mat = V_flat.reshape(m*p, r)
        W_mat = W_flat.reshape(p*n, r)

        U_reshaped = U_mat.reshape(n, m, r)
        V_reshaped = V_mat.reshape(m, p, r)
        W_reshaped = W_mat.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return (res - 1.0).flatten()

    def _try_rank(r, max_trials=8):
        """Try to find a rank-r decomposition."""
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False

        for trial in range(max_trials):
            if time.time() > deadline - 15:
                return False

            # Different initialization strategies
            if trial == 0:
                # Structured: use small integers
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for t in range(r):
                    for idx in range(n*m):
                        U[idx, t] = float(((idx + t) % 3) - 1)
                    for idx in range(m*p):
                        V[idx, t] = float(((idx + t) % 3) - 1)
                    for idx in range(p*n):
                        W[idx, t] = float(((idx + t) % 3) - 1)
            elif trial == 1:
                np.random.seed(42 + r * 10 + trial)
                U = np.random.randint(-2, 3, size=(n*m, r)).astype(float) * 0.5
                V = np.random.randint(-2, 3, size=(m*p, r)).astype(float) * 0.5
                W = np.random.randint(-2, 3, size=(p*n, r)).astype(float) * 0.5
            elif trial == 2:
                np.random.seed(123 + r * 10 + trial)
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            elif trial == 3:
                np.random.seed(456 + r * 10 + trial)
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            elif trial == 4:
                # Start from schoolbook-like structure
                U = U_sb[:, :r].copy()
                V = V_sb[:, :r].copy()
                W = W_sb[:, :r].copy()
                U += np.random.randn(n*m, r) * 0.1
                V += np.random.randn(m*p, r) * 0.1
                W += np.random.randn(p*n, r) * 0.1
            elif trial == 5:
                # Binary-like initialization
                np.random.seed(789 + r * 10 + trial)
                U = (np.random.rand(n*m, r) > 0.5).astype(float) * 2 - 1
                V = (np.random.rand(m*p, r) > 0.5).astype(float) * 2 - 1
                W = (np.random.rand(p*n, r) > 0.5).astype(float) * 2 - 1
            elif trial == 6:
                # Sparse initialization
                np.random.seed(321 + r * 10 + trial)
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for t in range(r):
                    nz_U = np.random.choice(n*m, size=min(3, n*m), replace=False)
                    nz_V = np.random.choice(m*p, size=min(3, m*p), replace=False)
                    nz_W = np.random.choice(p*n, size=min(3, p*n), replace=False)
                    U[nz_U, t] = np.random.choice([-1, 1], size=len(nz_U))
                    V[nz_V, t] = np.random.choice([-1, 1], size=len(nz_V))
                    W[nz_W, t] = np.random.choice([-1, 1], size=len(nz_W))
            else:
                # Random with scaling
                np.random.seed(654 + r * 10 + trial)
                scale = 1.0 / np.sqrt(r)
                U = np.random.randn(n*m, r) * scale
                V = np.random.randn(m*p, r) * scale
                W = np.random.randn(p*n, r) * scale

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def residuals(x):
                return _compute_residuals_vec(x, n, m, p, r)

            try:
                # First pass: moderate tolerance
                sol = least_squares(residuals, x0, method='lm', max_nfev=800)

                if sol.cost < 1e-6:
                    x_sol = sol.x
                    U_sol = x_sol[:n*m*r].reshape(n*m, r)
                    V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)

                    # Polish step: refine with tighter tolerance
                    if time.time() < deadline - 5:
                        sol2 = least_squares(residuals, x_sol, method='lm', max_nfev=400, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                        if sol2.cost < sol.cost:
                            x_sol = sol2.x
                            U_sol = x_sol[:n*m*r].reshape(n*m, r)
                            V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                            W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)

                    # Try rounding to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2

                    if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                        best_rank = r
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        return True

                    # Also try rounding to integers
                    U_round_i = np.round(U_sol)
                    V_round_i = np.round(V_sol)
                    W_round_i = np.round(W_sol)
                    if _verify_decomposition(U_round_i, V_round_i, W_round_i, n, m, p):
                        best_rank = r
                        best_U = U_round_i
                        best_V = V_round_i
                        best_W = W_round_i
                        return True

                    # Check if unrounded solution is valid (should be exact if cost is very small)
                    if sol.cost < 1e-14:
                        if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                            # Try to find a rational representation
                            for denom in [2, 4, 8]:
                                U_r = np.round(U_sol * denom) / denom
                                V_r = np.round(V_sol * denom) / denom
                                W_r = np.round(W_sol * denom) / denom
                                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                                    best_rank = r
                                    best_U = U_r
                                    best_V = V_r
                                    best_W = W_r
                                    return True
            except Exception:
                pass

        return False

    # Search for lower rank: start from realistic lower bounds
    # For 4x4 x 4x5, known results suggest rank around 40-56
    # Try from lower to higher, but skip very low ranks
    search_ranks = list(range(38, min(60, rank_school)))
    # Also try some specific values
    for r in search_ranks:
        if time.time() > deadline - 20:
            break
        if _try_rank(r, max_trials=8):
            break

    # If we haven't found anything better, try a few more ranks with fewer trials
    if best_rank == rank_school:
        for r in range(35, 38):
            if time.time() > deadline - 10:
                break
            if _try_rank(r, max_trials=4):
                break

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
