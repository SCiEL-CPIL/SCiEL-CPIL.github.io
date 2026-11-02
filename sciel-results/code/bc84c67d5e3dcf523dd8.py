import time
import random
import numpy as np
from fractions import Fraction

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification: check only the 105 non-zero tensor entries."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions for the 105 non-zero entries."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(len(U[u_idx])):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank_block_solve(r, num_restarts=10, seed_base=42):
        """
        Try to find rank-r decomposition using block-structured linear solve.
        Fix V and W randomly, then for each i, solve linear system for U block.
        """
        if time.time() > deadline:
            return None

        for restart in range(num_restarts):
            if time.time() > deadline:
                return None

            np.random.seed(seed_base + r * 100 + restart)
            random.seed(seed_base + r * 100 + restart)

            # Generate random V and W with small integer/half-integer entries
            # Use values from {-1, 0, 1} or {-2, -1, 0, 1, 2}
            scale = random.choice([1, 2])
            V = np.random.randint(-scale, scale + 1, size=(mp, r)).astype(np.float64)
            W = np.random.randint(-scale, scale + 1, size=(pn, r)).astype(np.float64)

            # For each i, solve for U[i*m:(i+1)*m, :]
            U = np.zeros((nm, r))
            solvable = True

            for i in range(n):
                # Build linear system for this i
                # Unknowns: U[i*m+j, c] for j in 0..m-1, c in 0..r-1
                # That's m*r unknowns
                # Equations: for each (j, k) in 0..m-1, 0..p-1:
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # That's m*p equations

                A_mat = np.zeros((m * p, m * r))
                b_vec = np.ones(m * p)

                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        prod = v_row * w_row  # length r
                        for c in range(r):
                            A_mat[eq_idx, j * r + c] = prod[c]

                # Solve least squares
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    solvable = False
                    break

                U[i * m:(i + 1) * m, :] = sol.reshape(m, r)

            if not solvable:
                continue

            # Round U, V, W to half-integers
            U_r = np.round(U * 2.0) / 2.0
            V_r = np.round(V * 2.0) / 2.0
            W_r = np.round(W * 2.0) / 2.0

            # Check if rounding introduced significant error
            max_err = max(np.max(np.abs(U - U_r)), np.max(np.abs(V - V_r)), np.max(np.abs(W - W_r)))
            if max_err > 0.01:
                continue

            # Verify
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                # Convert to fractions for exact check
                U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
                V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
                W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]

                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    def try_rank_als(r, num_iters=15, num_restarts=5):
        """
        Try to find rank-r decomposition using ALS with rounding.
        """
        if time.time() > deadline:
            return None

        # Build target tensor
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0

        for restart in range(num_restarts):
            if time.time() > deadline:
                return None

            np.random.seed(1000 + r * 100 + restart)

            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)

            for it in range(num_iters):
                if time.time() > deadline:
                    return None

                # Update U: solve U M^T = T_flat where M = [outer(V_c, W_c)]
                M = np.zeros((mp * pn, r))
                for c in range(r):
                    M[:, c] = np.outer(V[:, c], W[:, c]).flatten()

                T_flat = T.reshape(nm, mp * pn)
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_sol.T
                except np.linalg.LinAlgError:
                    continue

                # Update V
                M_v = np.zeros((nm * pn, r))
                for c in range(r):
                    M_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()

                T_flat_v = np.zeros((mp, nm * pn))
                for v in range(mp):
                    for u in range(nm):
                        for w in range(pn):
                            T_flat_v[v, u * pn + w] = T[u, v, w]

                try:
                    V_sol, _, _, _ = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)
                    V = V_sol.T
                except np.linalg.LinAlgError:
                    continue

                # Update W
                M_w = np.zeros((nm * mp, r))
                for c in range(r):
                    M_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()

                T_flat_w = np.zeros((pn, nm * mp))
                for w in range(pn):
                    for u in range(nm):
                        for v in range(mp):
                            T_flat_w[w, u * mp + v] = T[u, v, w]

                try:
                    W_sol, _, _, _ = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)
                    W = W_sol.T
                except np.linalg.LinAlgError:
                    continue

            # Round and verify
            U_r = np.round(U * 2.0) / 2.0
            V_r = np.round(V * 2.0) / 2.0
            W_r = np.round(W * 2.0) / 2.0

            if np.max(np.abs(U - U_r)) > 0.01 or np.max(np.abs(V - V_r)) > 0.01 or np.max(np.abs(W - W_r)) > 0.01:
                continue

            if _verify_fast(U_r, V_r, W_r, n, m, p):
                U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
                V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
                W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]

                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    best_result = None
    best_rank = n * m * p

    # Try ranks from lower bound upward
    # Lower bound considerations: rank >= max(n*m, m*p, p*n) is not quite right
    # For matrix multiplication, the rank must be at least the maximum of the dimensions
    # in a more nuanced way. Let's try from 35 (m*p) upward.
    for r in range(35, 105):
        if time.time() > deadline:
            break

        # Try block solve first (faster, more likely to find integer solutions)
        result = try_rank_block_solve(r, num_restarts=8, seed_base=42)
        if result is not None:
            best_result = result
            best_rank = r
            break

        if time.time() > deadline:
            break

        # Try ALS as backup
        result = try_rank_als(r, num_iters=10, num_restarts=3)
        if result is not None:
            best_result = result
            best_rank = r
            break

    if best_result is not None:
        return best_result

    return schoolbook()
