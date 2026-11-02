"""Matrix multiplication 3x5x7 - rank minimization via structured search."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n

    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_structured_rank(r, max_iters=50):
        """
        Try structured sparse decomposition.
        Use patterns where each column has limited support.
        """
        if time.time() > deadline:
            return None

        # Strategy: use a block-diagonal-like structure
        # For each column c, assign it to a specific (i,j,k) "group"
        # and use small integer values

        for seed_val in range(max_iters):
            if time.time() > deadline:
                return None

            random.seed(777 + r * 13 + seed_val)

            # Generate sparse V and W
            V = [[0]*r for _ in range(mp)]
            W = [[0]*r for _ in range(pn)]
            U = [[0]*r for _ in range(nm)]

            # For each column, randomly assign non-zero entries
            for c in range(r):
                # V: pick 1-2 rows to be non-zero
                num_v_nonzero = random.randint(1, 2)
                v_rows = random.sample(range(mp), min(num_v_nonzero, mp))
                for row in v_rows:
                    V[row][c] = random.choice([-2, -1, 1, 2])

                # W: pick 1-2 rows to be non-zero
                num_w_nonzero = random.randint(1, 2)
                w_rows = random.sample(range(pn), min(num_w_nonzero, pn))
                for row in w_rows:
                    W[row][c] = random.choice([-2, -1, 1, 2])

                # U: pick 1-3 rows to be non-zero
                num_u_nonzero = random.randint(1, 3)
                u_rows = random.sample(range(nm), min(num_u_nonzero, nm))
                for row in u_rows:
                    U[row][c] = random.choice([-3, -2, -1, 1, 2, 3])

            # Check if this works
            if _verify_np(U, V, W, n, m, p):
                # Convert to exact fractions
                U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]

                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    def try_solve_rank(r, num_restarts=30):
        """
        Fix V,W with specific structure, solve for U exactly.
        Use a pattern where V and W have known structure.
        """
        if time.time() > deadline:
            return None

        for restart in range(num_restarts):
            if time.time() > deadline:
                return None

            random.seed(999 + r * 7 + restart)

            # Create structured V and W
            # V: each row gets values from a small set
            # W: each row gets values from a small set
            vals = [-2, -1, 0, 1, 2]

            V = [[random.choice(vals) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(pn)]

            U = [[0]*r for _ in range(nm)]

            valid = True
            for i in range(n):
                # Build linear system for U block corresponding to i
                # Unknowns: U[i*m+j, c] for j=0..m-1, c=0..r-1 -> m*r unknowns
                # Equations: for each (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # That's m*p equations

                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)

                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for c in range(r):
                            A_mat[eq_idx, j*r + c] = V[j*p+k][c] * W[k*n+i][c]

                # Solve
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    valid = False
                    break

                # Try to find rational approximation
                found = False
                best_sol = None
                best_err = float('inf')

                for d in range(1, 21):
                    sol_scaled = sol * d
                    sol_rounded = np.round(sol_scaled) / d
                    err = np.max(np.abs(sol - sol_rounded))
                    if err < 1e-10:
                        found = True
                        best_sol = sol_rounded
                        break
                    if err < best_err:
                        best_err = err
                        best_sol = sol_rounded

                if not found:
                    # Try with half-integers
                    sol_half = np.round(sol * 2.0) / 2.0
                    err_half = np.max(np.abs(sol - sol_half))
                    if err_half < 1e-9:
                        best_sol = sol_half
                    else:
                        valid = False
                        break

                for j in range(m):
                    for c in range(r):
                        U[i*m+j][c] = float(best_sol[j*r + c])

            if not valid:
                continue

            if _verify_np(U, V, W, n, m, p):
                U_frac = [[Fraction(U[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]

                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    def try_gram_als(r, num_iters=20, num_restarts=3):
        """
        Use ALS with Gram matrix structure.
        Represent the tensor as sum of rank-1 terms.
        """
        if time.time() > deadline:
            return None

        # Build target tensor
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0

        for restart in range(num_restarts):
            if time.time() > deadline:
                return None

            np.random.seed(2000 + r * 11 + restart)

            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)

            for it in range(num_iters):
                if time.time() > deadline:
                    return None

                # Update U
                M_u = np.zeros((mp*pn, r))
                for c in range(r):
                    M_u[:, c] = np.outer(V[:, c], W[:, c]).flatten()

                T_flat_u = T.reshape(nm, mp*pn)
                try:
                    U_sol = np.linalg.lstsq(M_u, T_flat_u.T, rcond=None)[0].T
                    U = U_sol
                except:
                    continue

                # Update V
                M_v = np.zeros((nm*pn, r))
                for c in range(r):
                    M_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()

                T_flat_v = np.zeros((mp, nm*pn))
                for v_idx in range(mp):
                    for u_idx in range(nm):
                        for w_idx in range(pn):
                            T_flat_v[v_idx, u_idx*pn + w_idx] = T[u_idx, v_idx, w_idx]

                try:
                    V_sol = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)[0].T
                    V = V_sol
                except:
                    continue

                # Update W
                M_w = np.zeros((nm*mp, r))
                for c in range(r):
                    M_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()

                T_flat_w = np.zeros((pn, nm*mp))
                for w_idx in range(pn):
                    for u_idx in range(nm):
                        for v_idx in range(mp):
                            T_flat_w[w_idx, u_idx*mp + v_idx] = T[u_idx, v_idx, w_idx]

                try:
                    W_sol = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)[0].T
                    W = W_sol
                except:
                    continue

            # After ALS, try to round to rationals
            U_r = np.round(U * 2.0) / 2.0
            V_r = np.round(V * 2.0) / 2.0
            W_r = np.round(W * 2.0) / 2.0

            if _verify_np(U_r, V_r, W_r, n, m, p):
                U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
                V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
                W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]

                if _verify_fraction(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    # Try different strategies in order of likelihood
    # Lower bound: max(nm, mp, pn) = 35
    # Realistic: try from 40 upward

    # First try structured sparse (fast, may work for moderate rank)
    for r in range(40, 70):
        if time.time() > deadline:
            break
        result = try_structured_rank(r, max_iters=20)
        if result is not None:
            return result

    # Try solve-based approach
    for r in range(40, 65):
        if time.time() > deadline:
            break
        result = try_solve_rank(r, num_restarts=15)
        if result is not None:
            return result

    # Try ALS with Gram structure
    for r in range(40, 60):
        if time.time() > deadline:
            break
        result = try_gram_als(r, num_iters=10, num_restarts=2)
        if result is not None:
            return result

    # Extended search
    for r in range(60, 106):
        if time.time() > deadline:
            break
        result = try_structured_rank(r, max_iters=10)
        if result is not None:
            return result

    # Fallback to schoolbook
    return schoolbook()
