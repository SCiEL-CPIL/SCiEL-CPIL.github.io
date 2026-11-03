import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
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
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank(r, num_attempts, seed_base=0):
        """
        Try to find a decomposition of rank r.
        Uses NumPy for fast random generation and linear algebra.
        """
        if time.time() > deadline:
            return None

        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None

            np.random.seed(seed_base + r * 1000 + attempt)

            # Generate random V and W with entries from {-1, 0, 1}
            # Using NumPy for speed
            V = np.random.choice([-1, 0, 1], size=(mp, r))
            W = np.random.choice([-1, 0, 1], size=(pn, r))

            # Ensure no column is all-zeros in V or W
            for c in range(r):
                if not np.any(V[:, c]):
                    V[np.random.randint(mp), c] = 1
                if not np.any(W[:, c]):
                    W[np.random.randint(pn), c] = 1

            U = np.zeros((nm, r))
            valid = True

            for i in range(n):
                # Build linear system for U[i*m+j, c] over j=0..m-1
                # Variables: m*r
                # Equations: m*p
                # A[eq_idx, var_idx] = V[j*p+k, c] * W[k*n+i, c]
                
                # We can construct A more efficiently
                A_mat = np.zeros((m*p, m*r))
                b_vec = np.ones(m*p)

                for j in range(m):
                    for k in range(p):
                        eq_idx = j * p + k
                        v_row = V[j*p+k, :]
                        w_row = W[k*n+i, :]
                        prod = v_row * w_row
                        # The variables for this equation are U[i*m+j, :]
                        # So we put prod into the row corresponding to j
                        A_mat[eq_idx, j*r:(j+1)*r] = prod

                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break

                # Round to nearest half-integer
                sol_rounded = np.round(sol * 2.0) / 2.0
                diff = np.max(np.abs(sol - sol_rounded))

                # If half-integers don't work well, try integers
                if diff > 1e-8:
                    sol_int = np.round(sol)
                    diff_int = np.max(np.abs(sol - sol_int))
                    if diff_int < diff:
                        sol_rounded = sol_int
                        diff = diff_int

                if diff > 1e-6:
                    valid = False
                    break

                U[i*m:(i+1)*m, :] = sol_rounded

            if not valid:
                continue

            # Verify numerically first (fast)
            if _verify_np(U, V, W, n, m, p):
                # Convert to fractions for exact verification
                # Only do this if numerical verification passes
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                U_frac = [[Fraction(U_list[i][j]).limit_denominator(100) for j in range(r)] for i in range(nm)]
                V_frac = [[Fraction(V_list[i][j]).limit_denominator(100) for j in range(r)] for i in range(mp)]
                W_frac = [[Fraction(W_list[i][j]).limit_denominator(100) for j in range(r)] for i in range(pn)]

                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }

        return None

    # Search for lower ranks
    # Focus on ranks close to schoolbook first, as they are more likely to be reachable
    # if any improvement exists.
    for r in range(104, 99, -1):
        if time.time() > deadline:
            break

        # Increase attempts for the most critical ranks
        num_attempts = 100
        result = try_rank(r, num_attempts, seed_base=0)
        if result is not None:
            return result
            
    # If no improvement found in the critical range, try a few more lower ranks with fewer attempts
    for r in range(99, 80, -1):
        if time.time() > deadline:
            break
        num_attempts = 20
        result = try_rank(r, num_attempts, seed_base=100)
        if result is not None:
            return result

    return schoolbook()
