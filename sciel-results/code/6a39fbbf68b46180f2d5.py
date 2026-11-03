"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 7
    num_outputs = p * n  # 14

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Precompute the target tensor: T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # Strategy:
    # 1. Try structured low-rank decompositions (Strassen-like ideas)
    # 2. Random search with rational rounding
    # 3. Fallback to schoolbook

    # For 2x3x7, the known lower bound is not well established, but let's try to find
    # a rank lower than 42.
    #
    # Key insight: We can fix U and V, then solve for W linearly.
    # For each (i,k), we have m=3 linear equations in r unknowns for W[k*n+i, :].
    #
    # Let's try a systematic search:
    # - For rank r from some lower bound up to 41
    # - Try random integer U, V with entries in {-1, 0, 1} or {-1, 0, 1, 2}
    # - Solve for W
    # - Check if W entries are close to integers or half-integers
    # - If so, round and verify exactly

    def is_half_integer(x, tol=1e-6):
        """Check if x is close to an integer or half-integer."""
        int_part = round(x)
        half_int_part = round(2 * x) / 2.0
        return abs(x - int_part) < tol or abs(x - half_int_part) < tol

    def to_exact_fraction(x):
        """Convert a float that should be integer or half-integer to a Fraction."""
        # Check if it's an integer
        if abs(x - round(x)) < 1e-6:
            return Fraction(int(round(x)))
        # Check if it's a half-integer
        if abs(2 * x - round(2 * x)) < 1e-6:
            return Fraction(int(round(2 * x)), 2)
        # Fallback: use Fraction from float (imprecise)
        return Fraction(x).limit_denominator(1000)

    def verify_decomposition(U, V, W, r, n, m, p, tol=1e-6):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(r))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_exact(U_frac, V_frac, W_frac, r, n, m, p):
        """Verify the decomposition exactly using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c] for c in range(r))
                    if val != 1:
                        return False
        return True

    def try_rank(r, max_attempts=100, entry_range=(-1, 2)):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None

        lo, hi = entry_range

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None

            # Random U and V with small integer entries
            U = np.random.randint(lo, hi + 1, size=(n * m, r))
            V = np.random.randint(lo, hi + 1, size=(m * p, r))

            W = np.zeros((p * n, r), dtype=float)
            valid = True

            for i in range(n):
                for k in range(p):
                    # Build m x r matrix
                    A_sys = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]

                    b_sys = np.ones(m)

                    if r == m:
                        det = np.linalg.det(A_sys)
                        if abs(det) < 1e-9:
                            valid = False
                            break
                        W_row = np.linalg.solve(A_sys, b_sys)
                    elif r > m:
                        # Use least squares
                        W_row = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    else:
                        # r < m: overdetermined, check consistency
                        W_row, residuals, rank, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-6:
                            valid = False
                            break

                    W[k * n + i, :] = W_row

                if not valid:
                    break

            if not valid:
                continue

            # Check if W entries are close to integer or half-integer
            entries_ok = True
            for row in W:
                for x in row:
                    if not is_half_integer(x, 1e-5):
                        entries_ok = False
                        break
                if not entries_ok:
                    break

            if not entries_ok:
                continue

            # Also check U and V (they are integers by construction, but verify)
            # U and V are already integers

            # Verify the decomposition
            if not verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r, n, m, p, 1e-5):
                continue

            # Convert to exact fractions
            U_frac = [[to_exact_fraction(x) for x in row] for row in U]
            V_frac = [[to_exact_fraction(x) for x in row] for row in V]
            W_frac = [[to_exact_fraction(x) for x in row] for row in W]

            # Verify exactly
            if verify_exact(U_frac, V_frac, W_frac, r, n, m, p):
                return {"U": U_frac, "V": V_frac, "W": W_frac}

        return None

    # Try ranks from lower to upper
    # The theoretical lower bound for 2x3x7 is not known to be less than 42,
    # but let's try a range.
    # Start from a reasonable lower bound and go up.
    # For small matrices, the rank is often close to the schoolbook rank.
    # Let's try ranks from 30 to 41.

    best_result = None
    best_rank = n * m * p

    # First, try some structured approaches
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8.
    # For 2x3x3, the rank is known to be 13 (vs schoolbook 18).
    # For 2x3x7, let's try to find something.

    # Try random search for various ranks
    for r in range(30, best_rank):
        if time.time() > deadline - 10:
            break

        # Adjust attempts based on rank
        if r <= 35:
            max_attempts = 200
        elif r <= 40:
            max_attempts = 100
        else:
            max_attempts = 50

        result = try_rank(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found the minimum rank, stop

    # If we didn't find anything better, try a few more with different entry ranges
    if best_result is None and time.time() < deadline - 20:
        for r in range(35, best_rank):
            if time.time() > deadline - 10:
                break
            result = try_rank(r, 100, entry_range=(-2, 3))
            if result is not None:
                best_result = result
                best_rank = r
                break

    if best_result is not None:
        # Convert to output format
        def frac_to_num(f):
            if f.denominator == 1:
                return f.numerator
            else:
                return float(f)

        U_out = [[frac_to_num(x) for x in row] for row in best_result["U"]]
        V_out = [[frac_to_num(x) for x in row] for row in best_result["V"]]
        W_out = [[frac_to_num(x) for x in row] for row in best_result["W"]]

        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fall back to schoolbook
        return make_schoolbook()
# EVOLVE-BLOCK-END
