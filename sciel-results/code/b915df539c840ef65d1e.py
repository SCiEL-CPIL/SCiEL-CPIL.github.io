"""Rewritten program: odd-symmetric degree-five polynomial closure with ridge regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared odd degree-five polynomial to the unresolved fast-scale forcing.
    Enforces physical antisymmetry: only odd powers (x, x^3, x^5) are fitted.
    Uses ridge regularization for stability and a small validation-free polish.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 3:
        # Not enough data; return zeros
        return np.zeros(6)

    # Build design matrix for odd powers: x, x^3, x^5
    # Coefficients are in ascending order: c0, c1, c2, c3, c4, c5
    # We only fit c1, c3, c5 (odd indices)
    A = np.column_stack([x, x**3, x**5])

    # Ridge regularization: solve (A^T A + lambda * I) c = A^T y
    # Choose lambda based on data scale
    # Scale features to have unit variance for better conditioning
    col_scales = np.std(A, axis=0)
    col_scales[col_scales < 1e-12] = 1.0
    A_scaled = A / col_scales

    # Multiple lambda values to find best regularized solution
    # We'll use a simple approach: try a few lambdas and pick the one
    # that gives the best training fit with reasonable coefficient magnitudes
    lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    best_coeff = np.zeros(3)
    best_score = np.inf

    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
        try:
            # Ridge solve
            AtA = A_scaled.T @ A_scaled
            Aty = A_scaled.T @ y
            reg_matrix = AtA + lam * np.eye(3)
            c_scaled = np.linalg.solve(reg_matrix, Aty)
            # Unscale back
            c = c_scaled / col_scales

            # Score: training MSE + penalty on coefficient magnitude
            pred = A @ c
            mse = np.mean((pred - y) ** 2)
            # Penalty: sum of squared coefficients (encourages simplicity)
            penalty = 1e-4 * np.sum(c**2)
            score = mse + penalty

            if score < best_score:
                best_score = score
                best_coeff = c
        except np.linalg.LinAlgError:
            continue

    # Final polish: if we have time, do a higher-precision solve with the best lambda
    # Actually, let's just use the best we found. The ridge solve is already exact.

    # Construct full 6-coefficient array in ascending power order
    coeff = np.zeros(6)
    coeff[1] = best_coeff[0]  # c1 for x^1
    coeff[3] = best_coeff[1]  # c3 for x^3
    coeff[5] = best_coeff[2]  # c5 for x^5

    # Ensure finite values
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
