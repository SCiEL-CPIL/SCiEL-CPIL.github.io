"""Rewritten program for unresolved-tendency closure task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses regularized least-squares with a time deadline guard.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        # Not enough data for degree-5 fit; fall back to zero
        return np.zeros(6)

    # Build Vandermonde matrix for polynomial basis [1, x, x^2, x^3, x^4, x^5]
    # Use scaled x to improve conditioning
    x_scale = np.std(x)
    if x_scale < 1e-12:
        x_scale = 1.0
    x_norm = x / x_scale

    V = np.vander(x_norm, N=6, increasing=True)  # columns: x^0, x^1, ..., x^5

    # Regularized least squares: minimize ||V @ c - y||^2 + lambda * ||c||^2
    # Use a small regularization to handle ill-conditioning
    # Try a few lambda values and pick the best based on training fit
    best_coeff = None
    best_mse = np.inf

    # Regularization strengths to try
    lambdas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]

    for lam in lambdas:
        if time.time() > DEADLINE:
            break

        try:
            if lam == 0.0:
                # Unregularized: use lstsq
                coeff_norm, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
            else:
                # Regularized: solve (V^T V + lambda*I) c = V^T y
                A = V.T @ V + lam * np.eye(6)
                b = V.T @ y
                coeff_norm = np.linalg.solve(A, b)

            # Convert back to original x scale
            # If y = sum c_i * (x/scale)^i, then y = sum (c_i / scale^i) * x^i
            coeff = np.zeros(6)
            for i in range(6):
                coeff[i] = coeff_norm[i] / (x_scale ** i)

            # Check if coefficients are finite
            if not np.all(np.isfinite(coeff)):
                continue

            # Compute training MSE
            pred = np.zeros_like(y)
            for i in range(6):
                pred += coeff[i] * (x ** i)
            mse = np.mean((pred - y) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_coeff = coeff.copy()

        except np.linalg.LinAlgError:
            continue

    if best_coeff is not None:
        return best_coeff

    # Fallback: simple linear fit
    try:
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    except Exception:
        return np.zeros(6)
# EVOLVE-BLOCK-END
