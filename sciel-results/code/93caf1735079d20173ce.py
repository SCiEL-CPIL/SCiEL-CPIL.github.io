"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    Uses ridge regression with the intercept excluded from the penalty.
    """
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Normalize x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree 5 polynomial in x_norm
    # Columns: 1, x_norm, x_norm^2, ..., x_norm^5
    V = np.vander(x_norm, N=6, increasing=True)

    # Ridge regression with intercept (first column) excluded from penalty
    # Regularization matrix: diag([0, lambda, lambda, ..., lambda])
    # We try a small lambda to stabilize without biasing the mean too much
    
    best_coeff = None
    best_score = np.inf

    # Try a few regularization strengths
    for lam in [0.0, 1e-6, 1e-5, 1e-4]:
        if time.time() > deadline:
            break

        # Build regularized normal equations
        # (V^T V + lambda * diag([0,1,1,1,1,1])) c = V^T y
        VtV = V.T @ V
        Vty = V.T @ y

        # Add regularization to all but the intercept
        reg = np.zeros((6, 6))
        reg[1:, 1:] = lam * np.eye(5)
        reg[0, 0] = 0.0  # No regularization on intercept

        A = VtV + reg
        b = Vty

        try:
            c_norm = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                c_norm = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                continue

        # Transform coefficients back to original x scale
        # If p(x_norm) = sum c_i * x_norm^i, and x_norm = (x - x_mean)/x_std
        # We need to express in terms of x
        # Use polynomial transformation
        try:
            # c_norm are coefficients in powers of x_norm
            # Convert to coefficients in powers of x
            # x_norm = (x - x_mean)/x_std = x/x_std - x_mean/x_std
            # Let a = 1/x_std, b0 = -x_mean/x_std
            # x_norm = a*x + b0
            # We want sum c_i * (a*x + b0)^i = sum d_j * x^j
            a = 1.0 / x_std
            b0 = -x_mean / x_std

            # Compute the transformation
            d = np.zeros(6)
            # For each power i of x_norm, expand (a*x + b0)^i
            for i in range(6):
                if abs(c_norm[i]) < 1e-15:
                    continue
                # Expand (a*x + b0)^i using binomial theorem
                for j in range(i + 1):
                    # Coefficient of x^j in (a*x + b0)^i is C(i,j) * a^j * b0^(i-j)
                    coeff = c_norm[i] * _comb(i, j) * (a ** j) * (b0 ** (i - j))
                    d[j] += coeff
        except:
            # Fallback: if x_std is very small, just use c_norm directly
            d = c_norm.copy()

        # Evaluate training MSE to check quality
        pred = np.polyval(d[::-1], x)  # polyval expects descending order
        mse = np.mean((pred - y) ** 2)
        
        if mse < best_score:
            best_score = mse
            best_coeff = d

    if best_coeff is None:
        # Fallback to simple linear fit
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c_lin
        except:
            best_coeff = np.zeros(6)

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.nan_to_num(best_coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return best_coeff


def _comb(n, k):
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
# EVOLVE-BLOCK-END
