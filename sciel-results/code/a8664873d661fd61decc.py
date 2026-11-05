"""
Improved program for unresolved-tendency closure task.
Uses piecewise cubic spline approximation projected onto degree-5 monomial basis.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses a piecewise cubic spline with quantile-based knots as an intermediate
    representation, then projects onto a degree-5 monomial polynomial.
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
        return np.zeros(6)

    # Sort by x for spline fitting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Determine number of knots based on data size
    # More knots for larger datasets, capped to keep computation fast
    max_knots = 20
    if n < 100:
        n_knots = max(5, min(n // 4, max_knots))
    else:
        n_knots = max(10, min(n // 10, max_knots))

    # Quantile-based knot placement
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]  # interior knots
    knot_values = np.quantile(x_sorted, quantiles)

    # Remove duplicate knots (can happen with quantiles)
    unique_knots = np.unique(knot_values)
    if len(unique_knots) < 4:
        # Fallback: evenly spaced knots
        x_min, x_max = x_sorted[0], x_sorted[-1]
        if x_max - x_min < 1e-12:
            return np.zeros(6)
        unique_knots = np.linspace(x_min, x_max, max(5, n_knots))

    # Add boundary points for natural spline
    x_min, x_max = x_sorted[0], x_sorted[-1]
    all_knots = np.concatenate([[x_min], unique_knots, [x_max]])
    # Ensure strictly increasing
    all_knots = np.unique(all_knots)

    if len(all_knots) < 4:
        # Not enough knots for cubic spline, fall back to polynomial
        return _fit_direct_polynomial(x, y, start_time, DEADLINE)

    # Fit cubic spline with natural boundary conditions
    try:
        cs = CubicSpline(all_knots, y_sorted, bc_type='natural')
    except Exception:
        return _fit_direct_polynomial(x, y, start_time, DEADLINE)

    # Project the spline onto a degree-5 monomial polynomial
    # We want to find c = [c0, c1, c2, c3, c4, c5] such that
    # sum(c_i * x^i) approximates cs(x) well over the training data range
    best_c_x = _project_spline_to_polynomial(cs, x, y, start_time, DEADLINE)

    if best_c_x is not None and np.all(np.isfinite(best_c_x)):
        return best_c_x

    # Fallback to direct polynomial fit
    return _fit_direct_polynomial(x, y, start_time, DEADLINE)


def _project_spline_to_polynomial(cs, x, y, start_time, DEADLINE):
    """
    Project a cubic spline onto a degree-5 monomial polynomial by minimizing
    the squared error between the spline and the polynomial on the training data.
    """
    # Evaluate spline on training points
    y_spline = cs(x)

    # Also include the original y for regularization toward the data
    # We want the polynomial to be close to both the spline and the data
    # Weight the spline fit more heavily since it captures local structure

    # Build Vandermonde matrix for degree-5 polynomial
    V = np.vander(x, N=6, increasing=True)

    # Objective: minimize ||V*c - y_spline||^2 + alpha * ||V*c - y||^2
    # This balances fidelity to the spline (local structure) and the data
    # alpha controls the trade-off; small alpha means trust the spline more

    def objective(c, alpha=0.1):
        pred = V @ c
        err_spline = np.sum((pred - y_spline) ** 2)
        err_data = np.sum((pred - y) ** 2)
        return err_spline + alpha * err_data

    # Use least squares with combined target
    # Equivalent to minimizing ||[V; sqrt(alpha)*V] @ c - [y_spline; sqrt(alpha)*y]||^2
    alpha = 0.1
    V_aug = np.vstack([V, np.sqrt(alpha) * V])
    y_aug = np.concatenate([y_spline, np.sqrt(alpha) * y])

    try:
        c_x, residuals, rank, sv = np.linalg.lstsq(V_aug, y_aug, rcond=None)
        if np.all(np.isfinite(c_x)):
            return c_x
    except Exception:
        pass

    # Fallback: just fit to the spline values
    try:
        c_x, residuals, rank, sv = np.linalg.lstsq(V, y_spline, rcond=None)
        if np.all(np.isfinite(c_x)):
            return c_x
    except Exception:
        pass

    return None


def _fit_direct_polynomial(x, y, start_time, DEADLINE):
    """
    Direct degree-5 polynomial fit with Tikhonov regularization.
    Fallback when spline approach fails.
    """
    n = len(x)
    if n < 6:
        return np.zeros(6)

    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    u = (x - x_mean) / x_std

    def convert_scaled_to_original(c_u, x_mean, x_std):
        c_x = np.zeros(6)
        for k in range(6):
            if abs(c_u[k]) < 1e-15:
                continue
            coeff_k = c_u[k] / (x_std ** k)
            for j in range(k + 1):
                c_x[j] += coeff_k * comb(k, j) * ((-x_mean) ** (k - j))
        return c_x

    def fit_regularized(u_data, y_data, lam, weight_vec):
        V = np.vander(u_data, N=6, increasing=True)
        VtV = V.T @ V
        W = np.diag(weight_vec)
        A = VtV + lam * W
        b = V.T @ y_data
        try:
            c_u = np.linalg.solve(A, b)
            if np.all(np.isfinite(c_u)):
                return c_u
        except np.linalg.LinAlgError:
            pass
        return None

    weight_vec = np.array([1.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    best_c_x = None
    best_mse = np.inf

    for lam in [1e-6, 1e-4, 1e-2, 1.0, 10.0]:
        if time.time() > DEADLINE:
            break
        c_u = fit_regularized(u, y, lam, weight_vec)
        if c_u is None:
            continue
        c_x = convert_scaled_to_original(c_u, x_mean, x_std)
        if not np.all(np.isfinite(c_x)):
            continue
        pred = np.zeros_like(x)
        for j in range(6):
            pred += c_x[j] * (x ** j)
        mse = np.mean((pred - y) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_c_x = c_x

    if best_c_x is None:
        try:
            coeffs = np.polyfit(x, y, 1)
            c_x = np.zeros(6)
            c_x[1] = coeffs[0]
            c_x[0] = coeffs[1]
            best_c_x = c_x
        except Exception:
            best_c_x = np.zeros(6)

    return best_c_x
# EVOLVE-BLOCK-END
