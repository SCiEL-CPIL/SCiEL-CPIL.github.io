"""Climate dynamics: unresolved-tendency closure via robust polynomial fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _tukey_weights(residuals, c=4.6851):
    """Tukey's biweight robust weights."""
    scale = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
    if scale < 1e-12:
        scale = np.std(residuals)
    if scale < 1e-12:
        return np.ones(len(residuals))
    r = residuals / scale
    w = np.ones(len(r))
    mask = np.abs(r) > c
    w[mask] = 0.0
    inner = ~mask
    w[inner] = (1.0 - (r[inner] / c) ** 2) ** 2
    return w


def _irls_polyfit(x, y, degree, max_iter=30, c_tukey=4.6851):
    """Iteratively reweighted least squares polynomial fit with Tukey weights."""
    n = len(x)
    if n <= degree + 1:
        # Fall back to ordinary least squares
        try:
            return np.polynomial.polynomial.polyfit(x, y, degree)
        except Exception:
            return np.zeros(degree + 1)

    # Initial fit (ordinary least squares)
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, degree)
    except Exception:
        return np.zeros(degree + 1)

    for _ in range(max_iter):
        # Compute residuals
        y_pred = np.polynomial.polynomial.polyval(x, coeff)
        residuals = y - y_pred

        # Compute Tukey weights
        w = _tukey_weights(residuals, c_tukey)

        # Weighted least squares
        W = np.diag(w)
        # Build Vandermonde matrix
        V = np.vander(x, degree + 1, increasing=True)
        # Solve weighted normal equations: (V^T W V) coeff = V^T W y
        VW = V * w[:, None]
        try:
            A = VW.T @ V
            b = VW.T @ y
            # Add small ridge for numerical stability
            A += 1e-10 * np.eye(degree + 1)
            new_coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Check convergence
        if np.allclose(coeff, new_coeff, rtol=1e-8, atol=1e-12):
            coeff = new_coeff
            break
        coeff = new_coeff

    return coeff


def _scale_coeffs(coeff, x_scale, y_scale):
    """
    Convert coefficients from scaled space to original space.
    If we fit p(t) where t = x/x_scale, and y is in y_scale units,
    then y = y_scale * p(x/x_scale) = y_scale * sum c_i (x/x_scale)^i
    = sum (y_scale / x_scale^i) * c_i * x^i
    So original coeff_i = y_scale * c_i / x_scale^i
    """
    n = len(coeff)
    original = np.zeros(n)
    for i in range(n):
        original[i] = coeff[i] * y_scale / (x_scale ** i)
    return original


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    Uses robust IRLS with Tukey biweight to downweight outliers.
    """
    start_time = time.time()
    deadline = 150.0  # seconds, with safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove any NaN/Inf
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Check time
    if time.time() - start_time > deadline - 5:
        # Not enough time, return a simple fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n - 1))
            out = np.zeros(6)
            out[:len(coeff)] = coeff
            return out
        except Exception:
            return np.zeros(6)

    # Scale x and y for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    # Fit in scaled space
    degree = 5
    coeff_scaled = _irls_polyfit(x_scaled, y_scaled, degree, max_iter=40, c_tukey=4.6851)

    # Check time
    if time.time() - start_time > deadline - 10:
        # Convert and return
        # y = y_mean + y_std * p((x - x_mean)/x_std)
        # p(t) = sum c_i t^i
        # y = y_mean + y_std * sum c_i ((x-x_mean)/x_std)^i
        # = y_mean + y_std * sum c_i (x-x_mean)^i / x_std^i
        # = y_mean + sum (y_std/x_std^i) c_i (x-x_mean)^i
        # Expand (x-x_mean)^i = sum_{j=0}^{i} binom(i,j) x^j (-x_mean)^{i-j}
        # This is complex; let's use a different approach.
        # Actually, let's just do the conversion by evaluating.
        # Better: re-express. We want coefficients in powers of x.
        # y = y_mean + y_std * sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
        # Let u = x - x_mean, so x = u + x_mean
        # y = y_mean + y_std * sum_{i=0}^{5} c_i * (u/x_std)^i
        # = y_mean + sum_{i=0}^{5} (y_std / x_std^i) * c_i * u^i
        # Now expand u^i = (x - x_mean)^i = sum_{j=0}^{i} C(i,j) x^j (-x_mean)^{i-j}
        # So y = y_mean + sum_{i=0}^{5} (y_std * c_i / x_std^i) * sum_{j=0}^{i} C(i,j) x^j (-x_mean)^{i-j}
        # = y_mean + sum_{j=0}^{5} x^j * [sum_{i=j}^{5} (y_std * c_i / x_std^i) * C(i,j) * (-x_mean)^{i-j}]

        orig_coeff = np.zeros(6)
        orig_coeff[0] = y_mean  # constant term from y_mean
        for j in range(6):
            s = 0.0
            for i in range(j, 6):
                if i < 6:
                    binom = 1.0
                    for k in range(1, j + 1):
                        binom *= (i - k + 1) / k
                    s += (y_std * coeff_scaled[i] / (x_std ** i)) * binom * ((-x_mean) ** (i - j))
            orig_coeff[j] += s

        return orig_coeff

    # Full conversion from scaled to original coefficients
    # y = y_mean + y_std * p((x - x_mean)/x_std)
    # where p(t) = sum_{i=0}^{5} c_i * t^i
    # We need: y = sum_{j=0}^{5} a_j * x^j
    #
    # y = y_mean + y_std * sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    #   = y_mean + sum_{i=0}^{5} (y_std * c_i / x_std^i) * (x - x_mean)^i
    #
    # Let d_i = y_std * c_i / x_std^i
    # y = y_mean + sum_{i=0}^{5} d_i * (x - x_mean)^i
    #
    # Expand (x - x_mean)^i using binomial:
    # (x - x_mean)^i = sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^{i-j}
    #
    # So y = y_mean + sum_{j=0}^{5} x^j * [sum_{i=j}^{5} d_i * C(i,j) * (-x_mean)^{i-j}]

    orig_coeff = np.zeros(6)
    orig_coeff[0] = y_mean

    for i in range(6):
        d_i = y_std * coeff_scaled[i] / (x_std ** i)
        for j in range(i + 1):
            # C(i, j)
            binom = 1.0
            for k in range(1, j + 1):
                binom *= (i - k + 1) / k
            orig_coeff[j] += d_i * binom * ((-x_mean) ** (i - j))

    # Final check for finite values
    if not np.all(np.isfinite(orig_coeff)):
        # Fall back to simple polyfit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n - 1))
            out = np.zeros(6)
            out[:len(coeff)] = coeff
            return out
        except Exception:
            return np.zeros(6)

    return orig_coeff
# EVOLVE-BLOCK-END
