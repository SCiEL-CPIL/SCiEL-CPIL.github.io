"""Rewritten program for unresolved-tendency closure via polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Try degree-5 polynomial fit with regularization
    # Use scaled/shifted x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    # Normalize x for fitting
    x_norm = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree 5 in normalized x
    deg = 5
    V = np.vander(x_norm, deg + 1, increasing=True)

    # Solve least squares with Tikhonov regularization
    # Try several regularization strengths
    best_coeff = None
    best_mse = np.inf

    for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
        try:
            if lam == 0.0:
                # Pure least squares
                coeff_norm, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
            else:
                # Regularized: (V^T V + lam * I) c = V^T y
                # But we want to penalize higher-order terms more
                # Use diagonal regularization weighted by degree
                A = V.T @ V
                b = V.T @ y
                # Regularization matrix: stronger for higher degrees
                reg = np.diag([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0] * lam)
                A_reg = A + reg
                coeff_norm = np.linalg.solve(A_reg, b)

            # Convert back to original x scale
            # y = sum_{i=0}^{5} c_i * x^i
            # We fit in terms of x_norm = (x - x_mean) / x_std
            # So x = x_std * x_norm + x_mean
            # We need to expand (x_std * t + x_mean)^i in terms of t
            # This is a polynomial transformation
            # For simplicity, let's just use the normalized coefficients directly
            # but we need coefficients in original x powers

            # Actually, the task asks for coefficients in powers of x (the local slow state)
            # So we need c_i such that f(x) = sum c_i * x^i
            # We fit g(t) = sum d_i * t^i where t = (x - x_mean) / x_std
            # Then f(x) = g((x - x_mean) / x_std)
            # We need to expand this in powers of x

            # Let's use numpy's polynomial conversion
            # g(t) = d_0 + d_1*t + d_2*t^2 + ... + d_5*t^5
            # t = (x - x_mean) / x_std = (1/x_std)*x - x_mean/x_std
            # So t = a*x + b where a = 1/x_std, b = -x_mean/x_std
            # We can use np.polynomial.polynomial.polyfromroots or manual expansion

            # Use polynomial composition
            # f(x) = g(a*x + b)
            # We can use np.polynomial.polynomial.polypow and polyadd, etc.
            # Or simply evaluate the transformation

            # Let's use a simpler approach: use np.poly1d for the transformation
            # Or use the fact that we can compute the coefficients via matrix multiplication

            # Build the transformation matrix T such that c = T @ d
            # where d are coefficients in t and c are coefficients in x
            # t = a*x + b
            # t^k = (a*x + b)^k = sum_{j=0}^{k} binom(k,j) * a^j * b^{k-j} * x^j
            # So d_k * t^k = d_k * sum_{j=0}^{k} binom(k,j) * a^j * b^{k-j} * x^j
            # c_j = sum_{k=j}^{5} d_k * binom(k,j) * a^j * b^{k-j}

            a = 1.0 / x_std
            b = -x_mean / x_std

            c = np.zeros(6)
            for k in range(6):
                for j in range(k + 1):
                    from math import comb
                    c[j] += coeff_norm[k] * comb(k, j) * (a ** j) * (b ** (k - j))

            # Compute MSE on training data
            pred = np.zeros(n)
            for i in range(6):
                pred += c[i] * (x ** i)
            mse = np.mean((pred - y) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_coeff = c

        except Exception:
            continue

        if time.time() - t_start > deadline:
            break

    if best_coeff is None:
        # Fallback to simple linear fit
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff[0] = np.mean(y)
        return coeff

    return best_coeff
# EVOLVE-BLOCK-END
