# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree-5 polynomial in x_scaled
    # V[:, j] = x_scaled^j for j=0..5
    V = np.vander(x_scaled, N=6, increasing=True)

    # Ridge regression with small regularization to stabilize
    # Try a few lambda values and pick best (simple, fast)
    best_coeff_raw = np.zeros(6)
    best_mse = np.inf

    lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    for lam in lambdas:
        if time.time() > deadline:
            break
        # Solve (V^T V + lam*I) c = V^T y
        A = V.T @ V
        b = V.T @ y
        if lam > 0:
            A_reg = A + lam * np.eye(6)
        else:
            A_reg = A
        try:
            c_scaled = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            c_scaled = np.linalg.lstsq(A_reg, b, rcond=None)[0]

        # Compute training MSE
        pred = V @ c_scaled
        mse = np.mean((pred - y) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_coeff_raw = c_scaled

    # Convert from scaled polynomial back to original x powers
    # If p(x_scaled) = sum c_j * x_scaled^j, and x_scaled = (x - mu)/sigma
    # Then x_scaled^j = (x - mu)^j / sigma^j = sum_{k=0}^{j} binom(j,k) * (-mu)^(j-k) * x^k / sigma^j
    # So coefficient of x^k in original = sum_{j=k}^{5} c_j * binom(j,k) * (-mu)^(j-k) / sigma^j

    mu = x_mean
    sigma = x_std
    coeff = np.zeros(6)
    for k in range(6):
        for j in range(k, 6):
            # binom(j, k) * (-mu)^(j-k) / sigma^j
            # Compute binom(j,k)
            binom_val = 1.0
            for m in range(k):
                binom_val *= (j - m) / (m + 1)
            coeff[k] += best_coeff_raw[j] * binom_val * ((-mu) ** (j - k)) / (sigma ** j)

    # Verify all coefficients are finite
    if not np.all(np.isfinite(coeff)):
        return np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
