"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    x_std_arr = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree 5 in standardized x
    V = np.vander(x_std_arr, N=6, increasing=True)

    # Add small ridge regularization for stability
    ridge = 1e-8 * np.eye(6)
    A = V.T @ V + ridge
    b = V.T @ y

    # Solve the normal equations
    try:
        coeff_std = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_std = np.linalg.lstsq(A, b, rcond=None)[0]

    # Convert from standardized coefficients to original x coefficients
    # y = sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    # = sum_{i=0}^{5} c_i * x_std^{-i} * (x - x_mean)^i
    # Expand (x - x_mean)^i using binomial theorem
    # (x - m)^i = sum_{k=0}^{i} C(i,k) * x^k * (-m)^(i-k)
    coeff = np.zeros(6, dtype=np.float64)
    for i in range(6):
        ci = coeff_std[i]
        if abs(ci) < 1e-15:
            continue
        for k in range(i + 1):
            # C(i,k) * (-x_mean)^(i-k) * x_std^{-i}
            binom = 1.0
            for j in range(k):
                binom *= (i - j) / (j + 1)
            coeff[k] += ci * binom * ((-x_mean) ** (i - k)) / (x_std ** i)

    # Check time
    if time.time() - t_start > deadline:
        return coeff

    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff)):
        return np.zeros(6, dtype=np.float64)

    return coeff
# EVOLVE-BLOCK-END
