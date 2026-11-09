import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    if time.time() - t_start > deadline - 5.0:
        return np.zeros(6, dtype=np.float64)

    mean_x = np.mean(x)
    std_x = np.std(x)
    
    if std_x < 1e-12:
        coeff = np.zeros(6, dtype=np.float64)
        coeff[0] = np.mean(y)
        return coeff

    x_scaled = (x - mean_x) / std_x

    # Construct Vandermonde matrix for degree 5
    # V[:, i] = x_scaled ** i
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Add regularization
    # lambda should be small relative to the scale of V^T V
    # Let's use a small value, e.g., 1e-6 * trace(V^T V) / 6
    VtV = V.T @ V
    Vty = V.T @ y
    
    # Regularization parameter
    # We want to penalize high-order terms more? Or just standard ridge?
    # Standard ridge is fine.
    reg = 1e-6 * np.trace(VtV) / 6.0
    A = VtV + reg * np.eye(6)
    
    try:
        c_scaled = np.linalg.solve(A, Vty)
    except np.linalg.LinAlgError:
        # Fallback to polyfit
        try:
            c_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Transform coefficients back to original scale
    # a_k = sum_{i=k}^5 c_i * binom(i, k) * (-mean)^(i-k) / std^i
    
    # Precomputed binom coefficients for i=0..5, k=0..i
    # binom[i][k]
    binom = [
        [1],
        [1, 1],
        [1, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 6, 4, 1],
        [1, 5, 10, 10, 5, 1]
    ]
    
    a = np.zeros(6, dtype=np.float64)
    for i in range(6):
        coeff_i = c_scaled[i] / (std_x ** i)
        for k in range(i + 1):
            a[k] += coeff_i * binom[i][k] * ((-mean_x) ** (i - k))

    if not np.all(np.isfinite(a)):
        return np.zeros(6, dtype=np.float64)

    return a
