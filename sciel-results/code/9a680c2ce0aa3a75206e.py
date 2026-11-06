# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0

    # --- Stage 1: Kernel-weighted polynomial smoother to denoise ---
    # Use a Gaussian-kernel weighted least-squares degree-5 polynomial fit
    # to estimate the smooth underlying E[T|X]
    mad = np.median(np.abs(x - np.median(x)))
    if mad < 1e-12:
        mad = x_std * 0.6745
    bandwidth = 0.9 * min(x_std, 1.06 * mad / 0.6745) * n ** (-1.0 / 5.0)
    if bandwidth < 1e-6:
        bandwidth = x_std * 0.1

    # Scale x for numerical stability in polynomial construction
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=6, increasing=True)

    # Try multiple bandwidths for the smoothing stage
    best_y_smoothed = y.copy()
    best_smooth_mse = np.inf

    bandwidth_factors = [0.5, 1.0, 1.5, 2.0, 3.0]
    for bf in bandwidth_factors:
        if time.time() > deadline - 20.0:
            break
        bw = bandwidth * bf
        # Compute kernel weights in chunks
        W = np.zeros((n, n))
        chunk_size = min(n, max(500, int(2e8 / (8 * n))))
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            diff = x[i_start:i_end, np.newaxis] - x[np.newaxis, :]
            u = diff / bw
            u = np.clip(u, -40.0, 40.0)
            W[i_start:i_end, :] = np.exp(-0.5 * u * u)

        # Weighted least squares: minimize sum_i w_i * (V_i @ c - y_i)^2
        # Equivalent to (V^T W V) c = V^T W y
        # For efficiency, use W_sqrt = W (diagonal of weights per point)
        # Actually, for NW-style: each point i has weight based on distance to all points
        # Better approach: compute smoothed values via local polynomial regression
        # Simplified: use the kernel matrix to compute effective weights

        # Use a simpler approach: for each evaluation point, compute weighted LS
        # But that's O(n^2 * 6^2). Instead, approximate with a global weighted fit
        # where each training point gets a weight based on its local density

        # Compute local density weights
        density = W.sum(axis=1)
        density_safe = np.where(density > 1e-12, density, 1.0)
        w_diag = 1.0 / density_safe  # Inverse density weighting

        # Weighted global polynomial fit
        A = (V * w_diag[:, np.newaxis]).T @ V
        b = (V * w_diag[:, np.newaxis]).T @ y
        try:
            c_smooth = np.linalg.solve(A + 1e-8 * np.eye(6), b)
        except np.linalg.LinAlgError:
            c_smooth = np.linalg.lstsq(A + 1e-8 * np.eye(6), b, rcond=None)[0]

        pred_smooth = V @ c_smooth
        # Measure how well the smooth curve fits the data (weighted)
        resid = pred_smooth - y
        mse_smooth = np.mean(w_diag * resid ** 2)
        if mse_smooth < best_smooth_mse:
            best_smooth_mse = mse_smooth
            best_y_smoothed = pred_smooth

    # --- Stage 2: Fit degree-5 polynomial to smoothed data with regularization ---
    A_full = V.T @ V
    b_full = V.T @ best_y_smoothed

    best_coeff_raw = np.zeros(6)
    best_mse = np.inf

    # Finer regularization grid
    lambdas = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5]
    for lam in lambdas:
        if time.time() > deadline - 10.0:
            break
        A_reg = A_full + lam * np.eye(6)
        try:
            c_scaled = np.linalg.solve(A_reg, b_full)
        except np.linalg.LinAlgError:
            c_scaled = np.linalg.lstsq(A_reg, b_full, rcond=None)[0]

        pred = V @ c_scaled
        mse = np.mean((pred - best_y_smoothed) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_coeff_raw = c_scaled

    # Final polish: very light regularization
    if time.time() < deadline - 5.0:
        for lam in [1e-10, 1e-9, 1e-8]:
            A_reg = A_full + lam * np.eye(6)
            try:
                c_scaled = np.linalg.solve(A_reg, b_full)
            except np.linalg.LinAlgError:
                continue
            pred = V @ c_scaled
            mse = np.mean((pred - best_y_smoothed) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_coeff_raw = c_scaled

    # Convert from scaled polynomial back to original x powers
    mu = x_mean
    sigma = x_std
    coeff = np.zeros(6)
    for k in range(6):
        for j in range(k, 6):
            binom_val = 1.0
            for m in range(k):
                binom_val *= (j - m) / (m + 1)
            coeff[k] += best_coeff_raw[j] * binom_val * ((-mu) ** (j - k)) / (sigma ** j)

    if not np.all(np.isfinite(coeff)):
        return np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
