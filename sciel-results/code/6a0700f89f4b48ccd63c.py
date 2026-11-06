import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Nadaraya-Watson smoothing to denoise the target ---
    # Estimate bandwidth using MAD
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad < 1e-12:
        mad = np.std(x)
    if mad < 1e-12:
        mad = 1.0
    # Silverman-like bandwidth scaled for the data range
    bw = 1.06 * max(mad, np.std(x) * 0.9) * n ** (-0.2)
    bw = max(bw, (x.max() - x.min()) / 20.0)  # ensure at least 20 bins worth

    # Evaluate NW kernel at each training point to get smoothed y
    # Use Gaussian kernel: K(u) = exp(-0.5 * u^2)
    # To avoid O(n^2) for very large n, subsample if needed
    if n > 50000:
        # Subsample for bandwidth evaluation, but use full data for regression
        idx = np.random.default_rng(42).choice(n, size=50000, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Compute smoothed values using subsampled points
    # Process in chunks to manage memory
    chunk_size = 5000
    y_smoothed = np.empty(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x[start:end]
        # u = (x_sub - x_chunk[:, None]) / bw
        u = (x_sub[None, :] - x_chunk[:, None]) / bw
        weights = np.exp(-0.5 * u * u)
        w_sum = np.sum(weights, axis=1)
        w_sum = np.where(w_sum < 1e-300, 1e-300, w_sum)
        y_smoothed[start:end] = np.sum(weights * y_sub[None, :], axis=1) / w_sum

    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: linear fit on smoothed data
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y_smoothed, 1)
        except Exception:
            coeff[1] = np.mean(y_smoothed) / np.mean(x) if np.mean(x) != 0 else 0
        return coeff

    # --- Degree-5 polynomial fit on smoothed data using Legendre basis for stability ---
    # Scale x to [-1, 1] for numerical stability in Legendre fit
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    if x_range < 1e-12:
        # Degenerate case: all x are the same
        c0 = np.mean(y_smoothed)
        return np.array([c0, 0.0, 0.0, 0.0, 0.0, 0.0])

    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Initial Legendre polynomial fit of degree 5
    leg_coeffs = np.polynomial.legendre.legfit(x_scaled, y_smoothed, 5)

    # Convert Legendre coefficients to standard monomial coefficients in scaled variable u
    monomial_u = np.polynomial.legendre.leg2poly(leg_coeffs)

    # Build transformation from scaled monomial to original monomial
    A = 1.0 / (x_range / 2.0)
    B = -(x_min + x_max) / 2.0 / (x_range / 2.0)

    powers = []
    current_power = np.array([1.0])
    for i in range(6):
        powers.append(current_power.copy())
        if i < 5:
            new_len = len(current_power) + 1
            new_power = np.zeros(new_len)
            new_power[0] = current_power[0] * B
            for k in range(1, len(current_power)):
                new_power[k] = current_power[k - 1] * A + current_power[k] * B
            new_power[-1] = current_power[-1] * A
            current_power = new_power

    c_x = np.zeros(6)
    for i in range(6):
        p = powers[i]
        padded_p = np.zeros(6)
        padded_p[:len(p)] = p
        c_x += monomial_u[i] * padded_p

    # --- Robust iterative reweighting with Huber loss ---
    # Compute residuals from initial fit
    def eval_poly(coeffs, x_val):
        """Evaluate polynomial sum(c_i * x^i) at x_val."""
        result = np.zeros_like(x_val)
        for i in range(len(coeffs)):
            result += coeffs[i] * (x_val ** i)
        return result

    residuals = y_smoothed - eval_poly(c_x, x)
    
    # Huber delta: use 1.5 * MAD of residuals (robust scale estimate)
    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    if mad_res < 1e-12:
        mad_res = np.std(residuals)
    if mad_res < 1e-12:
        mad_res = 1.0
    huber_delta = 1.5 * 1.4826 * mad_res  # 1.4826 converts MAD to std-equivalent

    # Perform iterative Huber reweighting (IRLS)
    n_iter_max = 5
    for iteration in range(n_iter_max):
        if time.time() > deadline - 15.0:
            break
        
        # Compute Huber weights
        abs_res = np.abs(residuals)
        huber_weights = np.ones(n)
        large_mask = abs_res > huber_delta
        huber_weights[large_mask] = huber_delta / abs_res[large_mask]
        
        # Weighted polynomial fit using weighted least squares
        # Build Vandermonde matrix V where V[i,j] = x[i]^j
        V = np.vander(x, N=6, increasing=True)
        
        # Weighted least squares: solve (V^T W V) c = V^T W y
        # Use sqrt(weights) for stability
        sqrt_w = np.sqrt(huber_weights)
        V_w = V * sqrt_w[:, None]
        y_w = y_smoothed * sqrt_w
        
        try:
            # Solve using lstsq for numerical stability
            c_new, _, _, _ = np.linalg.lstsq(V_w, y_w, rcond=None)
        except Exception:
            break
        
        if not np.all(np.isfinite(c_new)):
            break
            
        c_x = c_new
        residuals = y_smoothed - eval_poly(c_x, x)
        
        # Check convergence
        if iteration > 0:
            delta_c = np.max(np.abs(c_new - c_x)) if hasattr(c_x, '__len__') else 0
            # Actually we already updated c_x, so let's track differently
            pass

    # Check time
    if time.time() > deadline:
        # Fallback to initial fit if somehow slow
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y_smoothed, 1)
        except Exception:
            coeff[0] = np.mean(y_smoothed)
        return coeff

    # Ensure finite coefficients
    c_x = np.where(np.isfinite(c_x), c_x, 0.0)

    return c_x
