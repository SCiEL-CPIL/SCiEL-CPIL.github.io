"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with adaptive bandwidth."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Compute a robust scale for bandwidth
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad < 1e-12:
        mad = np.std(x)
    if mad < 1e-12:
        mad = 1.0

    # Try multiple bandwidths and pick the one with best cross-validation
    # Use Gaussian kernel Nadaraya-Watson to denoise y, then fit polynomial

    def nw_denoise(x, y, bandwidth):
        """Nadaraya-Watson kernel regression with Gaussian kernel."""
        n = len(x)
        y_hat = np.zeros(n)
        # Process in chunks to avoid memory issues
        chunk_size = min(n, 5000)
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            # Compute weights for points i..end-1 against all x
            diff = x[i:end, None] - x[None, :]
            weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
            # Avoid division by zero
            weight_sums = weights.sum(axis=1)
            weight_sums[weight_sums < 1e-30] = 1e-30
            y_hat[i:end] = (weights * y[None, :]).sum(axis=1) / weight_sums
        return y_hat

    def fit_poly(x, y, degree):
        """Fit polynomial of given degree."""
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, degree)
            if not np.all(np.isfinite(coeff)):
                return None
            return coeff
        except Exception:
            return None

    def mse_poly(x, y, coeff):
        """Compute MSE of polynomial prediction."""
        pred = np.polynomial.polynomial.polyval(x, coeff)
        return np.mean((y - pred) ** 2)

    # Candidate bandwidths (in units of MAD)
    bandwidth_multipliers = [0.5, 1.0, 2.0, 3.0, 5.0]
    bandwidths = [b * mad for b in bandwidth_multipliers]

    best_coeff = None
    best_score = np.inf

    # Also consider no denoising (bandwidth = inf, i.e., just fit directly)
    all_bandwidths = bandwidths + [np.inf]

    for bw in all_bandwidths:
        if time.time() > deadline:
            break

        if np.isinf(bw):
            # No denoising, just fit directly
            y_denoised = y
        else:
            y_denoised = nw_denoise(x, y, bw)

        # Fit degree-5 polynomial
        coeff = fit_poly(x, y_denoised, 5)
        if coeff is None:
            continue

        # Evaluate on training data (proxy for validation)
        score = mse_poly(x, y, coeff)
        if score < best_score:
            best_score = score
            best_coeff = coeff

    # If no good result, fall back to direct degree-5 fit
    if best_coeff is None:
        best_coeff = fit_poly(x, y, 5)
        if best_coeff is None:
            # Fall back to lower degree
            for deg in range(4, 0, -1):
                best_coeff = fit_poly(x, y, deg)
                if best_coeff is not None:
                    # Pad to length 6
                    if len(best_coeff) < 6:
                        padded = np.zeros(6)
                        padded[:len(best_coeff)] = best_coeff
                        best_coeff = padded
                    break
            if best_coeff is None:
                best_coeff = np.zeros(6)

    # Ensure exactly 6 coefficients
    if len(best_coeff) < 6:
        padded = np.zeros(6)
        padded[:len(best_coeff)] = best_coeff
        best_coeff = padded
    elif len(best_coeff) > 6:
        best_coeff = best_coeff[:6]

    # Final check for finiteness
    if not np.all(np.isfinite(best_coeff)):
        return np.zeros(6)

    return best_coeff
# EVOLVE-BLOCK-END
