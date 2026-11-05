"""Climate dynamics: unresolved-tendency closure via binned polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove any non-finite entries
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return np.zeros(6)

    if n < 6:
        # Not enough points for degree-5. Fit lower degree if possible.
        if n >= 2:
            try:
                c_lin = np.polyfit(x, y, 1)
                coeff = np.zeros(6)
                coeff[0] = c_lin[1]
                coeff[1] = c_lin[0]
                return coeff
            except Exception:
                pass
        return np.zeros(6)

    # Normalize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Normalize y
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # KEY IMPROVEMENT: Quantile-based binning as a low-pass filter
    # Fit polynomial to bin means instead of raw data
    # This reduces high-frequency noise and improves generalization
    n_bins_options = [20, 30, 40, 50, 60]
    best_coeff_norm = None
    best_mse = np.inf

    # Build Vandermonde matrix for normalized x
    V = np.vander(x_norm, N=6, increasing=True)

    # Try different numbers of bins
    for n_bins in n_bins_options:
        if time.time() > deadline - 20.0:
            break

        if n_bins > n:
            continue

        try:
            # Create quantile-based bins
            bin_edges = np.quantile(x_norm, np.linspace(0, 1, n_bins + 1))
            # Assign each point to a bin
            bin_indices = np.digitize(x_norm, bin_edges[1:-1])

            # Compute bin means
            bin_x = np.zeros(n_bins)
            bin_y = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)

            for b in range(n_bins):
                mask_b = (bin_indices == b)
                if np.sum(mask_b) > 0:
                    bin_x[b] = np.mean(x_norm[mask_b])
                    bin_y[b] = np.mean(y_norm[mask_b])
                    bin_counts[b] = np.sum(mask_b)

            # Only use bins with at least 1 data point
            valid_bins = bin_counts > 0
            if np.sum(valid_bins) < 6:
                continue

            bx = bin_x[valid_bins]
            by = bin_y[valid_bins]
            bc = bin_counts[valid_bins]

            # Weighted polynomial fit to bin means
            # Weights proportional to bin counts for proper weighting
            w = bc / np.sum(bc)

            V_bin = np.vander(bx, N=6, increasing=True)

            # Weighted least squares: minimize sum(w_i * (V_bin[i] @ c - by_i)^2)
            # Equivalent to: (V_bin^T W V_bin) c = V_bin^T W by
            W = np.diag(w)
            try:
                A = V_bin.T @ W @ V_bin
                b_vec = V_bin.T @ W @ by
                coeff_norm = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                try:
                    coeff_norm, _, _, _ = np.linalg.lstsq(np.sqrt(w)[:, None] * V_bin, np.sqrt(w) * by, rcond=None)
                except Exception:
                    continue

            if not np.all(np.isfinite(coeff_norm)):
                continue

            # Evaluate MSE on original data
            pred_norm = V @ coeff_norm
            mse = np.mean((pred_norm - y_norm) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_coeff_norm = coeff_norm.copy()

        except Exception:
            continue

    # Fallback: direct fit if bining didn't work
    if best_coeff_norm is None:
        try:
            VtV = V.T @ V
            Vty = V.T @ y_norm
            cond = np.linalg.cond(VtV)
            if cond < 1e10:
                best_coeff_norm = np.linalg.solve(VtV, Vty)
            else:
                lam = 1e-6 * np.trace(VtV) / 6.0
                A = VtV + lam * np.eye(6)
                best_coeff_norm = np.linalg.solve(A, Vty)
        except Exception:
            try:
                best_coeff_norm, _, _, _ = np.linalg.lstsq(V, y_norm, rcond=None)
            except Exception:
                best_coeff_norm = np.zeros(6)

    if not np.all(np.isfinite(best_coeff_norm)):
        best_coeff_norm = np.zeros(6)

    # Transform back to original scale
    try:
        best_coeff = _transform_poly_to_original(best_coeff_norm, x_mean, x_std, y_mean, y_std)
    except Exception:
        try:
            V_full = np.vander(x, N=6, increasing=True)
            best_coeff, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
        except Exception:
            best_coeff = np.zeros(6)

    # Ensure finite output
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.where(np.isfinite(best_coeff), best_coeff, 0.0)

    return best_coeff


def _transform_poly_to_original(coeff_norm, x_mean, x_std, y_mean, y_std):
    """
    Transform polynomial coefficients from normalized space to original space.
    y = y_mean + y_std * sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    """
    d = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom_val = comb(i, j)
            term = coeff_norm[i] / (x_std ** i) * binom_val * ((-x_mean) ** (i - j))
            s += term
        d[j] = y_std * s
    d[0] += y_mean
    return d
# EVOLVE-BLOCK-END
