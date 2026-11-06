"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved fast-scale forcing.
    Uses Nadaraya-Watson denoising + Huber-robust iterative reweighting
    with orthogonal polynomial basis for numerical stability.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 6:
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)[::-1]
        return coeff

    # Scale x to [-1, 1] for stable Legendre polynomial evaluation
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)

    mid = 0.5 * (x_max + x_min)
    half = 0.5 * x_range
    x_scaled = (x - mid) / half

    # Build Legendre polynomial basis up to degree 5
    t = x_scaled
    P = np.zeros((n, 6))
    P[:, 0] = 1.0
    P[:, 1] = t
    P[:, 2] = 0.5 * (3.0 * t**2 - 1.0)
    P[:, 3] = 0.5 * (5.0 * t**3 - 3.0 * t)
    P[:, 4] = 0.125 * (35.0 * t**4 - 30.0 * t**2 + 3.0)
    P[:, 5] = 0.125 * (63.0 * t**5 - 70.0 * t**3 + 15.0 * t)

    # Step 1: Nadaraya-Watson denoising to get smoother target
    # Use a bandwidth based on the data spread with small grid search
    y_smooth = _nadaraya_watson_search(x, y, n)

    # Step 2: Initial least-squares fit in Legendre basis on smoothed data
    a_leg, _, _, _ = np.linalg.lstsq(P, y_smooth, rcond=None)

    # Step 3: Huber-robust iterative reweighting (IRLS)
    # Compute residuals from initial fit
    y_pred = P @ a_leg
    residuals = y_smooth - y_pred
    sigma = np.std(residuals)
    if sigma < 1e-12:
        sigma = 1.0

    huber_delta = 1.5 * sigma  # Huber threshold
    n_irls = 8  # number of IRLS iterations

    for it in range(n_irls):
        if time.time() > deadline:
            break
        # Compute Huber weights
        abs_res = np.abs(residuals)
        w = np.where(abs_res <= huber_delta, 1.0, huber_delta / abs_res)
        # Ensure weights are positive and finite
        w = np.clip(w, 1e-6, 1.0)

        # Weighted least squares in Legendre basis
        sqrt_w = np.sqrt(w)
        P_w = P * sqrt_w[:, np.newaxis]
        y_w = y_smooth * sqrt_w

        a_leg, _, _, _ = np.linalg.lstsq(P_w, y_w, rcond=None)

        # Update residuals
        y_pred = P @ a_leg
        residuals = y_smooth - y_pred
        # Update sigma
        new_sigma = np.std(residuals)
        if new_sigma > 1e-12:
            sigma = new_sigma
            huber_delta = 1.5 * sigma
    
    # Final high-precision polish: one more unweighted least squares to refine
    a_leg, _, _, _ = np.linalg.lstsq(P, y_smooth, rcond=None)

    # Convert from Legendre basis to monomial coefficients
    c = _legendre_to_monomial(a_leg, mid, half)

    # Ensure all coefficients are finite
    for i in range(6):
        if not np.isfinite(c[i]):
            c[i] = 0.0

    # Fallback check: if the fit is terrible, try direct polyfit
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        x_test = x[:min(200, n)]
        y_pred_check = np.zeros(len(x_test))
        for i in range(6):
            y_pred_check += c[i] * (x_test ** i)
        mse_check = np.mean((y_pred_check - y_smooth[:min(200, n)])**2)
        if not np.isfinite(mse_check) or mse_check > 1e10:
            try:
                coeff_direct = np.polynomial.polynomial.polyfit(x, y_smooth, 5)
                if np.all(np.isfinite(coeff_direct)):
                    c = coeff_direct
            except Exception:
                pass

    return c


def _nadaraya_watson_search(x, y, n):
    """
    Nadaraya-Watson kernel regression with bandwidth selection via small grid search.
    Uses Gaussian kernel with bandwidths around Silverman's rule.
    """
    # Silverman's rule for bandwidth
    sigma_x = np.std(x)
    if sigma_x < 1e-12:
        sigma_x = 1.0
    h_base = 1.06 * sigma_x * (n ** (-1.0/5.0))
    h_base = max(h_base, 1e-6)
    
    # Try a few bandwidth multipliers around Silverman's estimate
    multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    best_h = h_base
    best_mse = np.inf
    
    # Use a subset for evaluation to save time
    eval_idx = np.random.RandomState(42).choice(n, size=min(5000, n), replace=False)
    x_eval = x[eval_idx]
    y_eval = y[eval_idx]
    
    for mult in multipliers:
        h_bw = h_base * mult
        y_smooth_eval = _nadaraya_watson_single(x, y, h_bw, eval_idx)
        # Use degree-5 polynomial fit MSE on the smoothed data as proxy
        try:
            coeff = np.polyfit(x_eval, y_smooth_eval, 5)
            y_pred = np.polyval(coeff, x_eval)
            mse = np.mean((y_pred - y_smooth_eval)**2)
            if np.isfinite(mse) and mse < best_mse:
                best_mse = mse
                best_h = h_bw
        except Exception:
            pass
    
    return _nadaraya_watson_single(x, y, best_h, None)


def _nadaraya_watson_single(x, y, h_bw, eval_idx=None):
    """
    Nadaraya-Watson kernel regression with given bandwidth.
    If eval_idx is not None, only compute for those indices.
    """
    n = len(x)
    if eval_idx is None:
        eval_idx = np.arange(n)
    
    n_eval = len(eval_idx)
    
    # For efficiency with large n, use a binned approach or direct computation
    if n > 50000:
        # Use a coarser grid approach
        n_grid = 500
        grid = np.linspace(np.min(x), np.max(x), n_grid)
        # Compute weights on the grid
        y_grid = np.zeros(n_grid)
        for i in range(n_grid):
            dist = np.abs(x - grid[i])
            w_kernel = np.exp(-0.5 * (dist / h_bw)**2)
            w_sum = np.sum(w_kernel)
            if w_sum > 1e-12:
                y_grid[i] = np.sum(w_kernel * y) / w_sum
            else:
                y_grid[i] = np.mean(y)
        # Interpolate back to original x
        y_full = np.interp(x, grid, y_grid)
        return y_full[eval_idx]
    else:
        # Direct computation
        y_smooth = np.zeros(n_eval)
        # Process in chunks to avoid memory issues
        chunk_size = 5000
        for start in range(0, n_eval, chunk_size):
            end = min(start + chunk_size, n_eval)
            idx_chunk = eval_idx[start:end]
            x_chunk = x[idx_chunk, np.newaxis]
            dist = np.abs(x_chunk - x[np.newaxis, :])
            w_kernel = np.exp(-0.5 * (dist / h_bw)**2)
            w_sum = np.sum(w_kernel, axis=1)
            for i in range(end - start):
                if w_sum[i] > 1e-12:
                    y_smooth[start + i] = np.sum(w_kernel[i] * y) / w_sum[i]
                else:
                    y_smooth[start + i] = np.mean(y)
        return y_smooth


def _legendre_to_monomial(a_leg, mid, half):
    """
    Convert Legendre polynomial coefficients (on [-1,1]) to monomial coefficients
    in powers of the original x.
    
    t = (x - mid) / half
    P_k(t) expanded in powers of t, then t expressed in terms of x.
    """
    s = 1.0 / half

    # Coefficients of Legendre polynomials in powers of t:
    b = np.zeros((6, 6))
    b[0, 0] = 1.0
    b[1, 1] = 1.0
    b[2, 0] = -0.5
    b[2, 2] = 1.5
    b[3, 1] = -1.5
    b[3, 3] = 2.5
    b[4, 0] = 3.0/8.0
    b[4, 2] = -30.0/8.0
    b[4, 4] = 35.0/8.0
    b[5, 1] = 15.0/8.0
    b[5, 3] = -70.0/8.0
    b[5, 5] = 63.0/8.0

    # Compute monomial coefficients c_i for i=0..5
    c = np.zeros(6)
    for i in range(6):
        total = 0.0
        for k in range(6):
            if a_leg[k] == 0.0:
                continue
            for j in range(i, 6):
                if b[k, j] != 0.0:
                    # binom(j, i) * (-mid)^(j-i) * s^j * b[k,j] * a_leg[k]
                    total += a_leg[k] * b[k, j] * (s ** j) * _binom(j, i) * ((-mid) ** (j - i))
        c[i] = total

    return c


def _binom(n, k):
    """Compute binomial coefficient C(n, k) for small integers."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
# EVOLVE-BLOCK-END
