import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.array([np.mean(y) if n > 0 else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Standardize x for better conditioning
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    xc = (x - mu) / sigma

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    best_score = np.inf
    best_coeffs_orig = np.zeros(6, dtype=np.float64)
    best_deg = 1
    best_alpha = 0.0

    # Precompute all powers of xc up to degree 5 (column k = xc^k)
    max_deg = 5
    xc_powers = np.ones((n, max_deg + 1), dtype=np.float64)
    for k in range(1, max_deg + 1):
        xc_powers[:, k] = xc_powers[:, k - 1] * xc

    def _fit_ridge(xc_powers_sub, y_sub, degree, alpha):
        """Fit ridge polynomial in centered-standardized basis."""
        V = xc_powers_sub[:, :degree + 1]
        if alpha > 0:
            A = V.T @ V
            b = V.T @ y_sub
            reg = alpha * np.eye(degree + 1)
            reg[0, 0] += 1e-12  # numerical safety
            try:
                c = np.linalg.solve(A + reg, b)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(V, y_sub, rcond=None)[0]
        else:
            try:
                c = np.linalg.lstsq(V, y_sub, rcond=None)[0]
            except Exception:
                c = np.zeros(degree + 1)
        return c

    def _cv_mse_fast(degree, alpha):
        """Fast 5-fold CV using precomputed powers."""
        n_samples = len(xc)
        if n_samples < (degree + 2) * 5:
            # Single split
            n_val = min(500, n_samples // 5)
            n_val = max(n_val, degree + 2)
            if n_val >= n_samples:
                return y_var
            rng = np.random.RandomState(42)
            idx = rng.permutation(n_samples)
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
            c = _fit_ridge(xc_powers[train_idx], y[train_idx], degree, alpha)
            pred = xc_powers[val_idx, :degree + 1] @ c
            return np.mean((pred - y[val_idx]) ** 2)
        else:
            K = 5
            rng = np.random.RandomState(42)
            idx = rng.permutation(n_samples)
            fold_size = n_samples // K
            mse_sum = 0.0
            count = 0
            for k in range(K):
                lo = k * fold_size
                hi = lo + fold_size
                if k == K - 1:
                    hi = n_samples
                val_idx = idx[lo:hi]
                train_idx = np.concatenate([idx[:lo], idx[hi:]])
                if len(train_idx) < degree + 2:
                    continue
                c = _fit_ridge(xc_powers[train_idx], y[train_idx], degree, alpha)
                pred = xc_powers[val_idx, :degree + 1] @ c
                mse_sum += np.mean((pred - y[val_idx]) ** 2)
                count += 1
            if count == 0:
                return y_var
            return mse_sum / count

    def _coeffs_in_original_basis(c_centered, degree):
        """Convert centered-standardized coefficients to original x basis."""
        coeffs_orig = np.zeros(6, dtype=np.float64)
        for i in range(degree + 1):
            ci = c_centered[i]
            if ci == 0.0:
                continue
            inv_si = (1.0 / sigma) ** i
            for k in range(i + 1):
                # binom(i, k)
                if k == 0 or k == i:
                    Ck = 1.0
                else:
                    mm = min(k, i - k)
                    num = 1
                    den = 1
                    for t in range(1, mm + 1):
                        num *= (i - mm + t)
                        den *= t
                    Ck = num / den
                coeffs_orig[k] += ci * Ck * ((-mu) ** (i - k)) * inv_si
        return coeffs_orig

    # Search grid
    degrees = [1, 2, 3, 4, 5]
    alphas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # First pass: coarse grid
    for deg in degrees:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break
            try:
                cv_mse = _cv_mse_fast(deg, alpha)
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    best_deg = deg
                    best_alpha = alpha
            except Exception:
                continue

    # Second pass: refine around the best degree with finer alphas
    if time.time() < deadline - 10.0:
        refined_alphas = np.logspace(-6, 3, 24)
        neighbor_degs = sorted(set([max(1, best_deg - 1), best_deg, min(5, best_deg + 1)]))
        for deg in neighbor_degs:
            if time.time() > deadline:
                break
            for alpha in refined_alphas:
                if time.time() > deadline:
                    break
                try:
                    cv_mse = _cv_mse_fast(deg, float(alpha))
                    score = cv_mse / y_var
                    if score < best_score:
                        best_score = score
                        best_deg = deg
                        best_alpha = float(alpha)
                except Exception:
                    continue

    # Third pass: very fine refinement if time remains
    if time.time() < deadline - 5.0:
        # Try even finer alpha grid around best_alpha
        if best_alpha > 0:
            local_alphas = best_alpha * np.logspace(-1.5, 1.5, 15)
        else:
            local_alphas = np.concatenate([[0.0], np.logspace(-7, -3, 12)])
        for deg in [best_deg]:
            for alpha in local_alphas:
                if time.time() > deadline:
                    break
                try:
                    cv_mse = _cv_mse_fast(deg, float(alpha))
                    score = cv_mse / y_var
                    if score < best_score:
                        best_score = score
                        best_deg = deg
                        best_alpha = float(alpha)
                except Exception:
                    continue

    # Final fit on all data with best settings
    c_centered = _fit_ridge(xc_powers, y, best_deg, best_alpha)

    # High-precision polish: if alpha was 0, try a tiny ridge to stabilize
    if best_alpha == 0.0:
        c_centered_tiny = _fit_ridge(xc_powers, y, best_deg, 1e-10)
        # Compare prediction quality
        pred_orig = xc_powers[:, :best_deg + 1] @ c_centered
        pred_tiny = xc_powers[:, :best_deg + 1] @ c_centered_tiny
        if np.mean((pred_tiny - y) ** 2) < np.mean((pred_orig - y) ** 2) * 1.001:
            c_centered = c_centered_tiny

    # Convert to original basis
    coeffs_orig = _coeffs_in_original_basis(c_centered, best_deg)

    # Ensure finite
    if not np.all(np.isfinite(coeffs_orig)):
        coeffs_orig = np.zeros(6, dtype=np.float64)
        coeffs_orig[0] = np.mean(y)

    return coeffs_orig
