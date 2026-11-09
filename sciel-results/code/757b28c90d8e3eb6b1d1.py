import numpy as np
import time
import math

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 105.0  # Safety margin before 120s limit

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

    # Subsample for efficiency
    M = min(n, 3000)
    if n > M:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, M, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y
        M = n

    # Scaling for polynomial fit stability
    mu = np.mean(x_sub)
    sigma = np.std(x_sub)
    if sigma < 1e-6:
        sigma = 1.0
    z_sub = (x_sub - mu) / sigma

    # Kernel smoothing to denoise: y_smooth(x) = sum(K((x-x_i)/h)*y_i) / sum(K((x-x_i)/h))
    # Select bandwidth via CV, and also select polynomial degree via CV on smoothed data

    # Silverman's rule
    h_base = 1.06 * sigma * (M ** (-0.2))
    h_candidates = h_base * np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0])

    # CV folds
    k_folds = 5
    rng_cv = np.random.default_rng(123)
    indices_cv = rng_cv.permutation(M)
    fold_size = M // k_folds

    # First, select best bandwidth using CV with a fixed degree (degree 3 as intermediate)
    best_h = h_base
    best_h_cv_mse = np.inf

    for h in h_candidates:
        if time.time() > deadline:
            break
        total_mse = 0.0
        count = 0
        for i in range(k_folds):
            val_idx = indices_cv[i*fold_size : (i+1)*fold_size]
            train_idx = np.concatenate([indices_cv[:i*fold_size], indices_cv[(i+1)*fold_size:]])

            x_train = x_sub[train_idx]
            y_train = y_sub[train_idx]
            x_val = x_sub[val_idx]
            y_val = y_sub[val_idx]

            # Kernel smoothing for validation points
            diff = (x_val[:, None] - x_train[None, :]) / h
            weights = np.exp(-0.5 * diff**2)
            denom = np.sum(weights, axis=1)
            denom[denom < 1e-12] = 1e-12
            numer = np.sum(weights * y_train[None, :], axis=1)
            y_pred = numer / denom

            mse = np.mean((y_val - y_pred)**2)
            total_mse += mse
            count += 1

        if count > 0:
            avg_mse = total_mse / count
            if avg_mse < best_h_cv_mse:
                best_h_cv_mse = avg_mse
                best_h = h

    # Now smooth the entire subsample with best_h
    diff = (x_sub[:, None] - x_sub[None, :]) / best_h
    weights = np.exp(-0.5 * diff**2)
    denom = np.sum(weights, axis=1)
    denom[denom < 1e-12] = 1e-12
    numer = np.sum(weights * y_sub[None, :], axis=1)
    y_smooth = numer / denom

    # Select polynomial degree via CV on smoothed data
    # Try degrees 0 through 5, with light Ridge regularization
    best_degree = 5
    best_degree_cv_mse = np.inf
    best_coeff_scaled = np.zeros(6)

    for degree in range(6):
        if time.time() > deadline:
            break

        total_mse = 0.0
        count = 0
        for i in range(k_folds):
            val_idx = indices_cv[i*fold_size : (i+1)*fold_size]
            train_idx = np.concatenate([indices_cv[:i*fold_size], indices_cv[(i+1)*fold_size:]])

            z_train = z_sub[train_idx]
            y_train = y_smooth[train_idx]
            z_val = z_sub[val_idx]
            y_val = y_smooth[val_idx]

            # Fit with Ridge regularization (small lambda for stability)
            V_train = np.vander(z_train, degree + 1, increasing=True)
            # Light ridge: add small regularization to improve conditioning
            lam = 1e-6
            A = V_train.T @ V_train + lam * np.eye(degree + 1)
            b = V_train.T @ y_train
            try:
                coeff_deg = np.linalg.solve(A, b)
            except:
                coeff_deg, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)

            V_val = np.vander(z_val, degree + 1, increasing=True)
            y_pred = V_val @ coeff_deg

            mse = np.mean((y_val - y_pred)**2)
            total_mse += mse
            count += 1

        if count > 0:
            avg_mse = total_mse / count
            if avg_mse < best_degree_cv_mse:
                best_degree_cv_mse = avg_mse
                best_degree = degree
                # Store coefficients from full-data fit (will re-fit below)
                best_coeff_scaled = coeff_deg  # temporary

    # Final fit on all smoothed data with best degree
    if time.time() > deadline:
        # Return what we have
        c = np.zeros(6)
        for k in range(min(best_degree + 1, 6)):
            if best_coeff_scaled[k] == 0:
                continue
            coeff_k = best_coeff_scaled[k] / (sigma ** k)
            for j in range(k + 1):
                binom_val = math.comb(k, j)
                pow_mu = (-mu) ** (k - j)
                c[j] += coeff_k * binom_val * pow_mu
        if not np.all(np.isfinite(c)):
            c = np.zeros(6)
        return c

    V_full = np.vander(z_sub, best_degree + 1, increasing=True)
    # Use ridge with very small lambda for final fit
    lam = 1e-8
    A = V_full.T @ V_full + lam * np.eye(best_degree + 1)
    b = V_full.T @ y_smooth
    try:
        a = np.linalg.solve(A, b)
    except:
        a, _, _, _ = np.linalg.lstsq(V_full, y_smooth, rcond=None)

    # Pad to length 6
    if len(a) < 6:
        a = np.pad(a, (0, 6 - len(a)))

    # Convert from scaled basis z=(x-mu)/sigma to original basis x
    c = np.zeros(6)
    for k in range(best_degree + 1):
        if a[k] == 0:
            continue
        coeff_k = a[k] / (sigma ** k)
        for j in range(k + 1):
            binom_val = math.comb(k, j)
            pow_mu = (-mu) ** (k - j)
            c[j] += coeff_k * binom_val * pow_mu

    # Ensure finite
    if not np.all(np.isfinite(c)):
        c = np.zeros(6)

    return c
