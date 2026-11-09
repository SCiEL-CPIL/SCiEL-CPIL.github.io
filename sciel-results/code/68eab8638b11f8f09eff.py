"""Rewritten program: GPR denoising + degree-5 polynomial fit, with CV-based degree selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s (evaluator kills at 150s but we want 120s budget)

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

    # Standardize x
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0

    x_norm = (x - x_mean) / x_std

    # --- GPR denoising ---
    # Use an RBF kernel to estimate E[Y|X] = mu(x), then fit polynomial to mu(x)
    # This filters out the high-variance noise in the fast-scale tendency

    def gpr_predict(x_train, y_train, x_eval, length_scale, sigma_f, sigma_n):
        """Gaussian Process prediction with RBF kernel."""
        # Compute kernel matrix
        diff = x_train[:, None] - x_train[None, :]
        K = sigma_f ** 2 * np.exp(-0.5 * (diff ** 2) / (length_scale ** 2))
        K += sigma_n ** 2 * np.eye(len(x_train))

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter
            K += 1e-8 * np.eye(len(x_train))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback: return mean
                return np.full_like(x_eval, float(np.mean(y_train)))

        # Alpha
        try:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            return np.full_like(x_eval, float(np.mean(y_train)))

        # Predict
        diff_eval = x_eval[:, None] - x_train[None, :]
        K_s = sigma_f ** 2 * np.exp(-0.5 * (diff_eval ** 2) / (length_scale ** 2))

        mu = K_s @ alpha

        return mu

    def gpr_denoise(x_fit, y_fit, length_scale, sigma_f, sigma_n, n_eval=500):
        """Denoise y using GPR by predicting at a grid of points."""
        # Evaluate at a dense grid covering the range of x
        x_min = np.min(x_fit)
        x_max = np.max(x_fit)
        margin = 0.5 * (x_max - x_min) if x_max > x_min else 1.0
        x_grid = np.linspace(x_min - margin, x_max + margin, n_eval)

        mu = gpr_predict(x_fit, y_fit, x_grid, length_scale, sigma_f, sigma_n)

        # Interpolate back to original x points
        y_denoised = np.interp(x_fit, x_grid, mu)

        return y_denoised

    # Try different GPR hyperparameters and pick the best via CV
    # Then fit polynomial to denoised data

    def fit_ridge(x_fit, y_fit, degree, lam):
        """Ridge regression in normalized space."""
        V = np.vander(x_fit, N=degree + 1, increasing=True)
        A = V.T @ V + lam * np.eye(degree + 1)
        b = V.T @ y_fit
        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                coeff = np.zeros(degree + 1)
        return coeff

    def poly_eval(coeff, x_eval):
        pred = np.zeros_like(x_eval)
        for i in range(len(coeff)):
            pred += coeff[i] * (x_eval ** i)
        return pred

    def mse(pred, target):
        return np.mean((pred - target) ** 2)

    # CV setup
    k_folds = 5
    np.random.seed(123)
    indices = np.random.permutation(n)
    if n < k_folds:
        k_folds = max(2, n)
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[:n % k_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    # GPR hyperparameters to try
    length_scales = [0.3, 0.5, 0.8, 1.2, 2.0]
    sigma_f_values = [1.0, 2.0]
    sigma_n_values = [0.1, 0.5, 1.0]

    # For efficiency, we'll do a coarse search over GPR params
    # and for each, do CV to evaluate denoising quality + polynomial fit

    degrees_to_try = [5, 4, 3]
    lambdas_to_try = [1e-4, 1e-3, 1e-2, 0.1, 1.0]

    best_score = np.inf
    best_config = None  # (length_scale, sigma_f, sigma_n, degree, lam)

    # Coarse GPR search
    for ls in length_scales:
        if time.time() - t_start > deadline - 30:
            break
        for sf in sigma_f_values:
            if time.time() - t_start > deadline - 30:
                break
            for sn in sigma_n_values:
                if time.time() - t_start > deadline - 30:
                    break

                # Quick CV: use first fold as validation to estimate denoising quality
                val_idx = folds[0]
                train_idx = np.concatenate([folds[j] for j in range(1, k_folds)])

                x_tr = x_norm[train_idx]
                y_tr = y[train_idx]
                x_va = x_norm[val_idx]
                y_va = y[val_idx]

                # Denoise training data
                y_tr_denoised = gpr_denoise(x_tr, y_tr, ls, sf, sn, n_eval=300)

                # Fit polynomial to denoised data
                for degree in degrees_to_try:
                    for lam in lambdas_to_try:
                        coeff = fit_ridge(x_tr, y_tr_denoised, degree, lam)
                        if not np.all(np.isfinite(coeff)):
                            continue
                        pred = poly_eval(coeff, x_va)
                        score = mse(pred, y_va)

                        if score < best_score:
                            best_score = score
                            best_config = (ls, sf, sn, degree, lam)

    if best_config is None:
        # Fallback: no GPR, just ridge
        best_config = (1.0, 1.0, 1.0, 5, 1e-3)

    ls_best, sf_best, sn_best, deg_best, lam_best = best_config

    # Refinement: finer GPR search around best
    if time.time() - t_start < deadline - 20:
        ls_refine = [ls_best * f for f in [0.6, 0.8, 1.0, 1.25, 1.6]]
        sf_refine = [sf_best * f for f in [0.7, 1.0, 1.4]]
        sn_refine = [sn_best * f for f in [0.5, 1.0, 2.0]]

        current_best = best_score
        current_config = best_config

        for ls in ls_refine:
            if time.time() - t_start > deadline - 15:
                break
            for sf in sf_refine:
                if time.time() - t_start > deadline - 15:
                    break
                for sn in sn_refine:
                    if time.time() - t_start > deadline - 15:
                        break

                    # Full CV
                    cv_mse_sum = 0.0
                    valid = True
                    for i in range(k_folds):
                        val_idx = folds[i]
                        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

                        x_tr = x_norm[train_idx]
                        y_tr = y[train_idx]
                        x_va = x_norm[val_idx]
                        y_va = y[val_idx]

                        y_tr_denoised = gpr_denoise(x_tr, y_tr, ls, sf, sn, n_eval=300)

                        coeff = fit_ridge(x_tr, y_tr_denoised, deg_best, lam_best)
                        if not np.all(np.isfinite(coeff)):
                            valid = False
                            break
                        pred = poly_eval(coeff, x_va)
                        cv_mse_sum += mse(pred, y_va)

                    if not valid:
                        continue
                    avg_cv = cv_mse_sum / k_folds
                    if avg_cv < current_best:
                        current_best = avg_cv
                        current_config = (ls, sf, sn, deg_best, lam_best)

        if current_best < best_score:
            best_score = current_best
            best_config = current_config

    ls_best, sf_best, sn_best, deg_best, lam_best = best_config

    # Also try different degrees with the best GPR config
    if time.time() - t_start < deadline - 10:
        for degree in [5, 4, 3, 2]:
            if time.time() - t_start > deadline - 10:
                break
            cv_mse_sum = 0.0
            valid = True
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

                x_tr = x_norm[train_idx]
                y_tr = y[train_idx]
                x_va = x_norm[val_idx]
                y_va = y[val_idx]

                y_tr_denoised = gpr_denoise(x_tr, y_tr, ls_best, sf_best, sn_best, n_eval=300)

                coeff = fit_ridge(x_tr, y_tr_denoised, degree, lam_best)
                if not np.all(np.isfinite(coeff)):
                    valid = False
                    break
                pred = poly_eval(coeff, x_va)
                cv_mse_sum += mse(pred, y_va)

            if not valid:
                continue
            avg_cv = cv_mse_sum / k_folds
            if avg_cv < best_score:
                best_score = avg_cv
                deg_best = degree

    # Final fit: denoise ALL training data, then fit polynomial
    y_denoised_all = gpr_denoise(x_norm, y, ls_best, sf_best, sn_best, n_eval=500)

    # Try a few lambda values for the final fit and pick best via a quick holdout
    # Use last 10% as a quick validation
    split_idx = int(0.9 * n)
    x_final_train = x_norm[:split_idx]
    y_final_train = y_denoised_all[:split_idx]
    x_final_val = x_norm[split_idx:]
    y_final_val = y[split_idx:]  # Use original y for validation

    best_final_coeff = None
    best_final_mse = np.inf
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        if time.time() - t_start > deadline - 5:
            break
        coeff = fit_ridge(x_final_train, y_final_train, deg_best, lam)
        if not np.all(np.isfinite(coeff)):
            continue
        pred = poly_eval(coeff, x_final_val)
        m = mse(pred, y_final_val)
        if m < best_final_mse:
            best_final_mse = m
            best_final_coeff = coeff

    if best_final_coeff is None:
        best_final_coeff = fit_ridge(x_norm, y_denoised_all, deg_best, lam_best)

    # If time permits, refit on ALL data with the best lambda
    if time.time() - t_start < deadline - 2:
        # Find the lambda that gave best_final_mse
        # We didn't track it, so just use lam_best or try a few
        for lam in [lam_best, 1e-4, 1e-3, 1e-2]:
            coeff_candidate = fit_ridge(x_norm, y_denoised_all, deg_best, lam)
            if np.all(np.isfinite(coeff_candidate)):
                best_final_coeff = coeff_candidate
                break

    if not np.all(np.isfinite(best_final_coeff)):
        best_final_coeff = fit_ridge(x_norm, y, 5, 1.0)

    # Pad to 6 coefficients
    if len(best_final_coeff) < 6:
        temp = np.zeros(6)
        temp[:len(best_final_coeff)] = best_final_coeff
        best_final_coeff = temp

    # Convert from normalized to original x scale
    max_deg = min(5, len(best_final_coeff) - 1)
    coeff_orig = np.zeros(6, dtype=np.float64)

    for i in range(max_deg + 1):
        if abs(best_final_coeff[i]) < 1e-15:
            continue

        ci_over_std_pow = best_final_coeff[i] / (x_std ** i)

        # Expand (x - x_mean)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
        for j in range(i + 1):
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)

            term = binom_val * ((-x_mean) ** (i - j)) * ci_over_std_pow
            coeff_orig[j] += term

    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff_orig
# EVOLVE-BLOCK-END
