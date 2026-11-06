"""Strategy: Binned-mean denoising with orthogonal polynomial fit and cross-validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 10:
        return np.zeros(6)

    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 10:
        return np.zeros(6)

    # Sort by x for efficient binning
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Determine x range
    x_min = x_sorted[0]
    x_max = x_sorted[-1]
    x_range = x_max - x_min
    if x_range < 1e-12:
        # Constant x, just return mean
        coeff = np.zeros(6)
        coeff[0] = np.mean(y_sorted)
        return coeff

    # Strategy: Bin the data to estimate E[y|x] locally, then fit polynomial.
    # This acts as a denoising step (low-pass filter) to remove fast-scale fluctuations.
    # Then use orthogonal polynomial basis (Legendre) for stable fitting.

    # Adaptive bin count
    # Aim for at least 20-50 points per bin if possible
    # sqrt(N) to N^(2/5) is common
    n_bins = max(50, min(200, int(np.sqrt(n))))

    # Create bin edges
    edges = np.linspace(x_min, x_max, n_bins + 1)

    # Assign points to bins
    bin_indices = np.digitize(x_sorted, edges) - 1

    # Filter out indices 0 or n_bins
    valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
    x_valid = x_sorted[valid_bins]
    y_valid = y_sorted[valid_bins]
    bin_indices_valid = bin_indices[valid_bins]

    # Compute bin centers and means
    sum_y = np.bincount(bin_indices_valid, weights=y_valid, minlength=n_bins)
    counts = np.bincount(bin_indices_valid, minlength=n_bins)

    non_empty = counts > 0
    if not np.any(non_empty):
        # Fallback
        return np.zeros(6)

    sum_x = np.bincount(bin_indices_valid, weights=x_valid, minlength=n_bins)
    bin_centers = np.zeros(n_bins)
    bin_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    bin_centers[non_empty] = sum_x[non_empty] / counts[non_empty]
    bin_means[non_empty] = sum_y[non_empty] / counts[non_empty]
    bin_counts[non_empty] = counts[non_empty]

    # Extract valid bin data for fitting
    fit_x = bin_centers[non_empty]
    fit_y = bin_means[non_empty]
    fit_counts = bin_counts[non_empty]

    # If we don't have enough points to fit degree 5, fall back
    if len(fit_x) < 6:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 5 - len(coeff)))
        except Exception:
            return np.zeros(6)

    # Cross-validation to select optimal number of bins
    # Try a few different bin counts and pick the best via CV
    candidate_bin_counts = []
    base_bins = n_bins
    for mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
        bc = int(base_bins * mult)
        bc = max(10, min(500, bc))
        if bc not in candidate_bin_counts:
            candidate_bin_counts.append(bc)

    best_coeff = None
    best_cv_error = np.inf
    best_n_bins = n_bins

    # Simple cross-validation: split into 5 folds
    n_folds = 5
    fold_size = n // n_folds

    for nb in candidate_bin_counts:
        if time.time() > deadline - 15:
            break

        # Bin the data with nb bins
        edges_cv = np.linspace(x_min, x_max, nb + 1)
        bin_idx_cv = np.digitize(x_sorted, edges_cv) - 1
        valid_cv = (bin_idx_cv >= 0) & (bin_idx_cv < nb)
        x_cv = x_sorted[valid_cv]
        y_cv = y_sorted[valid_cv]
        bin_idx_cv = bin_idx_cv[valid_cv]

        sum_y_cv = np.bincount(bin_idx_cv, weights=y_cv, minlength=nb)
        counts_cv = np.bincount(bin_idx_cv, minlength=nb)
        non_empty_cv = counts_cv > 0

        if not np.any(non_empty_cv):
            continue

        sum_x_cv = np.bincount(bin_idx_cv, weights=x_cv, minlength=nb)
        centers_cv = np.zeros(nb)
        means_cv = np.zeros(nb)
        counts_cv_arr = np.zeros(nb)

        centers_cv[non_empty_cv] = sum_x_cv[non_empty_cv] / counts_cv[non_empty_cv]
        means_cv[non_empty_cv] = sum_y_cv[non_empty_cv] / counts_cv[non_empty_cv]
        counts_cv_arr[non_empty_cv] = counts_cv[non_empty_cv]

        fx_cv = centers_cv[non_empty_cv]
        fy_cv = means_cv[non_empty_cv]
        fc_cv = counts_cv_arr[non_empty_cv]

        if len(fx_cv) < 6:
            continue

        # Fit polynomial using orthogonal basis
        # Scale x to [-1, 1] for Legendre polynomials
        x_center = (x_min + x_max) / 2.0
        x_scale = (x_max - x_min) / 2.0
        if x_scale < 1e-12:
            continue

        z = (fx_cv - x_center) / x_scale

        # Build Legendre polynomial design matrix
        # L_0(z) = 1
        # L_1(z) = z
        # L_2(z) = (3z^2 - 1)/2
        # L_3(z) = (5z^3 - 3z)/2
        # L_4(z) = (35z^4 - 30z^2 + 3)/8
        # L_5(z) = (63z^5 - 70z^3 + 15z)/8

        L = np.zeros((len(z), 6))
        L[:, 0] = 1.0
        L[:, 1] = z
        L[:, 2] = 0.5 * (3.0 * z**2 - 1.0)
        L[:, 3] = 0.5 * (5.0 * z**3 - 3.0 * z)
        L[:, 4] = (35.0 * z**4 - 30.0 * z**2 + 3.0) / 8.0
        L[:, 5] = (63.0 * z**5 - 70.0 * z**3 + 15.0 * z) / 8.0

        # Weighted least squares (weights = bin counts)
        W = np.diag(fc_cv)
        try:
            # Solve (L^T W L) a = L^T W y
            A = L.T @ (fc_cv[:, np.newaxis] * L)
            b_vec = L.T @ (fc_cv * fy_cv)

            # Add small regularization for stability
            lam = 1e-6 * np.trace(A) / 6.0
            A_reg = A + lam * np.eye(6)
            a_legendre = np.linalg.solve(A_reg, b_vec)
        except Exception:
            try:
                a_legendre = np.linalg.lstsq(L, fy_cv, rcond=None)[0]
            except Exception:
                continue

        if not np.all(np.isfinite(a_legendre)):
            continue

        # Convert Legendre coefficients to monomial coefficients
        # p(x) = sum_i a_i * L_i(z) where z = (x - x_center) / x_scale
        # We need to express this in terms of x^0, x^1, ..., x^5

        # L_0(z) = 1
        # L_1(z) = z = (x - c)/s
        # L_2(z) = (3z^2 - 1)/2
        # L_3(z) = (5z^3 - 3z)/2
        # L_4(z) = (35z^4 - 30z^2 + 3)/8
        # L_5(z) = (63z^5 - 70z^3 + 15z)/8

        # Let's build the transformation matrix T such that:
        # [1, x, x^2, x^3, x^4, x^5] @ T = [L_0(z), L_1(z), ..., L_5(z)]
        # Actually, we want: p(x) = sum_j c_j x^j = sum_i a_i L_i((x-c)/s)

        # Let's compute the monomial coefficients directly
        c = np.zeros(6)
        c[0] = a_legendre[0]

        # L_1(z) = (x - c)/s = x/s - c/s
        c[0] += a_legendre[1] * (-x_center / x_scale)
        c[1] += a_legendre[1] * (1.0 / x_scale)

        # L_2(z) = (3z^2 - 1)/2 = (3(x-c)^2/s^2 - 1)/2
        # = (3(x^2 - 2cx + c^2)/s^2 - 1)/2
        # = (3x^2/s^2 - 6cx/s^2 + 3c^2/s^2 - 1)/2
        # = (3x^2)/(2s^2) - (3cx)/s^2 + (3c^2)/(2s^2) - 1/2
        c[0] += a_legendre[2] * ((3.0 * x_center**2) / (2.0 * x_scale**2) - 0.5)
        c[1] += a_legendre[2] * (-3.0 * x_center / x_scale**2)
        c[2] += a_legendre[2] * (3.0 / (2.0 * x_scale**2))

        # L_3(z) = (5z^3 - 3z)/2
        # z^3 = (x-c)^3/s^3 = (x^3 - 3cx^2 + 3c^2x - c^3)/s^3
        # 5z^3 - 3z = 5(x^3 - 3cx^2 + 3c^2x - c^3)/s^3 - 3(x-c)/s
        # = (5x^3/s^3 - 15cx^2/s^3 + 15c^2x/s^3 - 5c^3/s^3) - (3x/s - 3c/s)
        # Divide by 2:
        c[0] += a_legendre[3] * 0.5 * (-5.0 * x_center**3 / x_scale**3 + 3.0 * x_center / x_scale)
        c[1] += a_legendre[3] * 0.5 * (15.0 * x_center**2 / x_scale**3 - 3.0 / x_scale)
        c[2] += a_legendre[3] * 0.5 * (-15.0 * x_center / x_scale**3)
        c[3] += a_legendre[3] * 0.5 * (5.0 / x_scale**3)

        # L_4(z) = (35z^4 - 30z^2 + 3)/8
        # z^4 = (x-c)^4/s^4 = (x^4 - 4cx^3 + 6c^2x^2 - 4c^3x + c^4)/s^4
        # 35z^4 - 30z^2 + 3
        # = 35(x^4/s^4 - 4cx^3/s^4 + 6c^2x^2/s^4 - 4c^3x/s^4 + c^4/s^4)
        #   - 30(x^2/s^2 - 2cx/s^2 + c^2/s^2)
        #   + 3
        # Divide by 8:
        c[0] += a_legendre[4] * 0.125 * (35.0 * x_center**4 / x_scale**4 - 30.0 * x_center**2 / x_scale**2 + 3.0)
        c[1] += a_legendre[4] * 0.125 * (-140.0 * x_center**3 / x_scale**4 + 60.0 * x_center / x_scale**2)
        c[2] += a_legendre[4] * 0.125 * (210.0 * x_center**2 / x_scale**4 - 30.0 / x_scale**2)
        c[3] += a_legendre[4] * 0.125 * (-140.0 * x_center / x_scale**4)
        c[4] += a_legendre[4] * 0.125 * (35.0 / x_scale**4)

        # L_5(z) = (63z^5 - 70z^3 + 15z)/8
        # z^5 = (x-c)^5/s^5 = (x^5 - 5cx^4 + 10c^2x^3 - 10c^3x^2 + 5c^4x - c^5)/s^5
        # 63z^5 - 70z^3 + 15z
        # = 63(x^5/s^5 - 5cx^4/s^5 + 10c^2x^3/s^5 - 10c^3x^2/s^5 + 5c^4x/s^5 - c^5/s^5)
        #   - 70(x^3/s^3 - 3cx^2/s^3 + 3c^2x/s^3 - c^3/s^3)
        #   + 15(x/s - c/s)
        # Divide by 8:
        c[0] += a_legendre[5] * 0.125 * (-63.0 * x_center**5 / x_scale**5 + 70.0 * x_center**3 / x_scale**3 - 15.0 * x_center / x_scale)
        c[1] += a_legendre[5] * 0.125 * (315.0 * x_center**4 / x_scale**5 - 210.0 * x_center**2 / x_scale**3 + 15.0 / x_scale)
        c[2] += a_legendre[5] * 0.125 * (-630.0 * x_center**3 / x_scale**5 + 210.0 * x_center / x_scale**3)
        c[3] += a_legendre[5] * 0.125 * (630.0 * x_center**2 / x_scale**5 - 70.0 / x_scale**3)
        c[4] += a_legendre[5] * 0.125 * (-315.0 * x_center / x_scale**5)
        c[5] += a_legendre[5] * 0.125 * (63.0 / x_scale**5)

        if not np.all(np.isfinite(c)):
            continue

        # Cross-validate: evaluate polynomial on held-out data
        cv_errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n

            # Train on all except validation fold
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_start:val_end] = False

            x_train = x_sorted[train_mask]
            y_train = y_sorted[train_mask]
            x_val = x_sorted[val_start:val_end]
            y_val = y_sorted[val_start:val_end]

            if len(x_train) < 20 or len(x_val) < 5:
                continue

            # Bin training data
            edges_train = np.linspace(x_train.min(), x_train.max(), nb + 1)
            bin_idx_train = np.digitize(x_train, edges_train) - 1
            valid_train = (bin_idx_train >= 0) & (bin_idx_train < nb)
            x_tr = x_train[valid_train]
            y_tr = y_train[valid_train]
            bin_idx_tr = bin_idx_train[valid_train]

            sum_y_tr = np.bincount(bin_idx_tr, weights=y_tr, minlength=nb)
            counts_tr = np.bincount(bin_idx_tr, minlength=nb)
            non_empty_tr = counts_tr > 0

            if not np.any(non_empty_tr):
                continue

            sum_x_tr = np.bincount(bin_idx_tr, weights=x_tr, minlength=nb)
            centers_tr = np.zeros(nb)
            means_tr = np.zeros(nb)
            counts_tr_arr = np.zeros(nb)

            centers_tr[non_empty_tr] = sum_x_tr[non_empty_tr] / counts_tr[non_empty_tr]
            means_tr[non_empty_tr] = sum_y_tr[non_empty_tr] / counts_tr[non_empty_tr]
            counts_tr_arr[non_empty_tr] = counts_tr[non_empty_tr]

            fx_tr = centers_tr[non_empty_tr]
            fy_tr = means_tr[non_empty_tr]
            fc_tr = counts_tr_arr[non_empty_tr]

            if len(fx_tr) < 6:
                continue

            # Fit
            z_tr = (fx_tr - x_center) / x_scale
            L_tr = np.zeros((len(z_tr), 6))
            L_tr[:, 0] = 1.0
            L_tr[:, 1] = z_tr
            L_tr[:, 2] = 0.5 * (3.0 * z_tr**2 - 1.0)
            L_tr[:, 3] = 0.5 * (5.0 * z_tr**3 - 3.0 * z_tr)
            L_tr[:, 4] = (35.0 * z_tr**4 - 30.0 * z_tr**2 + 3.0) / 8.0
            L_tr[:, 5] = (63.0 * z_tr**5 - 70.0 * z_tr**3 + 15.0 * z_tr) / 8.0

            try:
                A_tr = L_tr.T @ (fc_tr[:, np.newaxis] * L_tr)
                b_tr = L_tr.T @ (fc_tr * fy_tr)
                lam_tr = 1e-6 * np.trace(A_tr) / 6.0
                A_tr_reg = A_tr + lam_tr * np.eye(6)
                a_tr = np.linalg.solve(A_tr_reg, b_tr)
            except Exception:
                try:
                    a_tr = np.linalg.lstsq(L_tr, fy_tr, rcond=None)[0]
                except Exception:
                    continue

            if not np.all(np.isfinite(a_tr)):
                continue

            # Convert to monomial
            c_tr = np.zeros(6)
            c_tr[0] = a_tr[0]
            c_tr[0] += a_tr[1] * (-x_center / x_scale)
            c_tr[1] += a_tr[1] * (1.0 / x_scale)

            c_tr[0] += a_tr[2] * ((3.0 * x_center**2) / (2.0 * x_scale**2) - 0.5)
            c_tr[1] += a_tr[2] * (-3.0 * x_center / x_scale**2)
            c_tr[2] += a_tr[2] * (3.0 / (2.0 * x_scale**2))

            c_tr[0] += a_tr[3] * 0.5 * (-5.0 * x_center**3 / x_scale**3 + 3.0 * x_center / x_scale)
            c_tr[1] += a_tr[3] * 0.5 * (15.0 * x_center**2 / x_scale**3 - 3.0 / x_scale)
            c_tr[2] += a_tr[3] * 0.5 * (-15.0 * x_center / x_scale**3)
            c_tr[3] += a_tr[3] * 0.5 * (5.0 / x_scale**3)

            c_tr[0] += a_tr[4] * 0.125 * (35.0 * x_center**4 / x_scale**4 - 30.0 * x_center**2 / x_scale**2 + 3.0)
            c_tr[1] += a_tr[4] * 0.125 * (-140.0 * x_center**3 / x_scale**4 + 60.0 * x_center / x_scale**2)
            c_tr[2] += a_tr[4] * 0.125 * (210.0 * x_center**2 / x_scale**4 - 30.0 / x_scale**2)
            c_tr[3] += a_tr[4] * 0.125 * (-140.0 * x_center / x_scale**4)
            c_tr[4] += a_tr[4] * 0.125 * (35.0 / x_scale**4)

            c_tr[0] += a_tr[5] * 0.125 * (-63.0 * x_center**5 / x_scale**5 + 70.0 * x_center**3 / x_scale**3 - 15.0 * x_center / x_scale)
            c_tr[1] += a_tr[5] * 0.125 * (315.0 * x_center**4 / x_scale**5 - 210.0 * x_center**2 / x_scale**3 + 15.0 / x_scale)
            c_tr[2] += a_tr[5] * 0.125 * (-630.0 * x_center**3 / x_scale**5 + 210.0 * x_center / x_scale**3)
            c_tr[3] += a_tr[5] * 0.125 * (630.0 * x_center**2 / x_scale**5 - 70.0 / x_scale**3)
            c_tr[4] += a_tr[5] * 0.125 * (-315.0 * x_center / x_scale**5)
            c_tr[5] += a_tr[5] * 0.125 * (63.0 / x_scale**5)

            if not np.all(np.isfinite(c_tr)):
                continue

            # Evaluate on validation set
            pred = np.polynomial.polynomial.polyval(x_val, c_tr)
            mse = np.mean((pred - y_val) ** 2)
            cv_errors.append(mse)

        if cv_errors:
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                best_coeff = c
                best_n_bins = nb

    if time.time() > deadline - 5:
        if best_coeff is not None:
            return best_coeff
        # Fallback: simple linear fit
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            return np.zeros(6)

    if best_coeff is not None:
        return best_coeff

    # Fallback: direct polynomial fit with regularization
    try:
        V = np.vander(x, 6, increasing=True)
        lam = 1e-4
        A = V.T @ V + lam * np.eye(6)
        b_vec = V.T @ y
        coeff = np.linalg.solve(A, b_vec)
        return coeff
    except Exception:
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            return np.zeros(6)

# EVOLVE-BLOCK-END
