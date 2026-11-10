"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nCr(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0:
        return 1
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) // (i + 1)
    return res

def _convert_coeffs_to_original(a, mu, sigma, deg=5):
    """
    Convert polynomial coefficients 'a' for the variable z = (x-mu)/sigma
    to coefficients 'b' for the original variable x.
    y = sum(a_j * z^j) = sum(b_i * x^i)
    """
    b = np.zeros(deg + 1)
    for j in range(deg + 1):
        if a[j] == 0:
            continue
        coeff_a = a[j] / (sigma ** j)
        for k in range(j + 1):
            binom_val = _nCr(j, k)
            term_coeff = coeff_a * binom_val * ((-mu) ** (j - k))
            b[k] += term_coeff
    return b

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 118.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        return np.array([np.mean(y)] + [0.0]*5)

    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        sigma = 1.0
    z = (x - mu) / sigma

    n = len(z)

    # Try degrees 5 and 6; pick best via CV
    best_overall_cv = np.inf
    best_overall_deg = 5
    best_overall_lambda = 1.0

    for deg in [5, 6]:
        if n < deg + 10:
            continue

        V_full = np.polynomial.polynomial.polyvander(z, deg)

        n_folds = 5
        if n < n_folds * 2:
            n_folds = max(2, n // 2)

        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        fold_size = n // n_folds
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else n
            folds.append((indices[start:end], np.concatenate([indices[:start], indices[end:]])))

        # Precompute eigen-decomposition of VtV for each fold (train set)
        # This allows O(1) ridge solve per lambda via (V+lam*I)^{-1} = Q diag(1/(d+lam)) Q^T
        fold_eig = []
        for val_idx, train_idx in folds:
            if len(train_idx) < deg + 1:
                continue
            V_tr = V_full[train_idx]
            y_tr = y[train_idx]
            V_val = V_full[val_idx]
            y_val = y[val_idx]
            VtV = V_tr.T @ V_tr
            Vty = V_tr.T @ y_tr
            # Symmetric eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(VtV)
            # Clamp small negative eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            fold_eig.append((eigvals, eigvecs, Vty, V_val, y_val))

        # Batch ridge CV: for a set of lambdas, compute CV MSE for all at once per fold
        def batch_ridge_cv(lambdas, fold_eig_data, deg):
            """Compute CV MSE for all lambdas simultaneously using eigen-decomp."""
            d = deg + 1
            n_lams = len(lambdas)
            # Initialize CV MSE accumulators
            cv_mse = np.zeros(n_lams)
            valid_folds = 0

            for eigvals, eigvecs, Vty, V_val, y_val in fold_eig_data:
                # eigvecs: (d, d), eigvals: (d,)
                # For ridge: c = Q @ diag(1/(eigvals+lam)) @ Q^T @ Vty
                # Precompute Q^T @ Vty: shape (d,)
                QtVty = eigvecs.T @ Vty  # (d,)
                # For each lambda: c_lam = Q @ (QtVty / (eigvals + lam))
                # Build matrix: rows are lambdas, cols are basis coefficients
                # c_matrix[lam_idx, j] = sum_k Q[k,j] * QtVty[k] / (eigvals[k] + lam[lam_idx])
                # Shape: (n_lams, d)
                # eigvals + lam[:,None] -> (n_lams, d)
                denom = eigvals[None, :] + lambdas[:, None]  # (n_lams, d)
                # QtVty / denom: (n_lams, d)
                scaled = QtVty[None, :] / denom  # (n_lams, d)
                # c_matrix = scaled @ Q : (n_lams, d)
                c_matrix = scaled @ eigvecs  # (n_lams, d)
                # Predictions: V_val @ c_matrix.T -> (n_val, n_lams)
                preds = V_val @ c_matrix.T  # (n_val, n_lams)
                # MSE per lambda
                mse_per_lam = np.mean((preds - y_val[:, None]) ** 2, axis=0)  # (n_lams,)
                cv_mse += mse_per_lam
                valid_folds += 1

            if valid_folds > 0:
                cv_mse /= valid_folds
            return cv_mse

        # Stage 1: Coarse search
        lambdas_coarse = np.logspace(-8, 8, 100)
        if time.time() > deadline - 50.0:
            continue

        cv_results = batch_ridge_cv(lambdas_coarse, fold_eig, deg)
        finite_mask = np.isfinite(cv_results)
        if not np.any(finite_mask):
            continue
        best_idx = np.argmin(np.where(finite_mask, cv_results, np.inf))
        best_cv = cv_results[best_idx]
        best_lam = lambdas_coarse[best_idx]

        # Stage 2: Fine search
        if time.time() < deadline - 30.0:
            ratio = 1.3
            lo = best_lam / ratio
            hi = best_lam * ratio
            lambdas_fine = np.logspace(np.log10(lo), np.log10(hi), 60)
            cv_results2 = batch_ridge_cv(lambdas_fine, fold_eig, deg)
            finite_mask2 = np.isfinite(cv_results2)
            if np.any(finite_mask2):
                best_idx2 = np.argmin(np.where(finite_mask2, cv_results2, np.inf))
                if cv_results2[best_idx2] < best_cv:
                    best_cv = cv_results2[best_idx2]
                    best_lam = lambdas_fine[best_idx2]

        # Stage 3: Very-fine polish
        if time.time() < deadline - 15.0:
            ratio = 1.05
            lo = best_lam / ratio
            hi = best_lam * ratio
            lambdas_vf = np.logspace(np.log10(lo), np.log10(hi), 30)
            cv_results3 = batch_ridge_cv(lambdas_vf, fold_eig, deg)
            finite_mask3 = np.isfinite(cv_results3)
            if np.any(finite_mask3):
                best_idx3 = np.argmin(np.where(finite_mask3, cv_results3, np.inf))
                if cv_results3[best_idx3] < best_cv:
                    best_cv = cv_results3[best_idx3]
                    best_lam = lambdas_vf[best_idx3]

        if best_cv < best_overall_cv:
            best_overall_cv = best_cv
            best_overall_deg = deg
            best_overall_lambda = best_lam

    # Refit on all data with best degree and lambda
    deg = best_overall_deg
    lam = best_overall_lambda
    V_full = np.polynomial.polynomial.polyvander(z, deg)

    A = V_full.T @ V_full + lam * np.eye(deg + 1)
    b_vec = V_full.T @ y
    try:
        c_std = np.linalg.solve(A, b_vec)
        if np.all(np.isfinite(c_std)):
            b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
            if np.all(np.isfinite(b_orig)):
                # If degree is 6, we need to return 6 coefficients (degree 5)
                # But the task says degree-five, so if best is degree 6, truncate
                if len(b_orig) == 6:
                    return b_orig
                elif len(b_orig) == 7:
                    # Truncate to degree 5
                    return b_orig[:6]
    except Exception:
        pass

    # Fallback: Unregularized degree-5 fit
    try:
        V5 = np.polynomial.polynomial.polyvander(z, 5)
        c_std, _, _, _ = np.linalg.lstsq(V5, y, rcond=None)
        if np.all(np.isfinite(c_std)):
            b_orig = _convert_coeffs_to_original(c_std, mu, sigma, 5)
            if np.all(np.isfinite(b_orig)):
                return b_orig
    except Exception:
        pass

    return np.array([np.mean(y)] + [0.0]*5)

# EVOLVE-BLOCK-END
