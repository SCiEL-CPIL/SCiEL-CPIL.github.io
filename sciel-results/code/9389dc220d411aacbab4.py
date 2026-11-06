"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()

    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]

    if len(x) == 0:
        return np.zeros(6)

    # Scale x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
    x_scaled = (x - mu) / sigma

    # Build Vandermonde in scaled basis: [1, xs, xs^2, ..., xs^5]
    n = len(x)
    V = np.vander(x_scaled, N=6, increasing=True)

    # Ridge regression with multiple alpha values, pick best via validation-like split
    # Use a simple train/test split for alpha selection
    if n > 100:
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        split = int(0.8 * n)
        train_idx = idx[:split]
        val_idx = idx[split:]
        V_train = V[train_idx]
        t_train = t[train_idx]
        V_val = V[val_idx]
        t_val = t[val_idx]
    else:
        train_idx = np.arange(n)
        val_idx = train_idx
        V_train = V
        t_train = t
        V_val = V
        t_val = t

    alphas = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_coeff_scaled = None
    best_val_mse = np.inf

    for alpha in alphas:
        if time.time() > DEADLINE:
            break
        if alpha == 0.0:
            # OLS
            coeff_scaled, _, _, _ = np.linalg.lstsq(V_train, t_train, rcond=None)
        else:
            # Ridge: (V^T V + alpha*I) c = V^T t
            A = V_train.T @ V_train + alpha * np.eye(6)
            b = V_train.T @ t_train
            try:
                coeff_scaled = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff_scaled = np.linalg.lstsq(A, b, rcond=None)[0]

        # Evaluate on validation
        pred_val = V_val @ coeff_scaled
        val_mse = np.mean((pred_val - t_val) ** 2)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_coeff_scaled = coeff_scaled.copy()

    if best_coeff_scaled is None:
        # Fallback: linear fit
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, t, 1)[::-1]
        return coeff

    # Now refit on ALL data with the best alpha to get final coefficients
    # Determine which alpha was best
    best_alpha = None
    for alpha in alphas:
        if time.time() > DEADLINE:
            break
        if alpha == 0.0:
            coeff_test, _, _, _ = np.linalg.lstsq(V_train, t_train, rcond=None)
        else:
            A = V_train.T @ V_train + alpha * np.eye(6)
            b = V_train.T @ t_train
            try:
                coeff_test = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff_test = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.allclose(coeff_test, best_coeff_scaled):
            best_alpha = alpha
            break

    # Refit on all data
    if best_alpha == 0.0:
        coeff_scaled_all, _, _, _ = np.linalg.lstsq(V, t, rcond=None)
    else:
        A_all = V.T @ V + best_alpha * np.eye(6)
        b_all = V.T @ t
        try:
            coeff_scaled_all = np.linalg.solve(A_all, b_all)
        except np.linalg.LinAlgError:
            coeff_scaled_all = np.linalg.lstsq(A_all, b_all, rcond=None)[0]

    # Convert from scaled basis to original x basis
    # x_scaled = (x - mu) / sigma
    # So x = sigma * x_scaled + mu
    # We have p(x_scaled) = sum_i c_i * x_scaled^i
    # We need q(x) = p((x-mu)/sigma) = sum_i c_i * ((x-mu)/sigma)^i
    # Expand (x-mu)^i / sigma^i using binomial theorem
    coeff_orig = np.zeros(6)
    for i in range(6):
        c_i = coeff_scaled_all[i] / (sigma ** i)
        # (x - mu)^i = sum_{j=0}^{i} C(i,j) x^j (-mu)^(i-j)
        for j in range(i + 1):
            binom = 1.0
            # Compute C(i, j)
            if j == 0 or j == i:
                binom = 1.0
            else:
                # Simple binomial
                num = 1
                for k in range(min(j, i - j)):
                    num *= (i - k)
                den = 1
                for k in range(1, min(j, i - j) + 1):
                    den *= k
                binom = num / den
            coeff_orig[j] += c_i * binom * ((-mu) ** (i - j))

    # Ensure finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff_orig
# EVOLVE-BLOCK-END
