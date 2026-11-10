# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb

def _fit_ridge(V, y, alpha):
    A = V.T @ V
    b = V.T @ y
    reg = alpha * np.eye(A.shape[0])
    reg[0, 0] += 1e-12
    try:
        c = np.linalg.solve(A + reg, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A + reg, b, rcond=None)[0]
    return c

def _convert_to_original_basis(c_centered, mu, sigma, powers, degree=5):
    coeff_orig = np.zeros(degree + 1)
    for idx, power in enumerate(powers):
        ci = c_centered[idx] / (sigma ** power)
        for j in range(power + 1):
            binom_coeff = comb(power, j)
            term = ci * binom_coeff * ((-mu) ** (power - j))
            coeff_orig[j] += term
    return coeff_orig

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 130.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 6:
        return np.zeros(6)

    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0

    degree = 5
    xc = (x - mu) / sigma

    V_full = np.column_stack([xc**k for k in range(degree + 1)])
    V_odd = np.column_stack([xc**k for k in [1, 3, 5]])

    # 5-fold CV to select alpha and model form
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_folds = 5
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        folds.append(indices[start:end])

    alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

    def cv_select(V, alpha_list):
        best_alpha = alpha_list[0]
        best_cv_mse = np.inf
        for alpha in alpha_list:
            if time.time() > deadline:
                break
            cv_errors = []
            for fold_idx in range(len(folds)):
                val_idx = folds[fold_idx]
                train_idx = np.concatenate([folds[i] for i in range(len(folds)) if i != fold_idx])
                V_train = V[train_idx]
                y_train = y[train_idx]
                V_val = V[val_idx]
                y_val = y[val_idx]

                c = _fit_ridge(V_train, y_train, alpha)
                pred_val = V_val @ c
                mse = np.mean((y_val - pred_val) ** 2)
                cv_errors.append(mse)
            avg_mse = np.mean(cv_errors)
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_alpha = alpha
        return best_alpha, best_cv_mse

    alpha_full, cv_mse_full = cv_select(V_full, alphas)
    alpha_odd, cv_mse_odd = cv_select(V_odd, alphas)

    c_full = _fit_ridge(V_full, y, alpha_full)
    coeff_full_orig = _convert_to_original_basis(c_full, mu, sigma, list(range(degree + 1)), degree)

    c_odd = _fit_ridge(V_odd, y, alpha_odd)
    coeff_odd_orig = _convert_to_original_basis(c_odd, mu, sigma, [1, 3, 5], degree)

    if cv_mse_full <= cv_mse_odd:
        best_coeff_orig = coeff_full_orig
    else:
        best_coeff_orig = coeff_odd_orig

    if not np.all(np.isfinite(best_coeff_orig)):
        try:
            c_lin = np.polyfit(x, y, 1)
            best_coeff_orig = np.zeros(6)
            best_coeff_orig[0] = c_lin[1]
            best_coeff_orig[1] = c_lin[0]
        except:
            best_coeff_orig = np.zeros(6)

    return best_coeff_orig
# EVOLVE-BLOCK-END
