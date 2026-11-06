"""Tikhonov-regularized degree-five polynomial closure with CV-selected lambda."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_derivative_matrix(powers, x):
    """Build matrix of derivatives of x^i for i in powers, evaluated at x.
    Returns D where D[k, i] = d^k/dx^k (x^i) for each i in powers.
    We use up to 2nd derivatives for smoothness penalty."""
    n = len(x)
    n_powers = len(powers)
    # We'll build a combined matrix for 0th, 1st, 2nd derivatives
    # Shape: (3*n, n_powers)
    D = np.zeros((3 * n, n_powers))
    for idx, i in enumerate(powers):
        # 0th derivative: x^i
        D[0:n, idx] = np.power(x, i)
        # 1st derivative: i * x^(i-1)
        if i >= 1:
            D[n:2*n, idx] = i * np.power(x, i - 1)
        # 2nd derivative: i*(i-1) * x^(i-2)
        if i >= 2:
            D[2*n:3*n, idx] = i * (i - 1) * np.power(x, i - 2)
    return D


def _fit_regularized(V, y, lam, lam_deriv=0.0, x=None):
    """Solve (V^T V + lam*I + lam_deriv * D^T D) c = V^T y for polynomial coefficients."""
    A = V.T @ V + lam * np.eye(V.shape[1])
    b = V.T @ y
    if lam_deriv > 0 and x is not None:
        D = _build_derivative_matrix(np.arange(6), x)
        A += lam_deriv * (D.T @ D)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _predict(V, c):
    return V @ c


def _transform_coeffs(coeff_norm, mu, sig):
    """Transform coefficients from normalized x to original x.
    
    If x_norm = (x - mu)/sig, then:
    sum_{i=0}^{5} c_i * x_norm^i = sum_{j=0}^{5} d_j * x^j
    
    where d_j = sum_{i=j}^{5} c_i * C(i,j) * (-mu)^(i-j) / sig^i
    """
    coeff = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            binom_ij = comb(i, j)
            s += coeff_norm[i] * binom_ij * ((-mu) ** (i - j)) / (sig ** i)
        coeff[j] = s
    return coeff


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(training_x)
    if n < 10:
        # Fallback: just fit a linear polynomial
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        return coeff

    # Normalize x to improve conditioning (standardize)
    x_mean = np.mean(training_x)
    x_std = np.std(training_x)
    if x_std < 1e-12:
        x_norm = np.zeros_like(training_x)
    else:
        x_norm = (training_x - x_mean) / x_std

    powers = np.arange(6)
    V_norm = np.power(x_norm[:, None], powers[None, :])

    # Precompute derivative matrix for smoothness penalty
    D_norm = _build_derivative_matrix(powers, x_norm)
    DTD = D_norm.T @ D_norm

    # K-fold cross-validation to select lambda and lambda_deriv
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_sizes = np.array_split(indices, n_folds)

    # Candidate lambdas: log-spaced from small to large
    # Use trace(V^T V)/n as a natural scale for the regularization
    trace_vtv = np.trace(V_norm.T @ V_norm)
    lam_scale = trace_vtv / n if trace_vtv > 0 else 1.0
    
    # Also consider the magnitude of y for scaling
    y_std = np.std(training_subgrid_tendency)
    if y_std < 1e-12:
        y_std = 1.0
    
    # Derivative regularization scale
    trace_dtd = np.trace(DTD)
    lam_deriv_scale = trace_dtd / n if trace_dtd > 0 else 1.0

    # Lambda candidates should be comparable to the scale of the Gram matrix
    # We want lam to range from near-zero to something that dominates the Gram matrix
    lambda_candidates = np.concatenate([
        [0.0],
        np.logspace(-8, 2, 30) * lam_scale
    ])
    
    # Derivative lambda candidates (smaller range, smoother penalty)
    lambda_deriv_candidates = np.concatenate([
        [0.0],
        np.logspace(-10, 0, 15) * lam_deriv_scale
    ])

    best_lam = 0.0
    best_lam_deriv = 0.0
    best_cv_score = np.inf

    # First pass: select lambda with lam_deriv=0
    for lam in lambda_candidates:
        if _time_remaining() < 3.0:
            break
        mse_sum = 0.0
        for fold in range(n_folds):
            val_idx = fold_sizes[fold]
            train_idx = np.concatenate([fold_sizes[i] for i in range(n_folds) if i != fold])

            V_train = V_norm[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            V_val = V_norm[val_idx]
            y_val = training_subgrid_tendency[val_idx]

            c = _fit_regularized(V_train, y_train, lam, lam_deriv=0.0, x=None)
            pred = _predict(V_val, c)
            mse_sum += np.mean((pred - y_val) ** 2)

        cv_score = mse_sum / n_folds
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lam = lam

    # Second pass: select lambda_deriv with best lambda
    if _time_remaining() > 5.0:
        best_lam_deriv_score = np.inf
        for lam_d in lambda_deriv_candidates:
            if _time_remaining() < 2.0:
                break
            mse_sum = 0.0
            for fold in range(n_folds):
                val_idx = fold_sizes[fold]
                train_idx = np.concatenate([fold_sizes[i] for i in range(n_folds) if i != fold])

                V_train = V_norm[train_idx]
                y_train = training_subgrid_tendency[train_idx]
                V_val = V_norm[val_idx]
                y_val = training_subgrid_tendency[val_idx]
                x_train = x_norm[train_idx]

                c = _fit_regularized(V_train, y_train, best_lam, lam_deriv=lam_d, x=x_train)
                pred = _predict(V_val, c)
                mse_sum += np.mean((pred - y_val) ** 2)

            cv_score = mse_sum / n_folds
            if cv_score < best_lam_deriv_score:
                best_lam_deriv_score = cv_score
                best_lam_deriv = lam_d

    # Final fit on all training data with best lambdas
    coeff_norm = _fit_regularized(V_norm, training_subgrid_tendency, best_lam, 
                                   lam_deriv=best_lam_deriv, x=x_norm)

    # High-precision polish: try nearby lambdas for fine-tuning
    if _time_remaining() > 5.0:
        # Create a finer grid around the best lambda
        if best_lam > 0:
            polish_lams = np.logspace(np.log10(best_lam) - 0.5, np.log10(best_lam) + 0.5, 15)
        else:
            polish_lams = np.linspace(0, lam_scale * 0.1, 15)
        
        # Also include 0 and the original best
        polish_lams = np.unique(np.concatenate([[0.0], [best_lam], polish_lams]))
        
        # Derivative lambda polish
        if best_lam_deriv > 0:
            polish_lam_ds = np.logspace(np.log10(best_lam_deriv) - 0.5, 
                                         np.log10(best_lam_deriv) + 0.5, 10)
        else:
            polish_lam_ds = np.linspace(0, lam_deriv_scale * 0.1, 10)
        polish_lam_ds = np.unique(np.concatenate([[0.0], [best_lam_deriv], polish_lam_ds]))
        
        best_polish_score = np.inf
        best_polish_coeff = coeff_norm
        
        for plam in polish_lams:
            if _time_remaining() < 2.0:
                break
            for plam_d in polish_lam_ds:
                if _time_remaining() < 1.0:
                    break
                c = _fit_regularized(V_norm, training_subgrid_tendency, plam,
                                      lam_deriv=plam_d, x=x_norm)
                # Use a combination of training MSE and coefficient norm as proxy
                pred = _predict(V_norm, c)
                train_mse = np.mean((pred - training_subgrid_tendency) ** 2)
                # The regularization penalty is already implicitly in the solution,
                # but we can use a smooth score that balances fit and smoothness
                coeff_norm_sq = np.sum(c ** 2)
                # Add derivative penalty to the score
                Dc = D_norm @ c
                deriv_norm_sq = np.sum(Dc ** 2)
                score = train_mse + plam * coeff_norm_sq / (n * y_std ** 2 + 1e-10) + \
                        plam_d * deriv_norm_sq / (n * y_std ** 2 + 1e-10)
                
                if score < best_polish_score:
                    best_polish_score = score
                    best_polish_coeff = c
        
        coeff_norm = best_polish_coeff

    # Transform coefficients from normalized x to original x
    if x_std < 1e-12:
        coeff = np.zeros(6)
        coeff[0] = np.mean(training_subgrid_tendency)
        return coeff

    mu = x_mean
    sig = x_std
    coeff = _transform_coeffs(coeff_norm, mu, sig)

    # Sanity check: ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        except:
            pass
        if not np.all(np.isfinite(coeff)):
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
