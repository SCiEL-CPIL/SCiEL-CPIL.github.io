"""Symbolic regression: adaptive kernel + Chebyshev + rational + Fourier with LOO-CV."""
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 325.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _fit_rbf(X, y, sigma):
    """Fit Gaussian RBF model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    reg = 1e-6

    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2.0 * sigma**2))
    K_reg = K + reg * np.eye(n)

    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        diff_new = xn[:, None] - x[None, :]
        K_new = np.exp(-diff_new**2 / (2.0 * sigma**2))
        return K_new @ w

    return predict


def _fit_chebyshev(X, y, deg):
    """Fit Chebyshev polynomial model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0

    x_norm = (2.0 * x - x_min - x_max) / (x_max - x_min)

    T = np.zeros((len(x), deg + 1))
    T[:, 0] = 1.0
    if deg >= 1:
        T[:, 1] = x_norm
        for k in range(2, deg + 1):
            T[:, k] = 2.0 * x_norm * T[:, k - 1] - T[:, k - 2]

    try:
        w = np.linalg.lstsq(T, y, rcond=None)[0]
    except Exception:
        w = np.zeros(deg + 1)

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_norm = (2.0 * xn - x_min - x_max) / (x_max - x_min)
        Tn = np.zeros((len(xn), deg + 1))
        Tn[:, 0] = 1.0
        if deg >= 1:
            Tn[:, 1] = xn_norm
            for k in range(2, deg + 1):
                Tn[:, k] = 2.0 * xn_norm * Tn[:, k - 1] - Tn[:, k - 2]
        return Tn @ w

    return predict


def _fit_fourier(X, y, n_terms):
    """Fit Fourier series model."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    basis = [np.ones(n)]
    for k in range(1, n_terms + 1):
        freq = 2.0 * np.pi * k / x_range
        basis.append(np.sin(freq * x))
        basis.append(np.cos(freq * x))

    F = np.column_stack(basis)

    try:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    except Exception:
        w = np.zeros(F.shape[1])

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        basis_new = [np.ones(len(xn))]
        for k in range(1, n_terms + 1):
            freq = 2.0 * np.pi * k / x_range
            basis_new.append(np.sin(freq * xn))
            basis_new.append(np.cos(freq * xn))
        Fn = np.column_stack(basis_new)
        return Fn @ w

    return predict


def _fit_spline(X, y, s_factor):
    """Fit B-spline model."""
    from scipy.interpolate import UnivariateSpline

    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    s = max(s_factor * np.var(y) * n, 1e-10)

    try:
        sp = UnivariateSpline(x_sorted, y_sorted, s=s, k=4)

        def predict(X_new):
            return sp(X_new[:, 0].astype(np.float64))

        return predict
    except Exception:
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            return predict
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            return predict


def _fit_exp(X, y):
    """Fit exponential model: a*exp(b*x) + c."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    # Try to fit y = a*exp(b*x) + c
    # Linearize: y - c = a*exp(b*x)
    # We'll use a grid search over b and linear fit for a, c
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    best_err = np.inf
    best_params = None

    b_values = np.linspace(-3.0 / x_range, 3.0 / x_range, 20)
    for b in b_values:
        exp_x = np.exp(b * x)
        # Fit y = a*exp_x + c
        A = np.column_stack([exp_x, np.ones(len(x))])
        try:
            coeff = np.linalg.lstsq(A, y, rcond=None)[0]
            a, c = coeff
            preds = a * exp_x + c
            err = np.mean((preds - y) ** 2)
            if err < best_err:
                best_err = err
                best_params = (a, b, c)
        except Exception:
            pass

    if best_params is None:
        c = np.mean(y)

        def predict(X_new):
            return np.full(len(X_new), c)

        return predict

    a, b, c = best_params

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        return a * np.exp(b * xn) + c

    return predict


def _fit_rational(X, y, num_deg=2, den_deg=2):
    """Fit rational function: P(x)/Q(x) where P, Q are polynomials."""
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0

    x_scaled = (x - x_mean) / x_std

    # Build basis: [x^0, x^1, ..., x^num_deg] and [y*x^0, y*x^1, ..., y*x^den_deg-1]
    # We want: y * (1 + q1*x + ... + q_{den_deg-1}*x^{den_deg-1}) = p0 + p1*x + ... + p_{num_deg}*x^{num_deg}
    # So: y * Q(x) = P(x)
    # This is linear in the coefficients!

    n_params = num_deg + den_deg + 1
    A = np.zeros((n, n_params))

    # P(x) part
    for k in range(num_deg + 1):
        A[:, k] = x_scaled ** k

    # -y*Q(x) part (Q has den_deg terms, but we fix q0=1)
    for k in range(1, den_deg + 1):
        A[:, num_deg + k] = -y * (x_scaled ** k)

    b = y * np.ones(n)

    try:
        coeff = np.linalg.lstsq(A, b, rcond=None)[0]
        p_coeffs = coeff[:num_deg + 1]
        q_coeffs = coeff[num_deg + 1:]  # q1, q2, ..., q_{den_deg}
    except Exception:
        c = np.mean(y)

        def predict(X_new):
            return np.full(len(X_new), c)

        return predict

    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x_mean) / x_std

        # Compute P(x)
        P = np.zeros(len(xn))
        for k in range(num_deg + 1):
            P += p_coeffs[k] * (xn_scaled ** k)

        # Compute Q(x) = 1 + q1*x + ... + q_{den_deg}*x^{den_deg}
        Q = np.ones(len(xn))
        for k in range(1, den_deg + 1):
            Q += q_coeffs[k - 1] * (xn_scaled ** k)

        # Avoid division by zero
        Q_safe = np.where(np.abs(Q) < 1e-10, 1e-10, Q)
        return P / Q_safe

    return predict


def _loo_cv_error(fit_func, X, y):
    """Calculate Leave-One-Out CV MSE."""
    n = len(X)
    if n < 5:
        return np.inf

    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)

    # For RBF, we can use the hat matrix trick
    # For other models, we'll do LOO properly but it's expensive
    # Let's use a faster approach: fit on all data and use the hat matrix if possible

    # For simplicity, use k-fold CV with k=n (LOO) but that's O(n^2)
    # Instead, use 5-fold CV as a proxy
    k = min(5, n)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    total_err = 0.0
    count = 0
    for f in folds:
        if _remaining() < 2:
            return np.inf
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[f], y[f]

        try:
            predict_func = fit_func(X_tr, y_tr)
            preds = predict_func(X_te)
            preds = np.asarray(preds, dtype=np.float64).flatten()
            if len(preds) != len(y_te) or not np.all(np.isfinite(preds)):
                continue
            total_err += np.sum((preds - y_te) ** 2)
            count += len(y_te)
        except Exception:
            return np.inf

    if count == 0:
        return np.inf
    return total_err / count


def fit(X_train, y_train):
    global _GLOBAL_START
    _GLOBAL_START = time.time()

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(X)

    if n < 4:
        c = np.mean(y) if n > 0 else 0.0

        def predict(X_new):
            return np.full(len(X_new), c)

        return predict

    x = X[:, 0]
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0

    # Build candidate models
    candidates = []

    # RBF with different sigmas
    sigma_fracs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    for sf in sigma_fracs:
        sigma = sf * x_range
        candidates.append((f"RBF_{sf}", lambda X_tr, y_tr, s=sigma: _fit_rbf(X_tr, y_tr, s)))

    # Chebyshev
    for deg in [3, 5, 7, 9, 11, 13]:
        candidates.append((f"Cheb_{deg}", lambda X_tr, y_tr, d=deg: _fit_chebyshev(X_tr, y_tr, d)))

    # Fourier
    for nt in [2, 3, 4, 5, 7, 10]:
        candidates.append((f"Fourier_{nt}", lambda X_tr, y_tr, nt=nt: _fit_fourier(X_tr, y_tr, nt)))

    # Splines
    for sf in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0]:
        candidates.append((f"Spline_{sf}", lambda X_tr, y_tr, s=sf: _fit_spline(X_tr, y_tr, s)))

    # Exponential
    candidates.append(("Exp", lambda X_tr, y_tr: _fit_exp(X_tr, y_tr)))

    # Rational
    for nd, dd in [(2, 2), (3, 3), (2, 3), (3, 2)]:
        candidates.append((f"Rational_{nd}_{dd}", lambda X_tr, y_tr, nd=nd, dd=dd: _fit_rational(X_tr, y_tr, nd, dd)))

    # Evaluate with CV, keep track of best
    best_cv_err = np.inf
    best_model = None
    best_name = ""

    # Sort by priority
    def _priority(name):
        if name.startswith('RBF'):
            return 0
        if name.startswith('Spline'):
            return 1
        if name.startswith('Cheb'):
            return 2
        if name.startswith('Fourier'):
            return 3
        if name.startswith('Rational'):
            return 4
        if name.startswith('Exp'):
            return 5
        return 6

    candidates.sort(key=lambda c: _priority(c[0]))

    for name, fit_func in candidates:
        if _remaining() < 5:
            break
        try:
            cv_err = _loo_cv_error(fit_func, X, y)
            if cv_err < best_cv_err:
                best_cv_err = cv_err
                best_model = fit_func(X, y)
                best_name = name
        except Exception:
            pass

    # If no model selected, fallback
    if best_model is None:
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]

            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0].astype(np.float64)])
                return An @ w

            best_model = predict
        except Exception:
            c = np.mean(y)

            def predict(X_new):
                return np.full(len(X_new), c)

            best_model = predict

    def safe_predict(X_new):
        try:
            r = best_model(np.asarray(X_new, dtype=np.float64))
            r = np.asarray(r, dtype=np.float64).flatten()
            r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
            return r
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
