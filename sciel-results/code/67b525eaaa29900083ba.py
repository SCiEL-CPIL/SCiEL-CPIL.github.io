"""Symbolic regression over 4 hidden 1-D noiseless equations.

Strategy:
- For 1-D noiseless data, exact interpolation on training points is possible.
- The key challenge is generalization to test points.
- Use multiple smooth interpolators (splines, RBF, GP) and select by hold-out.
- Add Chebyshev spectral collocation as a global, stable approximation.
- Also try polynomial fits, Fourier, and a neural network as fallbacks.
- Since data is noiseless, the best model on a random hold-out subset
  of training points should also be the best on the hidden test set.
"""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

_START = time.time()
_DEADLINE = _START + 290.0  # safety margin before 330s kill


def _now():
    return time.time()


def _fit_spline(X_train, y_train, method='cubic'):
    """Smooth spline interpolator (exact on training points)."""
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    # Remove duplicate x values
    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs, ys = xs[keep], ys[keep]
    if xs.size < 2:
        return None
    if method == 'pchip':
        spl = PchipInterpolator(xs, ys)
    else:
        spl = CubicSpline(xs, ys)

    def predict(X):
        return spl(X[:, 0])

    return predict


def _fit_poly(X_train, y_train, degree):
    """Polynomial fit."""
    x = X_train[:, 0]
    y = y_train
    try:
        coeffs = np.polyfit(x, y, degree)
    except Exception:
        return None

    def predict(X):
        return np.polyval(coeffs, X[:, 0])

    return predict


def _fit_rbf(X_train, y_train, width):
    """RBF interpolator with given width."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-(diff * diff) / (2.0 * width * width))
    K[np.diag_indices_from(K)] += 1e-8
    try:
        coef = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(K, y, rcond=None)[0]

    def predict(X):
        xt = X[:, 0]
        d = xt[:, None] - x[None, :]
        return np.exp(-(d * d) / (2.0 * width * width)) @ coef

    return predict


def _fit_gaussian_process(X_train, y_train, length_scale, noise=1e-6):
    """Simple Gaussian Process regression."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-0.5 * (diff * diff) / (length_scale * length_scale)) + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            return None

    def predict(X):
        xt = X[:, 0]
        Ks = np.exp(-0.5 * (xt[:, None] - x[None, :]) ** 2 / (length_scale * length_scale))
        return Ks @ alpha

    return predict


def _fit_chebyshev(X_train, y_train, max_degree=50, lambda_reg=1e-6):
    """Chebyshev spectral collocation with L2 regularization."""
    x = X_train[:, 0]
    y = y_train
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return None
    # Map x to [-1, 1]
    x_norm = (x - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
    
    best_degree = max_degree
    best_coeffs = None
    best_mse = float('inf')
    
    for degree in range(1, max_degree + 1):
        if _now() > _DEADLINE - 5:
            break
        try:
            # Evaluate Chebyshev polynomials at x_norm
            T = np.vander(x_norm, N=degree + 1, increasing=True)
            # Actually, we need Chebyshev polynomials, not monomials
            # Use numpy's chebyshev evaluation
            T = np.zeros((len(x_norm), degree + 1))
            for k in range(degree + 1):
                T[:, k] = np.polynomial.chebyshev.chebval(x_norm, np.eye(degree + 1)[k])
            
            # Solve regularized least squares
            A = T.T @ T
            b = T.T @ y
            A_reg = A + lambda_reg * np.eye(degree + 1)
            A_reg[0, 0] += lambda_reg * 0.1  # Less regularization on constant term
            try:
                coeffs = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A_reg, b, rcond=None)[0]
            
            # Evaluate on training data
            y_pred = T @ coeffs
            mse = np.mean((y_pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_degree = degree
                best_coeffs = coeffs
        except Exception:
            continue
    
    if best_coeffs is None:
        return None
    
    def predict(X):
        xt = X[:, 0]
        xt_norm = (xt - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
        T = np.zeros((len(xt_norm), best_degree + 1))
        for k in range(best_degree + 1):
            T[:, k] = np.polynomial.chebyshev.chebval(xt_norm, np.eye(best_degree + 1)[k])
        return T @ best_coeffs

    return predict


def _fit_fourier(X_train, y_train, n_harmonics=10):
    """Fourier series fit."""
    x = X_train[:, 0]
    y = y_train
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return None
    period = x_max - x_min
    
    # Build Fourier basis
    cols = [np.ones(len(x))]
    for k in range(1, n_harmonics + 1):
        freq = 2.0 * np.pi * k / period
        cols.append(np.sin(freq * (x - x_min)))
        cols.append(np.cos(freq * (x - x_min)))
    F = np.column_stack(cols)
    
    try:
        coeffs, *_ = np.linalg.lstsq(F, y, rcond=None)
    except Exception:
        return None
    
    def predict(X):
        xt = X[:, 0]
        cols = [np.ones(len(xt))]
        for k in range(1, n_harmonics + 1):
            freq = 2.0 * np.pi * k / period
            cols.append(np.sin(freq * (xt - x_min)))
            cols.append(np.cos(freq * (xt - x_min)))
        F = np.column_stack(cols)
        return F @ coeffs

    return predict


def _fit_nn(X_train, y_train, hidden, act, lr, epochs, seed, deadline):
    """Small MLP with Adam optimizer."""
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]
    n = X_train.shape[0]
    y = y_train
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    yt = (y - y_mean) / y_std

    layers = [d] + list(hidden) + [1]
    L = len(layers)
    W = [rng.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
         for i in range(L - 1)]
    B = [np.zeros(layers[i + 1]) for i in range(L - 1)]

    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mB = [np.zeros_like(b) for b in B]
    vB = [np.zeros_like(b) for b in B]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    best_loss = float('inf')
    best_W = [w.copy() for w in W]
    best_B = [b.copy() for b in B]

    for ep in range(epochs):
        if _now() > deadline - 2:
            break
        # forward
        a = X_train
        act_cache = [a]
        for i in range(L - 1):
            z = a @ W[i] + B[i]
            a = act(z) if i < L - 2 else z
            act_cache.append(z)
            act_cache.append(a)
        out = a
        err = out - yt
        loss = float(np.mean(err * err))
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_B = [b.copy() for b in B]

        # backward
        d_out = 2.0 * err / n
        for i in range(L - 2, -1, -1):
            z = act_cache[2 * i + 1]
            a_in = act_cache[2 * i]
            d_z = d_out
            if i < L - 2:
                if act == 'tanh':
                    d_z = d_z * (1.0 - a * a)
                elif act == 'relu':
                    d_z = d_z * (z > 0)
                else:  # sigmoid
                    d_z = d_z * a * (1.0 - a)
            dW = a_in.T @ d_z
            dB = d_z.sum(axis=0)
            d_out = d_z @ W[i].T

            t += 1
            mW[i] = beta1 * mW[i] + (1 - beta1) * dW
            vW[i] = beta2 * vW[i] + (1 - beta2) * (dW * dW)
            W[i] -= lr * (mW[i] / (1 - beta1 ** t)) / (np.sqrt(vW[i] / (1 - beta2 ** t)) + eps)

            mB[i] = beta1 * mB[i] + (1 - beta1) * dB
            vB[i] = beta2 * vB[i] + (1 - beta2) * (dB * dB)
            B[i] -= lr * (mB[i] / (1 - beta1 ** t)) / (np.sqrt(vB[i] / (1 - beta2 ** t)) + eps)

    def predict(X):
        a = X
        for i in range(L - 2):
            z = a @ best_W[i] + best_B[i]
            if act == 'tanh':
                a = np.tanh(z)
            elif act == 'relu':
                a = np.maximum(0, z)
            else:
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        a = a @ best_W[L - 2] + best_B[L - 2]
        return a * y_std + y_mean

    return predict


def fit(X_train, y_train):
    deadline = _DEADLINE
    d = X_train.shape[1]
    y = y_train
    n = len(y)
    x = X_train[:, 0]

    # Create a validation split to select the best model
    rng = np.random.RandomState(42)
    val_frac = 0.15
    val_n = max(10, int(n * val_frac))
    indices = np.arange(n)
    val_idx = rng.choice(n, val_n, replace=False)
    train_idx = np.setdiff1d(indices, val_idx)

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    candidates = []  # (val_mse, predict_callable)

    def _add(pred):
        try:
            yp_val = np.asarray(pred(X_val), dtype=float)
            if yp_val.shape != y_val.shape:
                yp_val = yp_val.reshape(y_val.shape)
            if not np.all(np.isfinite(yp_val)):
                return
            mse_val = float(np.mean((yp_val - y_val) ** 2))
            if np.isfinite(mse_val):
                candidates.append((mse_val, pred))
        except Exception:
            pass

    # ---- Spline models ----
    for method in ('cubic', 'pchip'):
        try:
            pred = _fit_spline(X_train, y, method)
            if pred is not None:
                _add(pred)
        except Exception:
            pass

    # ---- Polynomial models ----
    for deg in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15):
        if _now() > deadline - 5:
            break
        try:
            pred = _fit_poly(X_train, y, deg)
            if pred is not None:
                _add(pred)
        except Exception:
            pass

    # ---- RBF models with various widths ----
    rng_x = (x.max() - x.min())
    if rng_x > 1e-12:
        for frac in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            if _now() > deadline - 5:
                break
            try:
                pred = _fit_rbf(X_train, y, width=frac * rng_x)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Gaussian Process with various length scales ----
    if rng_x > 1e-12:
        for frac in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5):
            if _now() > deadline - 5:
                break
            try:
                pred = _fit_gaussian_process(X_train, y, length_scale=frac * rng_x)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Chebyshev spectral collocation ----
    for max_deg in (10, 20, 30, 50):
        if _now() > deadline - 5:
            break
        for lam in (1e-8, 1e-6, 1e-4, 1e-2):
            if _now() > deadline - 5:
                break
            try:
                pred = _fit_chebyshev(X_train, y, max_degree=max_deg, lambda_reg=lam)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

    # ---- Fourier models ----
    for n_harm in (5, 10, 15, 20, 30):
        if _now() > deadline - 5:
            break
        try:
            pred = _fit_fourier(X_train, y, n_harmonics=n_harm)
            if pred is not None:
                _add(pred)
        except Exception:
            pass

    # ---- Neural Networks ----
    nn_configs = [
        ([32, 32], 'tanh', 0.01, 2000, 0),
        ([64, 32], 'tanh', 0.005, 2000, 1),
        ([32, 32], 'relu', 0.01, 3000, 2),
        ([64, 64], 'tanh', 0.003, 2500, 3),
        ([128, 64], 'tanh', 0.002, 2000, 4),
    ]
    for hidden, act, lr, epochs, seed in nn_configs:
        if _now() > deadline - 10:
            break
        try:
            pred = _fit_nn(X_train, y, hidden, act, lr, epochs, seed, deadline)
            _add(pred)
        except Exception:
            continue

    # ---- Selection ----
    if not candidates:
        mean_val = float(np.mean(y))
        def predict(X):
            return np.full(X.shape[0], mean_val)
        return predict

    candidates.sort(key=lambda c: c[0])
    best_predict = candidates[0][1]

    def final_predict(X):
        try:
            yout = np.asarray(best_predict(X), dtype=float)
            yout = np.where(np.isfinite(yout), yout, 0.0)
            return yout
        except Exception:
            return np.zeros(X.shape[0])

    return final_predict
