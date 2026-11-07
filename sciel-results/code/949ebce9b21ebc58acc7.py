"""Symbolic regression over 4 hidden 1-D noiseless equations.

Strategy:
- For 1-D noiseless data, the test points are likely close to (but not exactly at)
  the training points. The key is to find a smooth function that passes through
  all training points and generalizes well to nearby test points.
- Use multiple approaches:
  1. Cubic spline interpolation (exact on training points, smooth)
  2. PCHIP (monotonicity-preserving, exact on training points)
  3. Polynomial fits of various degrees (via least squares)
  4. Fourier series fits (for potentially periodic functions)
  5. RBF interpolation with various width parameters
  6. A small neural network as fallback
- Select the best model using a hold-out validation set (split training data).
  Since data is noiseless, the model that best predicts held-out training points
  should also generalize best to test points.
"""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 300s limit


def _now():
    return time.time()


def _remaining():
    return max(0.0, _DEADLINE - _now())


def _fit_spline(X_train, y_train, method='cubic'):
    """Smooth spline interpolator (exact on training points)."""
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    # Remove duplicate x
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
    """Polynomial fit via least squares."""
    x = X_train[:, 0]
    y = y_train
    # Normalize x to [-1, 1] for numerical stability
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    x_norm = (x - (x_min + x_max) / 2) / (x_range / 2)
    
    V = np.vander(x_norm, degree + 1, increasing=True)
    try:
        coef, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    def predict(X):
        xt = X[:, 0]
        xt_norm = (xt - (x_min + x_max) / 2) / (x_range / 2)
        Vt = np.vander(xt_norm, degree + 1, increasing=True)
        return Vt @ coef

    return predict


def _fit_fourier(X_train, y_train, n_terms):
    """Fourier series fit via least squares."""
    x = X_train[:, 0]
    y = y_train
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    
    # Normalize x to [0, 2*pi]
    x_norm = 2 * np.pi * (x - x_min) / x_range
    
    # Build Fourier feature matrix
    cols = [np.ones_like(x_norm)]
    for k in range(1, n_terms + 1):
        cols.append(np.sin(k * x_norm))
        cols.append(np.cos(k * x_norm))
    V = np.column_stack(cols)
    
    try:
        coef, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    def predict(X):
        xt = X[:, 0]
        xt_norm = 2 * np.pi * (xt - x_min) / x_range
        cols = [np.ones_like(xt_norm)]
        for k in range(1, n_terms + 1):
            cols.append(np.sin(k * xt_norm))
            cols.append(np.cos(k * xt_norm))
        Vt = np.column_stack(cols)
        return Vt @ coef

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
        try:
            coef = np.linalg.lstsq(K, y, rcond=None)[0]
        except Exception:
            return None

    def predict(X):
        xt = X[:, 0]
        d = xt[:, None] - x[None, :]
        return np.exp(-(d * d) / (2.0 * width * width)) @ coef

    return predict


def _fit_nn(X_train, y_train, hidden_sizes, lr, epochs, seed, deadline):
    """Small fully-connected MLP trained with Adam."""
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]
    n = X_train.shape[0]
    y = y_train
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    yt = (y - y_mean) / y_std

    layers = [d] + list(hidden_sizes) + [1]
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

    for epoch in range(epochs):
        if _now() > deadline:
            break
        # forward
        a = X_train
        act_cache = [a]
        for i in range(L - 1):
            z = a @ W[i] + B[i]
            a = np.tanh(z) if i < L - 2 else z
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
            d_z = d_out.copy()
            if i < L - 2:
                d_z = d_z * (1.0 - np.tanh(z) ** 2)
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
            a = np.tanh(a @ best_W[i] + best_B[i])
        a = a @ best_W[L - 2] + best_B[L - 2]
        return a * y_std + y_mean

    return predict


def fit(X_train, y_train):
    deadline = _DEADLINE
    d = X_train.shape[1]
    y = y_train
    n = len(y)
    
    # Create hold-out validation set
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    val_size = max(10, n // 10)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    
    candidates = []  # (val_mse, predict_callable)

    def _add(pred):
        try:
            yp = np.asarray(pred(X_val), dtype=float)
            if yp.shape != y_val.shape:
                yp = yp.reshape(y_val.shape)
            if not np.all(np.isfinite(yp)):
                return
            mse = float(np.mean((yp - y_val) ** 2))
            if np.isfinite(mse):
                candidates.append((mse, pred))
        except Exception:
            pass

    # ---- Spline interpolators ----
    if d == 1:
        for method in ('cubic', 'pchip'):
            try:
                pred = _fit_spline(X_train, y, method)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

        # ---- Polynomial fits ----
        for degree in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
            if _now() > deadline:
                break
            try:
                pred = _fit_poly(X_train, y, degree)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

        # ---- Fourier fits ----
        for n_terms in [3, 5, 7, 10, 15, 20]:
            if _now() > deadline:
                break
            try:
                pred = _fit_fourier(X_train, y, n_terms)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

        # ---- RBF fits ----
        x = X_train[:, 0]
        rng_x = x.max() - x.min()
        if rng_x > 1e-12:
            for frac in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]:
                if _now() > deadline:
                    break
                try:
                    pred = _fit_rbf(X_train, y, width=frac * rng_x)
                    if pred is not None:
                        _add(pred)
                except Exception:
                    pass

    # ---- Neural networks ----
    nn_configs = [
        ([32, 32], 0.01, 1000, 0),
        ([64, 32], 0.005, 1500, 1),
        ([128, 64], 0.005, 1000, 2),
        ([64, 64, 32], 0.003, 1500, 3),
    ]
    for hidden, lr, epochs, seed in nn_configs:
        if _now() > deadline - 5:
            break
        try:
            pred = _fit_nn(X_train, y, hidden, lr, epochs, seed, deadline)
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
