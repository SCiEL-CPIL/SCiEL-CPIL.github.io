"""Symbolic regression over 4 hidden 1-D noiseless equations.

Strategy: fit several interpolating/smooth models (cubic spline, PCHIP,
well-tuned RBF / Gaussian process, small MLPs) and select the best by
training MSE.  Because the data is noiseless, the model that fits the
training points most closely while staying smooth also generalizes best.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

_START = time.time()
_DEADLINE = _START + 300.0  # evaluator kills at ~330s; leave a safety margin.


def _now():
    return time.time()


def _relu(x):
    return np.maximum(0.0, x)


def _tanh(x):
    return np.tanh(x)


def _fit_spline(X_train, y_train, method='cubic'):
    """Smooth spline interpolator (exact on training points)."""
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    # Remove duplicate x (keep last value) to satisfy spline requirements.
    keep = np.concatenate(([True], np.diff(xs) > 0))
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


def _fit_rbf(X_train, y_train, width):
    """RBF / Gaussian-process style interpolator.  Exact on training points
    (tiny ridge for stability) and smooth between them."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-(diff * diff) / (2.0 * width * width))
    K[np.diag_indices_from(K)] += 1e-9  # ridge
    try:
        coef = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(K, y, rcond=None)[0]

    def predict(X):
        xt = X[:, 0]
        d = xt[:, None] - x[None, :]
        return np.exp(-(d * d) / (2.0 * width * width)) @ coef

    return predict


def _fit_nn(X_train, y_train, hidden, act, lr, epochs, seed, deadline):
    """Small fully-connected MLP trained with Adam (full-batch, vectorized)."""
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

    for _ in range(epochs):
        if _now() > deadline:
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
                if act is _relu:
                    d_z = d_z * (z > 0)
                elif act is _tanh:
                    d_z = d_z * (1.0 - out * out)
                else:  # sigmoid
                    d_z = d_z * a_in * (1.0 - a_in)
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
            a = act(a @ best_W[i] + best_B[i])
        a = a @ best_W[L - 2] + best_B[L - 2]
        return a * y_std + y_mean

    return predict, best_loss


def fit(X_train, y_train):
    deadline = _DEADLINE
    d = X_train.shape[1]
    y = y_train
    candidates = []  # (train_mse, predict_callable)

    def _add(pred):
        try:
            yp = np.asarray(pred(X_train), dtype=float)
            if yp.shape != y.shape:
                yp = yp.reshape(y.shape)
            if not np.all(np.isfinite(yp)):
                return
            mse = float(np.mean((yp - y) ** 2))
            if np.isfinite(mse):
                candidates.append((mse, pred))
        except Exception:
            pass

    # ---- fast, exact, smooth models (1-D) ----
    if d == 1:
        for method in ('cubic', 'pchip'):
            try:
                pred = _fit_spline(X_train, y, method)
                if pred is not None:
                    _add(pred)
            except Exception:
                pass

        # RBF with a few length scales (data range based).
        x = X_train[:, 0]
        rng_x = (x.max() - x.min())
        if rng_x > 1e-12:
            for frac in (0.05, 0.1, 0.2, 0.35, 0.6):
                if _now() > deadline:
                    break
                try:
                    pred = _fit_rbf(X_train, y, width=frac * rng_x)
                    if pred is not None:
                        _add(pred)
                except Exception:
                    pass

    # ---- neural networks (multiple configs / restarts) ----
    nn_configs = [
        ([64, 64], _tanh, 0.01, 1500, 0),
        ([128, 64], _tanh, 0.005, 1500, 1),
        ([32, 32], _relu, 0.01, 2500, 2),
        ([64, 64], _relu, 0.005, 2000, 3),
    ]
    for hidden, act, lr, epochs, seed in nn_configs:
        if _now() > deadline - 5:
            break
        try:
            pred, _loss = _fit_nn(X_train, y, hidden, act, lr, epochs, seed, deadline)
            _add(pred)
        except Exception:
            continue

    # ---- selection ----
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
# EVOLVE-BLOCK-END
