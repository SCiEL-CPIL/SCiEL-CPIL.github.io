# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.interpolate import make_interp_spline, UnivariateSpline

_TOTAL_DEADLINE = time.time() + 318.0  # safety margin under 330s


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _poly_features(X, degree):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 7):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (x + 1e-9))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
    return np.column_stack(cols)


def _mixed_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(x)
        cols.append(x ** 2)
        cols.append(x ** 3)
        cols.append(x ** 4)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-9))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-6):
    try:
        n, p = F.shape
        F_reg = F.copy()
        F_reg[:, 0] += ridge_alpha * np.sqrt(n)
        w, *_ = np.linalg.lstsq(F_reg, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _rmse(F, w, y):
    pred = F @ w
    return np.sqrt(np.mean((pred - y) ** 2))


# ---------------------------------------------------------------------------
# Function-basis for direct structure search (y ~ a*f1 + b*f2 + c)
# ---------------------------------------------------------------------------
def _basis_fns(x):
    """Return dict name->(callable, n_extra_params, param_bounds)."""
    x = np.asarray(x, dtype=np.float64)
    def _poly(p):
        return lambda x, a, b, c: a * (x ** p) + b * x + c
    def _sin(p):
        return lambda x, a, b, c: a * np.sin(p * x) + b * x + c
    def _cos(p):
        return lambda x, a, b, c: a * np.cos(p * x) + b * x + c
    def _exp(p):
        return lambda x, a, b, c: a * np.exp(p * x) + b * x + c
    def _negexp(p):
        return lambda x, a, b, c: a * np.exp(-p * x) + b * x + c
    def _gauss(p):
        return lambda x, a, b, c: a * np.exp(-((x - p) ** 2)) + b * x + c
    def _log(p):
        return lambda x, a, b, c: a * np.log(np.abs(x) + p) + b * x + c
    def _sqrt(p):
        return lambda x, a, b, c: a * np.sqrt(np.abs(x) + p) + b * x + c
    def _recip(p):
        return lambda x, a, b, c: a / (x + p) + b * x + c
    def _tanh(p):
        return lambda x, a, b, c: a * np.tanh(p * x) + b * x + c
    def _atan(p):
        return lambda x, a, b, c: a * np.arctan(p * x) + b * x + c
    def _xexp(p):
        return lambda x, a, b, c: a * (x * np.exp(-p * x)) + b * x + c

    B = {}
    for p in [1, 2, 3, 4, 5]:
        B[f"poly{p}"] = (_poly(p), 0, [])
    for p in [1, 2, 3]:
        B[f"sin{p}"] = (_sin(p), 0, [])
        B[f"cos{p}"] = (_cos(p), 0, [])
        B[f"exp{p}"] = (_exp(p), 0, [])
        B[f"negexp{p}"] = (_negexp(p), 0, [])
        B[f"gauss{p}"] = (_gauss(p), 1, [(0.1, 3.0)])
        B[f"log{p}"] = (_log(p), 0, [])
        B[f"sqrt{p}"] = (_sqrt(p), 0, [])
        B[f"recip{p}"] = (_recip(p), 0, [])
        B[f"tanh{p}"] = (_tanh(p), 0, [])
        B[f"atan{p}"] = (_atan(p), 0, [])
        B[f"xexp{p}"] = (_xexp(p), 0, [])
    return B


def _fit_function_pair(X, y, basis, max_time=25.0):
    """Search over pairs of basis functions + coefficients. Returns (predict, rmse)."""
    try:
        x = X[:, 0]
        names = list(basis.keys())
        n = len(names)
        best_rmse = np.inf
        best_predict = None
        t0 = time.time()

        for i in range(n):
            if _remaining() < 5.0 or (time.time() - t0) > max_time:
                break
            f1, n1, b1 = basis[names[i]]
            for j in range(i, n):
                if _remaining() < 5.0 or (time.time() - t0) > max_time:
                    break
                f2, n2, b2 = basis[names[j]]

                def obj(coeffs):
                    a, b, c = coeffs[0], coeffs[1], coeffs[2]
                    try:
                        pred = a * f1(x, *coeffs[3:3 + n1]) + b * f2(x, *coeffs[3 + n1:3 + n1 + n2]) + c
                        pred = np.nan_to_num(pred, nan=0.0, posinf=1e12, neginf=-1e12)
                        return float(np.mean((pred - y) ** 2))
                    except Exception:
                        return 1e12

                # initial guess
                init = [1.0, 1.0, 0.0]
                for (lo, hi) in b1:
                    init.append((lo + hi) / 2)
                for (lo, hi) in b2:
                    init.append((lo + hi) / 2)

                try:
                    res = minimize(obj, init, method='Nelder-Mead',
                                   options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-8,
                                            'adaptive': True})
                    if res.fun < best_rmse:
                        best_rmse = float(res.fun)
                        bc = res.x
                        a, b, c = bc[0], bc[1], bc[2]
                        p1 = list(bc[3:3 + n1])
                        p2 = list(bc[3 + n1:3 + n1 + n2])

                        def make(a=a, b=b, c=c, p1=p1, p2=p2, f1=f1, f2=f2):
                            def predict(Xp):
                                Xp = np.asarray(Xp, dtype=np.float64)
                                if Xp.ndim == 1:
                                    Xp = Xp.reshape(1, -1)
                                xp = Xp[:, 0]
                                out = a * f1(xp, *p1) + b * f2(xp, *p2) + c
                                return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
                            return predict
                        best_predict = _safe_predict(make())
                except Exception:
                    pass

        if best_predict is None:
            return None, np.inf
        return best_predict, np.sqrt(best_rmse)
    except Exception:
        return None, np.inf


def _fit_spline_opt(X, y):
    """Smoothing spline with optimized smoothing parameter."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        # dedupe
        keep = np.concatenate(([True], np.diff(xs) > 1e-12))
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 5:
            return None
        y_var = float(np.var(y)) + 1e-12
        best = None
        best_mse = np.inf
        for s in [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            try:
                sp = UnivariateSpline(xs, ys, s=s)
                pred = sp(xs)
                mse = float(np.mean((pred - ys) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best = sp
            except Exception:
                continue
        if best is None:
            return None
        sp = best
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            return np.nan_to_num(sp(xp), nan=0.0, posinf=1e12, neginf=-1e12)
        return predict, np.sqrt(best_mse)
    except Exception:
        return None, np.inf


def _fit_mlp(X, y, hidden_sizes=(48, 48), lr=0.01, epochs=2000, seed=42, max_time=25.0):
    try:
        d = X.shape[1]
        n = X.shape[0]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma

        np.random.seed(seed)
        layers = [d] + list(hidden_sizes) + [1]
        W = [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
             for i in range(len(layers) - 1)]
        b = [np.zeros(layers[i + 1]) for i in range(len(layers) - 1)]

        mW = [np.zeros_like(w) for w in W]
        vW = [np.zeros_like(w) for w in W]
        mb = [np.zeros_like(bb) for bb in b]
        vb = [np.zeros_like(bb) for bb in b]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t0 = time.time()

        def forward(Xb):
            a = Xb
            acts = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
                acts.append(a)
            return acts

        for epoch in range(epochs):
            if _remaining() < 5.0 or (time.time() - t0) > max_time:
                break
            t += 1
            acts = forward(Xn)
            pred = acts[-1]
            dz = (pred - yn) / n
            grads_W = [None] * len(W)
            grads_b = [None] * len(b)
            for i in range(len(W) - 1, -1, -1):
                dz_i = dz if i == len(W) - 1 else dz @ W[i].T * (1 - acts[i + 1] ** 2)
                grads_W[i] = acts[i].T @ dz_i
                grads_b[i] = dz_i.sum(axis=0)
            for i in range(len(W)):
                mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                vW[i] = beta2 * vW[i] + (1 - beta2) * grads_W[i] ** 2
                W[i] -= lr * (mW[i] / (1 - beta1 ** t)) / (np.sqrt(vW[i] / (1 - beta2 ** t)) + eps)
                mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                vb[i] = beta2 * vb[i] + (1 - beta2) * grads_b[i] ** 2
                b[i] -= lr * (mb[i] / (1 - beta1 ** t)) / (np.sqrt(vb[i] / (1 - beta2 ** t)) + eps)

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            a = Xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        return predict
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        rmse = _rmse(F, w, y)
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _fit_gaussian_process(X, y):
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sig = y.std() + 1e-8
        yn = (y - y_mu) / y_sig

        def kernel(X1, X2, ls):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq = np.sum(diff ** 2, axis=2) / (ls ** 2)
            return np.exp(-0.5 * sq)

        best_ls, best_alpha = None, None
        for ls in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
            K = kernel(Xn, Xn, ls) + 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
                pred = kernel(Xn, Xn, ls) @ alpha
                mse = float(np.mean((pred - yn) ** 2))
                if best_alpha is None or mse < best_alpha[0]:
                    best_alpha = (mse, alpha)
                    best_ls = ls
            except Exception:
                continue
        if best_alpha is None:
            return None
        ls = best_ls
        alpha = best_alpha[1]

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn, ls)
            return (Ks @ alpha) * y_sig + y_mu
        return predict
    except Exception:
        return None


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    const = float(np.mean(y_train))
    const_rmse = float(np.sqrt(np.mean((y_train - const) ** 2)))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    def consider(predict, rmse):
        nonlocal best_rmse, best_predict
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Polynomial family (fast, strong baseline)
    for deg in [1, 2, 3, 4, 5, 6, 7, 8]:
        if _remaining() < 10.0:
            break
        predict, rmse = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_train, y_train)
        consider(predict, rmse)

    if _remaining() > 10.0:
        consider(*_try_fit_family(_trig_features, X_train, y_train))
    if _remaining() > 10.0:
        consider(*_try_fit_family(_exp_features, X_train, y_train))
    if _remaining() > 10.0:
        consider(*_try_fit_family(_mixed_features, X_train, y_train))

    # Smoothing spline with optimized smoothing (d=1)
    if _remaining() > 12.0 and d == 1:
        sp, rmse = _fit_spline_opt(X_train, y_train)
        if sp is not None:
            pred = sp(X_train)
            rmse = float(np.sqrt(np.mean((pred - y_train) ** 2)))
            consider(sp, rmse)

    # Direct function-structure search (new modelling family)
    if _remaining() > 25.0 and d == 1:
        basis = _basis_fns(X_train[:, 0])
        fp, rmse = _fit_function_pair(X_train, y_train, basis, max_time=20.0)
        if fp is not None:
            pred = fp(X_train)
            rmse = float(np.sqrt(np.mean((pred - y_train) ** 2)))
            consider(fp, rmse)

    # Gaussian Process
    if _remaining() > 15.0:
        gp = _fit_gaussian_process(X_train, y_train)
        if gp is not None:
            pred = gp(X_train)
            rmse = float(np.sqrt(np.mean((pred - y_train) ** 2)))
            consider(gp, rmse)

    # MLP (more capacity, time-bounded)
    if _remaining() > 25.0:
        mlp = _fit_mlp(X_train, y_train, max_time=20.0)
        if mlp is not None:
            pred = mlp(X_train)
            rmse = float(np.sqrt(np.mean((pred - y_train) ** 2)))
            consider(mlp, rmse)

    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
