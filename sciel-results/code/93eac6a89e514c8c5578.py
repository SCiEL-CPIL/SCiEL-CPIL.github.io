import numpy as np
import time

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 326.0  # safety margin under 330s
_PER_EQ_TIME = 29.0


def _matern52(K, ls):
    r = np.sqrt(np.maximum(K, 0.0))
    sr = r / ls
    return np.exp(-sr) * (1.0 + sr + sr * sr / 3.0)


def _rbf(K, ls):
    return np.exp(-0.5 * K / (ls * ls))


def _build_kernel(X, ls, sig2, rbf_ls, rbf_var):
    n = X.shape[0]
    d = X.shape[1]
    K = np.zeros((n, n))
    for j in range(d):
        col = X[:, j][:, None]
        K += (col - col.T) ** 2
    K = 0.5 * K  # squared distance / 2
    K = sig2 * _matern52(K, ls)
    if rbf_var > 1e-15:
        K = K + rbf_var * _rbf(K, rbf_ls)
    return K


def _loocv_score(X, y, ls, sig2, rbf_ls, rbf_var, jitter=1e-8):
    n = X.shape[0]
    K = _build_kernel(X, ls, sig2, rbf_ls, rbf_var)
    K = K + jitter * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-5 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K + 1e-3 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return np.inf
    # Solve L v = y
    v = np.linalg.solve(L, y)
    diag = np.einsum('ij,ij->i', L, v)  # diag(K^{-1})
    alpha = np.linalg.solve(L.T, v)     # K^{-1} y
    loocv_mse = np.mean(alpha * alpha / (diag * diag))
    return loocv_mse


def _predict(Xtr, ytr, Xq, ls, sig2, rbf_ls, rbf_var, jitter=1e-8):
    K = _build_kernel(Xtr, ls, sig2, rbf_ls, rbf_var)
    K = K + jitter * np.eye(len(Xtr))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-5 * np.eye(len(Xtr))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K + 1e-3 * np.eye(len(Xtr))
            L = np.linalg.cholesky(K)
    v = np.linalg.solve(L, ytr)
    alpha = np.linalg.solve(L.T, v)
    Kq = _build_kernel(Xq, ls, sig2, rbf_ls, rbf_var)
    # cross kernel: Xq (m x d) vs Xtr (n x d)
    m = Xq.shape[0]
    n = Xtr.shape[0]
    Kcross = np.zeros((m, n))
    for j in range(Xtr.shape[1]):
        a = Xq[:, j][:, None]
        b = Xtr[:, j][None, :]
        Kcross += (a - b) ** 2
    Kcross = 0.5 * Kcross
    Kcross = sig2 * _matern52(Kcross, ls)
    if rbf_var > 1e-15:
        Kcross = Kcross + rbf_var * _rbf(Kcross, rbf_ls)
    pred = Kcross @ alpha
    return pred


def fit(X_train, y_train):
    t0 = time.time()
    eq_deadline = min(t0 + _PER_EQ_TIME, _GLOBAL_DEADLINE - 2.0)

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        # constant target
        cval = float(y_mean)
        def predict_const(X):
            X = np.asarray(X, dtype=np.float64)
            return np.full(len(X), cval, dtype=np.float64)
        return predict_const

    # Standardize inputs for well-conditioned length scales
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    # Standardize outputs (GPR signal variance ~ 1)
    ys = (y - y_mean) / y_std

    def _params_to_ls(params):
        # params: [log_ls (d), log_sig2, log_rbf_ls, log_rbf_var]
        ls = np.exp(params[:d])
        sig2 = np.exp(params[d])
        rbf_ls = np.exp(params[d + 1])
        rbf_var = np.exp(params[d + 2])
        return ls, sig2, rbf_ls, rbf_var

    def _score(params):
        ls, sig2, rbf_ls, rbf_var = _params_to_ls(params)
        return _loocv_score(Xs, ys, ls, sig2, rbf_ls, rbf_var)

    # ---- Multi-start hyperparameter search ----
    rng = np.random.default_rng(12345)
    n_param = d + 3
    best_score = np.inf
    best_params = None

    # Candidate initial points
    inits = []
    # Default: isotropic moderate length scale, sig2=1, small rbf
    base = np.concatenate([np.full(d, np.log(0.8)), [np.log(1.0)], [np.log(0.5)], [np.log(1e-3)]])
    inits.append(base)
    # A few random starts
    for _ in range(20):
        p = np.concatenate([
            rng.uniform(np.log(0.1), np.log(3.0), size=d),
            [rng.uniform(np.log(0.1), np.log(3.0))],
            [rng.uniform(np.log(0.1), np.log(3.0))],
            [rng.uniform(np.log(1e-4), np.log(1e-1))],
        ])
        inits.append(p)

    # Evaluate inits, keep best
    for p in inits:
        if time.time() > eq_deadline - 3.0:
            break
        s = _score(p)
        if s < best_score:
            best_score = s
            best_params = p.copy()

    # Coordinate refinement (Nelder-Mead-ish via scipy if available, else manual)
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except Exception:
        have_scipy = False

    if have_scipy and best_params is not None and time.time() < eq_deadline - 3.0:
        # Run a few local minimizations from the best starts
        starts = [best_params]
        # add a couple more promising inits
        starts.append(np.concatenate([np.full(d, np.log(0.5)), [np.log(1.0)], [np.log(0.3)], [np.log(1e-3)]]))
        starts.append(np.concatenate([np.full(d, np.log(1.5)), [np.log(2.0)], [np.log(1.0)], [np.log(1e-3)]]))
        for sp in starts:
            if time.time() > eq_deadline - 3.0:
                break
            try:
                res = minimize(_score, sp, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        best_params = base

    ls, sig2, rbf_ls, rbf_var = _params_to_ls(best_params)

    # Final prediction function
    def predict(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        Xqs = (Xq - mu) / sd
        pred_s = _predict(Xs, ys, Xqs, ls, sig2, rbf_ls, rbf_var)
        return y_mean + y_std * pred_s

    return predict
