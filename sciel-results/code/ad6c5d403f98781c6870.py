"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 315.0  # 15s safety margin

    n, d = X_train.shape
    y = y_train.copy()
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)

    def time_ok():
        return time.time() < deadline

    def build_features(X):
        feats = [np.ones(len(X))]
        for j in range(d):
            feats.append(X[:, j])
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
        for j in range(d):
            feats.append(np.sin(X[:, j]))
            feats.append(np.cos(X[:, j]))
        for j in range(d):
            c = np.clip(X[:, j], -30, 30)
            feats.append(np.exp(c))
            feats.append(np.exp(-c))
        for j in range(d):
            feats.append(np.log(np.maximum(X[:, j], 1e-6)))
        for i in range(d):
            for j in range(d):
                if i != j:
                    feats.append(X[:, i] * np.sin(X[:, j]))
                    feats.append(X[:, i] * np.cos(X[:, j]))
        return np.column_stack(feats)

    Xf = build_features(X_train)
    n_f = Xf.shape[1]

    best_mse = np.inf
    best_pred_fn = None

    def poly2_feat(X):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)

    try:
        Xp2 = poly2_feat(X_train)
        w2, *_ = np.linalg.lstsq(Xp2, y, rcond=None)
        mse2 = np.mean((Xp2 @ w2 - y) ** 2)
        if mse2 < best_mse:
            best_mse = mse2
            best_pred_fn = lambda X: poly2_feat(X) @ w2
    except:
        pass

    def lasso_coord(Xf, yv, alpha, max_iter=200):
        n_s, n_f = Xf.shape
        X_mean = np.mean(Xf, axis=0)
        X_std = np.std(Xf, axis=0)
        X_std[X_std < 1e-10] = 1.0
        Xs = (Xf - X_mean) / X_std
        y_m = np.mean(yv)
        ys = yv - y_m
        w = np.zeros(n_f)
        for _ in range(max_iter):
            for j in range(n_f):
                r = ys - Xs @ w + Xs[:, j] * w[j]
                w[j] = 0.0
                rho = np.dot(Xs[:, j], r) / n_s
                z = 1.0 / (np.dot(Xs[:, j], Xs[:, j]) / n_s + 1e-10)
                if abs(rho) > alpha:
                    w[j] = np.sign(rho) * (abs(rho) - alpha) * z
                else:
                    w[j] = 0.0
        w_orig = w / X_std
        b = y_m - np.dot(X_mean, w_orig)
        return w_orig, b

    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        if not time_ok():
            break
        try:
            wl, bl = lasso_coord(Xf, y, alpha, max_iter=100)
            pred = Xf @ wl + bl
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                wl_c = wl.copy()
                bl_c = bl
                def _pred(X, wl_c=wl_c, bl_c=bl_c):
                    return build_features(X) @ wl_c + bl_c
                best_pred_fn = _pred
        except:
            pass

    if time_ok():
        selected = [0]
        max_steps = min(10, n_f)
        for _ in range(max_steps):
            if not time_ok():
                break
            best_gain = np.inf
            best_feat = -1
            best_w = None
            for j in range(n_f):
                if j in selected:
                    continue
                idx = selected + [j]
                try:
                    X_sub = Xf[:, idx]
                    w_try, *_ = np.linalg.lstsq(X_sub, y, rcond=None)
                    mse_try = np.mean((X_sub @ w_try - y) ** 2)
                    if mse_try < best_gain:
                        best_gain = mse_try
                        best_feat = j
                        best_w = w_try
                except:
                    continue
            if best_feat == -1:
                break
            selected.append(best_feat)
            if best_gain < best_mse:
                best_mse = best_gain
                sel_c = selected.copy()
                w_c = best_w.copy()
                def _pred_fs(X, sel_c=sel_c, w_c=w_c):
                    Xf_p = build_features(X)
                    return Xf_p[:, sel_c] @ w_c
                best_pred_fn = _pred_fs

    if best_pred_fn is None:
        best_pred_fn = lambda X: np.full(len(X), y_mean)

    return best_pred_fn
# EVOLVE-BLOCK-END
