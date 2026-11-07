# EVOLVE-BLOCK-START
import numpy as np
import time


def _tricube_weights(d, h):
    u = np.abs(d) / h
    u = np.minimum(u, 1.0)
    w = (1.0 - u ** 3) ** 3
    w[u >= 1.0] = 0.0
    return w


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 250.0  # safety margin before the 300 s wall-clock limit

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    rng = max(x_max - x_min, 1e-12)

    # Quadratic local design (constant across queries; depends only on data points)
    A = np.column_stack([np.ones(n), x, x ** 2])  # n x 3

    # Global degree-4 polynomial fallback for points far from the training support
    def _global_design(xx):
        xx = np.asarray(xx, dtype=float).reshape(-1)
        return np.column_stack([xx ** k for k in range(5)])

    gA = _global_design(x)
    gw, *_ = np.linalg.lstsq(gA, y, rcond=None)

    def _global_pred(xx):
        return _global_design(xx) @ gw

    # --- Adaptive bandwidth via leave-one-out cross-validation ---
    fracs = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]
    D = np.abs(x[:, None] - x[None, :])  # n x n pairwise distances
    I3 = np.eye(3)

    best_frac = fracs[0]
    best_lcv = np.inf
    for frac in fracs:
        if time.time() - t0 > DEADLINE:
            break
        h = frac * rng
        W = _tricube_weights(D, h)  # n x n
        np.fill_diagonal(W, 0.0)  # leave-one-out

        C = A[None, :, :] * W[:, :, None]            # n x n x 3
        M = np.einsum('ijk,jl->ikl', C, A) + 1e-10 * I3[None]  # n x 3 x 3
        rhs = W @ (A * y[:, None])                    # n x 3
        try:
            beta = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(M, rhs, rcond=None)[0]
        pred = np.einsum('ik,ik->i', A, beta)
        lcv = float(np.mean((y - pred) ** 2))
        if np.isfinite(lcv) and lcv < best_lcv:
            best_lcv = lcv
            best_frac = frac

    h_final = best_frac * rng

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(-1, 1)
        xq = Xq[:, 0]
        m = len(xq)
        Dq = np.abs(xq[:, None] - x[None, :])  # m x n
        Wq = _tricube_weights(Dq, h_final)     # m x n

        pred = np.empty(m, dtype=float)
        wsum = Wq.sum(axis=1)
        use_global = wsum < 1e-12
        pred[use_global] = _global_pred(xq[use_global])

        if np.any(~use_global):
            idx = np.where(~use_global)[0]
            Ws = Wq[idx]                                    # k x n
            C = A[None, :, :] * Ws[:, :, None]              # k x n x 3
            M = np.einsum('ijk,jl->ikl', C, A) + 1e-10 * I3[None]  # k x 3 x 3
            rhs = Ws @ (A * y[:, None])                     # k x 3
            try:
                beta = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(M, rhs, rcond=None)[0]
            Q = np.column_stack([np.ones(len(idx)), xq[idx], xq[idx] ** 2])
            pred[idx] = np.einsum('ik,ik->i', Q, beta)

        return pred

    return predict
# EVOLVE-BLOCK-END
