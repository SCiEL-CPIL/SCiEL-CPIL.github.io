"""Symbolic regression: multi-lengthscale RBF Gaussian-process regression
with cross-validated hyperparameters and a strict time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 315.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize inputs for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0

    def standardize(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        return (Xr - mu) / sd

    Xs = standardize(X)
    y_mean = y.mean()
    yc = y - y_mean

    # ---- Kernel definitions -------------------------------------------------
    # RBF: K(x1, x2) = exp(-0.5 * ||x1 - x2||^2 / l^2)
    # We use an anisotropic (per-dimension) lengthscale vector.
    def rbf_kernel(X1, X2, lvec):
        # X1: (n1, d), X2: (n2, d), lvec: (d,)
        # ||x1-x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1.x2
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        # scale each dim by 1/l
        X1s = X1 / lvec
        X2s = X2 / lvec
        sq1 = np.sum(X1s * X1s, axis=1)[:, None]
        sq2 = np.sum(X2s * X2s, axis=1)[None, :]
        d2 = sq1 + sq2 - 2.0 * (X1s @ X2s.T)
        np.maximum(d2, 0.0, out=d2)
        return np.exp(-0.5 * d2)

    def linear_kernel(X1, X2):
        return X1 @ X2.T

    # ---- Model: K_total = a_lin * K_lin + a_rbf * K_rbf + noise*I -----------
    # We fit coefficients a_lin, a_rbf and lengthscale via CV.

    # Candidate lengthscale vectors (in standardized units).
    # Small lengthscale = local, large = smooth.
    cand_lvec = []
    base_ls = [0.1, 0.2, 0.35, 0.6, 1.0, 1.8, 3.0]
    for l in base_ls:
        cand_lvec.append(np.full(d, l))
    # A couple of anisotropic choices: smaller in one dim
    if d >= 2:
        for l in [0.3, 0.6, 1.0]:
            for i in range(d):
                v = np.full(d, l)
                v[i] = l * 0.4
                cand_lvec.append(v)
    # Deduplicate
    seen = set()
    uniq = []
    for v in cand_lvec:
        k = tuple(np.round(v, 4))
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    cand_lvec = uniq

    # Candidate linear coefficient
    cand_alin = [0.0, 0.5, 1.0, 2.0]
    # Candidate RBF coefficient
    cand_arbf = [1.0]
    # Candidate noise (relative)
    cand_noise = [1e-6, 1e-4, 1e-3, 1e-2]

    # ---- Cross-validation over a subset of hyperparameters ------------------
    # Use 5-fold CV on the 200 training points to estimate test error.
    rng = np.random.RandomState(0)
    idx = np.arange(n)
    # Shuffle for CV folds
    perm = rng.permutation(n)
    folds = np.array_split(perm, 5)

    yvar = max(np.var(y), 1e-12)

    def evaluate_hyp(alin, lvec, noise):
        """Return mean CV normalized MSE for given hyperparams."""
        total = 0.0
        cnt = 0
        for fo in folds:
            tr = np.setdiff1d(idx, fo, assume_unique=False)
            va = fo
            if len(tr) < 5:
                continue
            Xtr = Xs[tr]
            ytr = yc[tr]
            Xva = Xs[va]
            yva = yc[va]
            Ktr = alin * linear_kernel(Xtr, Xtr) + rbf_kernel(Xtr, Xtr, lvec)
            Ktr = Ktr + (noise * yvar + 1e-8) * np.eye(len(tr))
            try:
                L = np.linalg.cholesky(Ktr)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
            except np.linalg.LinAlgError:
                try:
                    alpha = np.linalg.solve(Ktr + 1e-6 * np.eye(len(tr)), ytr)
                except np.linalg.LinAlgError:
                    return 1e12
            Ktr_va = alin * linear_kernel(Xva, Xtr) + rbf_kernel(Xva, Xtr, lvec)
            pred = Ktr_va @ alpha
            mse = np.mean((pred - yva) ** 2)
            total += mse / yvar
            cnt += 1
        return total / max(cnt, 1)

    best_cv = np.inf
    best_params = (0.0, cand_lvec[0], 1e-4)
    # Coarse search: try all (alin, lvec) with a couple of noise values
    for alin in cand_alin:
        for lvec in cand_lvec:
            for noise in cand_noise[:2]:
                cv = evaluate_hyp(alin, lvec, noise)
                if cv < best_cv:
                    best_cv = cv
                    best_params = (alin, lvec, noise)
            if time.time() - t0 > DEADLINE - 60:
                break
        if time.time() - t0 > DEADLINE - 60:
            break
        if time.time() - t0 > DEADLINE - 60:
            break

    # Refine noise around the best with the best (alin, lvec)
    alin_b, lvec_b, noise_b = best_params
    for noise in cand_noise:
        cv = evaluate_hyp(alin_b, lvec_b, noise)
        if cv < best_cv:
            best_cv = cv
            best_params = (alin_b, lvec_b, noise)
        if time.time() - t0 > DEADLINE - 40:
            break

    alin, lvec, noise = best_params

    # ---- Final fit on all data ---------------------------------------------
    Ktr = alin * linear_kernel(Xs, Xs) + rbf_kernel(Xs, Xs, lvec)
    Ktr = Ktr + (noise * yvar + 1e-8) * np.eye(n)
    try:
        L = np.linalg.cholesky(Ktr)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.solve(Ktr + 1e-6 * np.eye(n), yc)
        except np.linalg.LinAlgError:
            # Fall back to a simple mean predictor
            return lambda Xraw: np.full(len(np.asarray(Xraw, dtype=np.float64)), y_mean)

    def predict(Xraw):
        Xr = standardize(Xraw)
        Ktr_va = alin * linear_kernel(Xr, Xs) + rbf_kernel(Xr, Xs, lvec)
        return Ktr_va @ alpha + y_mean

    return predict
# EVOLVE-BLOCK-END
