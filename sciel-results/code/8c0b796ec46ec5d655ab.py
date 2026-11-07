"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Symbolic regression over 10 hidden equations.
    Strategy: Try multiple feature expansions with cross-validation and pick best.
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape

    var_y = np.var(y)
    if var_y < 1e-12:
        mean_y = np.mean(y)
        return lambda X_test: np.full(len(X_test), mean_y)

    # Normalize X for better conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0

    def normalize(X_test):
        return (X_test - X_mean) / X_std

    # 5-fold cross-validation
    n_folds = 5
    rng = np.random.RandomState(42)
    fold_indices = rng.permutation(n_samples)
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        folds.append(fold_indices[start:end])

    def cross_validate(build_features, X_raw, y_raw):
        """5-fold CV: return mean MSE and best fold weights"""
        mse_sum = 0.0
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            X_tr_raw = X_raw[tr_idx]
            y_tr_raw = y_raw[tr_idx]
            X_va_raw = X_raw[val_idx]
            y_va_raw = y_raw[val_idx]
            
            F_tr = build_features(normalize(X_tr_raw))
            F_va = build_features(normalize(X_va_raw))
            
            n_feat = F_tr.shape[1]
            n_train = F_tr.shape[0]
            
            if n_feat > n_train:
                lam = 1.0
                A = F_tr.T @ F_tr + lam * np.eye(n_feat)
                b = F_tr.T @ y_tr_raw
                w = np.linalg.solve(A, b)
            else:
                w, *_ = np.linalg.lstsq(F_tr, y_tr_raw, rcond=None)
            
            pred_va = F_va @ w
            mse_sum += np.mean((pred_va - y_va_raw) ** 2)
        return mse_sum / n_folds

    best_score = np.inf
    best_predictor = lambda X_test: np.full(len(X_test), np.mean(y))
    best_model_name = "const"

    def try_model(build_features, name=""):
        nonlocal best_score, best_predictor, best_model_name
        if time.time() > deadline:
            return
        try:
            cv_mse = cross_validate(build_features, X, y)
            score = cv_mse / var_y
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_model_name = name
                # Fit on all data
                F_all = build_features(normalize(X))
                n_feat = F_all.shape[1]
                n_train = F_all.shape[0]
                if n_feat > n_train:
                    lam = 1.0
                    A = F_all.T @ F_all + lam * np.eye(n_feat)
                    b = F_all.T @ y
                    w = np.linalg.solve(A, b)
                else:
                    w, *_ = np.linalg.lstsq(F_all, y, rcond=None)
                
                def predictor(X_test, _bf=build_features, _w=w):
                    F_test = _bf(normalize(np.asarray(X_test, dtype=np.float64)))
                    return F_test @ _w
                best_predictor = predictor
        except Exception:
            pass

    # Model 0: Constant
    try_model(lambda Xn: np.ones((len(Xn), 1)), "const")

    # Model 1: Linear
    try_model(lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]), "linear")

    # Model 2: Quadratic polynomial
    def quad_features(Xn):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        for i in range(d):
            for j in range(i, d):
                cols.append((Xn[:, i] * Xn[:, j]).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(quad_features, "quad")

    # Model 3: Cubic polynomial
    def cubic_features(Xn):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append((Xn[:, i] * Xn[:, j] * Xn[:, k]).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(cubic_features, "cubic")

    # Model 4: Trig features
    def trig_features(Xn):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        for i in range(d):
            cols.append(np.sin(Xn[:, i]).reshape(-1, 1))
            cols.append(np.cos(Xn[:, i]).reshape(-1, 1))
            cols.append((Xn[:, i] ** 2).reshape(-1, 1))
        for i in range(d):
            for j in range(i + 1, d):
                cols.append((Xn[:, i] * Xn[:, j]).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(trig_features, "trig")

    # Model 5: Exp and log
    def explog_features(Xn):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        for i in range(d):
            cols.append(np.exp(Xn[:, i]).reshape(-1, 1))
            cols.append(np.log(np.abs(Xn[:, i]) + 1e-8).reshape(-1, 1))
            cols.append((Xn[:, i] ** 2).reshape(-1, 1))
        for i in range(d):
            for j in range(i + 1, d):
                cols.append((Xn[:, i] * Xn[:, j]).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(explog_features, "explog")

    # Model 6: Rich combination
    def rich_features(Xn):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        for i in range(d):
            xi = Xn[:, i]
            cols.append((xi ** 2).reshape(-1, 1))
            cols.append((xi ** 3).reshape(-1, 1))
            cols.append(np.sin(xi).reshape(-1, 1))
            cols.append(np.cos(xi).reshape(-1, 1))
            cols.append(np.exp(xi).reshape(-1, 1))
            cols.append(np.log(np.abs(xi) + 1e-8).reshape(-1, 1))
        for i in range(d):
            for j in range(i + 1, d):
                cols.append((Xn[:, i] * Xn[:, j]).reshape(-1, 1))
                cols.append(np.sin(Xn[:, i] + Xn[:, j]).reshape(-1, 1))
                cols.append(np.cos(Xn[:, i] - Xn[:, j]).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(rich_features, "rich")

    # Model 7: RBF kernel features with multiple sigmas
    def make_rbf_features(n_centers, sigma):
        # Select centers from normalized training data
        Xn_train = normalize(X)
        if n_centers <= len(Xn_train):
            step = max(1, len(Xn_train) // n_centers)
            centers = Xn_train[::step][:n_centers]
        else:
            centers = Xn_train
        
        def rbf_feat(Xn_test):
            diff = Xn_test[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            phi = np.exp(-dist_sq / (2 * sigma ** 2))
            return np.column_stack([np.ones(len(Xn_test)), phi])
        return rbf_feat

    for sigma in [0.3, 0.5, 1.0, 1.5, 2.0]:
        try_model(make_rbf_features(30, sigma), f"rbf_{sigma}")

    # Model 8: Spline-like basis using Chebyshev polynomials
    def cheb_features(Xn, degree=5):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        for i in range(d):
            xi = Xn[:, i]
            # Chebyshev polynomials up to degree
            T0 = np.ones_like(xi)
            T1 = xi
            cols.append(T0.reshape(-1, 1))
            cols.append(T1.reshape(-1, 1))
            for k in range(2, degree + 1):
                Tk = 2 * xi * T1 - T0
                cols.append(Tk.reshape(-1, 1))
                T0, T1 = T1, Tk
        # Interactions
        for i in range(d):
            for j in range(i + 1, d):
                cols.append((Xn[:, i] * Xn[:, j]).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(cheb_features, "cheb5")

    # Model 9: Scaled higher-degree polynomial
    def scaled_poly_features(Xn, degree=3):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        # All monomials up to degree
        from itertools import combinations_with_replacement
        for deg in range(2, degree + 1):
            for combo in combinations_with_replacement(range(d), deg):
                term = np.ones(len(Xn))
                for idx in combo:
                    term = term * Xn[:, idx]
                cols.append(term.reshape(-1, 1))
        return np.column_stack(cols)
    try_model(lambda Xn: scaled_poly_features(Xn, 3), "scaled_poly3")

    # Model 10: Tanh features with multiple random projections
    def make_tanh_features(n_hidden, seed):
        d = n_features
        rng = np.random.RandomState(seed)
        W = rng.randn(d, n_hidden) * 1.0
        b = rng.randn(n_hidden) * 0.1
        def tanh_feat(Xn):
            Z = Xn @ W + b
            cols = [np.ones(len(Xn)), Xn, np.tanh(Z)]
            return np.column_stack(cols)
        return tanh_feat

    for nh in [10, 20, 30]:
        for seed in [0, 1, 2]:
            try_model(make_tanh_features(nh, seed), f"tanh{nh}_s{seed}")

    # Model 11: Gaussian process-like: use multiple RBF centers with adaptive selection
    def adaptive_rbf_features(Xn):
        d = Xn.shape[1]
        n_centers = min(40, len(Xn))
        # Use k-means-like selection: pick evenly spaced points
        idxs = np.linspace(0, len(Xn) - 1, n_centers, dtype=int)
        centers = Xn[idxs]
        diff = Xn[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        # Multiple bandwidths
        phis = []
        for sigma in [0.5, 1.0, 2.0]:
            phis.append(np.exp(-dist_sq / (2 * sigma ** 2)))
        return np.column_stack([np.ones(len(Xn))] + phis)
    try_model(adaptive_rbf_features, "adaptive_rbf")

    # Model 12: Fourier features
    def fourier_features(Xn, n_freqs=5):
        cols = [np.ones(len(Xn)), Xn]
        d = Xn.shape[1]
        freqs = np.arange(1, n_freqs + 1, dtype=np.float64)
        for i in range(d):
            xi = Xn[:, i]
            for f in freqs:
                cols.append(np.sin(f * xi).reshape(-1, 1))
                cols.append(np.cos(f * xi).reshape(-1, 1))
        return np.column_stack(cols)
    try_model(fourier_features, "fourier")

    # If we have time, try even more models
    if time.time() < deadline - 10:
        # Model 13: Product of trig functions
        def prod_trig_features(Xn):
            cols = [np.ones(len(Xn)), Xn]
            d = Xn.shape[1]
            for i in range(d):
                cols.append(np.sin(Xn[:, i]).reshape(-1, 1))
                cols.append(np.cos(Xn[:, i]).reshape(-1, 1))
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append((np.sin(Xn[:, i]) * np.cos(Xn[:, j])).reshape(-1, 1))
                    cols.append((np.cos(Xn[:, i]) * np.sin(Xn[:, j])).reshape(-1, 1))
                    cols.append((np.sin(Xn[:, i]) * np.sin(Xn[:, j])).reshape(-1, 1))
                    cols.append((np.cos(Xn[:, i]) * np.cos(Xn[:, j])).reshape(-1, 1))
            return np.column_stack(cols)
        try_model(prod_trig_features, "prod_trig")

        # Model 14: Rational-like: 1/(1+exp(-x)) sigmoid features
        def sigmoid_features(Xn):
            cols = [np.ones(len(Xn)), Xn]
            d = Xn.shape[1]
            for i in range(d):
                xi = Xn[:, i]
                for c in [-1, 0, 1]:
                    cols.append((1 / (1 + np.exp(-(xi - c)))).reshape(-1, 1))
            return np.column_stack(cols)
        try_model(sigmoid_features, "sigmoid")

    return best_predictor

# EVOLVE-BLOCK-END
