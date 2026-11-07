import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import solve

def fit(X_train, y_train):
    X_train = np.asarray(X_train).flatten()
    y_train = np.asarray(y_train).flatten()
    
    # Normalize inputs to [-1, 1] for better conditioning of basis functions
    x_min = X_train.min()
    x_max = X_train.max()
    if x_max == x_min:
        x_norm = np.zeros_like(X_train)
    else:
        x_norm = (X_train - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
    
    # Store normalization params
    x_center = (x_max + x_min) / 2
    x_scale = (x_max - x_min) / 2

    best_model = None
    best_error = np.inf

    # Helper to evaluate a model and update best
    def try_model(predict_func, name):
        nonlocal best_model, best_error
        pred = predict_func(x_norm)
        err = np.mean((pred - y_train) ** 2)
        if err < best_error:
            best_error = err
            best_model = predict_func

    # 1. Polynomial Models
    for deg in [2, 4, 6, 8, 10, 12]:
        V = np.vander(x_norm, N=deg + 1, increasing=True)
        # Add regularization if needed, but noiseless data usually allows exact fit
        try:
            w, residuals, rank, s = np.linalg.lstsq(V, y_train, rcond=None)
            def poly_pred(x, w=w, deg=deg):
                V_test = np.vander(x, N=deg + 1, increasing=True)
                return V_test @ w
            try_model(poly_pred, f"Poly_{deg}")
        except:
            pass

    # 2. Fourier Series (for periodic functions)
    for n_terms in [3, 5, 7, 10]:
        features = [np.ones_like(x_norm)]
        for k in range(1, n_terms + 1):
            features.append(np.sin(k * np.pi * x_norm))
            features.append(np.cos(k * np.pi * x_norm))
        V = np.column_stack(features)
        try:
            w, residuals, rank, s = np.linalg.lstsq(V, y_train, rcond=None)
            def fourier_pred(x, w=w, n_terms=n_terms):
                feats = [np.ones_like(x)]
                for k in range(1, n_terms + 1):
                    feats.append(np.sin(k * np.pi * x))
                    feats.append(np.cos(k * np.pi * x))
                V_test = np.column_stack(feats)
                return V_test @ w
            try_model(fourier_pred, f"Fourier_{n_terms}")
        except:
            pass

    # 3. RBF (Multiquadric) Interpolation
    # K(x, x') = sqrt(1 + eps^2 * (x-x')^2)
    # We optimize eps
    def rbf_predict(x_test, x_train, y_train, eps):
        D = np.abs(x_test[:, None] - x_train[None, :])
        K = np.sqrt(1.0 + (eps * D) ** 2)
        # Solve K * w = y_train
        try:
            w = solve(K, y_train, assume_a='pos')
        except:
            # Add jitter if singular
            try:
                w = solve(K + 1e-8 * np.eye(len(x_train)), y_train, assume_a='pos')
            except:
                return np.full_like(x_test, np.mean(y_train))
        return K @ w

    # Optimize eps for RBF
    # Note: rbf_predict is slow to call inside minimize_scalar if we re-solve every time.
    # But N=200 is small.
    
    # We need to capture x_train, y_train in the closure for optimization
    x_tr = x_norm
    y_tr = y_train
    
    def objective(eps_log):
        eps = np.exp(eps_log)
        pred = rbf_predict(x_tr, x_tr, y_tr, eps)
        return np.mean((pred - y_tr) ** 2)

    # Search over a range of eps
    # eps range: 0.1 to 100
    res = minimize_scalar(objective, bounds=(np.log(0.1), np.log(100)), method='bounded')
    if res.success:
        best_eps = np.exp(res.x)
        def rbf_pred(x, eps=best_eps):
            return rbf_predict(x, x_tr, y_tr, eps)
        try_model(rbf_pred, f"RBF_eps_{best_eps:.2f}")
    else:
        # Try a few fixed values if optimization fails
        for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
            def rbf_pred(x, eps=eps):
                return rbf_predict(x, x_tr, y_tr, eps)
            try_model(rbf_pred, f"RBF_fixed_{eps}")

    # 4. Gaussian Kernel (Softmax-like)
    # K(x, x') = exp(-gamma * (x-x')^2)
    def gaussian_predict(x_test, x_train, y_train, gamma):
        D = (x_test[:, None] - x_train[None, :]) ** 2
        K = np.exp(-gamma * D)
        try:
            w = solve(K, y_train, assume_a='pos')
        except:
            try:
                w = solve(K + 1e-8 * np.eye(len(x_train)), y_train, assume_a='pos')
            except:
                return np.full_like(x_test, np.mean(y_train))
        return K @ w

    def objective_gauss(gamma_log):
        gamma = np.exp(gamma_log)
        pred = gaussian_predict(x_tr, x_tr, y_tr, gamma)
        return np.mean((pred - y_tr) ** 2)

    res_g = minimize_scalar(objective_gauss, bounds=(np.log(0.1), np.log(100)), method='bounded')
    if res_g.success:
        best_gamma = np.exp(res_g.x)
        def gauss_pred(x, gamma=best_gamma):
            return gaussian_predict(x, x_tr, y_tr, gamma)
        try_model(gauss_pred, f"Gauss_gamma_{best_gamma:.2f}")

    # Fallback: If best_model is None (should not happen), use linear
    if best_model is None:
        def fallback_pred(x):
            return np.ones_like(x) * np.mean(y_train)
        best_model = fallback_pred

    def predict(X):
        X = np.asarray(X).flatten()
        # Normalize
        x_norm_test = (X - x_center) / x_scale if x_scale != 0 else np.zeros_like(X)
        preds = best_model(x_norm_test)
        return preds

    return predict
