"""Symbolic regression over 10 hidden equations with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    n_samples, d = X.shape
    cols = [np.ones(n_samples)]
    for deg in range(1, degree + 1):
        for indices in product(range(d), repeat=deg):
            # Only include if sorted (to avoid duplicates)
            if list(indices) == sorted(indices):
                feat = np.ones(n_samples)
                for idx in indices:
                    feat *= X[:, idx]
                cols.append(feat)
    return np.column_stack(cols)

def _sine_features(X):
    """Generate sine/cosine features for each input."""
    n_samples, d = X.shape
    cols = []
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    return np.column_stack(cols)

def _rational_features(X):
    """Generate some rational-like features (x_i / (x_j + eps))."""
    n_samples, d = X.shape
    cols = []
    eps = 1e-8
    for i in range(d):
        for j in range(d):
            if i != j:
                cols.append(X[:, i] / (X[:, j] + eps))
    return np.column_stack(cols)

def _log_features(X):
    """Generate log features (safe for positive inputs)."""
    n_samples, d = X.shape
    cols = []
    for j in range(d):
        cols.append(np.log1p(X[:, j]))
    return np.column_stack(cols)

def _exp_features(X):
    """Generate exp features (clipped for safety)."""
    n_samples, d = X.shape
    cols = []
    for j in range(d):
        cols.append(np.exp(np.clip(X[:, j], -10, 10)))
    return np.column_stack(cols)

def _sqrt_features(X):
    """Generate sqrt features."""
    n_samples, d = X.shape
    cols = []
    for j in range(d):
        cols.append(np.sqrt(np.abs(X[:, j])))
    return np.column_stack(cols)

def _fit_linear(F, y):
    """Fit linear regression with ridge regularization if needed."""
    n_samples, n_features = F.shape
    # Try plain least squares first
    try:
        w, residuals, rank, sv = np.linalg.lstsq(F, y, rcond=None)
        return w, F @ w
    except:
        # Fall back to ridge
        lam = 1e-6
        A = F.T @ F + lam * np.eye(n_features)
        b = F.T @ y
        try:
            w = np.linalg.solve(A, b)
        except:
            w = np.linalg.pinv(A) @ b
        return w, F @ w

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, d = X.shape
    
    # Store best model found so far
    best_predictor = None
    best_train_mse = np.inf
    
    def _evaluate_and_update(F, y, name=""):
        nonlocal best_predictor, best_train_mse
        if time.time() > deadline:
            return False
        n_features = F.shape[1]
        if n_features == 0:
            return False
        # Check for NaN/Inf in features
        if not np.all(np.isfinite(F)):
            return False
        w, pred = _fit_linear(F, y)
        mse = np.mean((pred - y) ** 2)
        if np.isfinite(mse) and mse < best_train_mse:
            best_train_mse = mse
            # Capture the feature matrix generation for prediction
            F_copy = F.copy()
            X_copy = X.copy()
            
            def _predict(X_test, F_train=F_copy, X_train_orig=X_copy, w=w):
                # Regenerate features for test data
                X_test = np.asarray(X_test, dtype=np.float64)
                # We need to regenerate features the same way
                # This is tricky - we'll store the feature function
                return _predict_with_features(X_test, w)
            
            # Better approach: store the feature function
            def _make_predictor(feature_func, w):
                def _predict(X_test):
                    X_test = np.asarray(X_test, dtype=np.float64)
                    F_test = feature_func(X_test)
                    if not np.all(np.isfinite(F_test)):
                        # Fallback: return mean of training y
                        return np.full(X_test.shape[0], np.mean(y))
                    pred = F_test @ w
                    # Ensure finite
                    pred = np.where(np.isfinite(pred), pred, np.mean(y))
                    return pred
                return _predict
            
            best_predictor = _make_predictor(_current_feature_func, w)
        return True
    
    # We need a way to track the current feature function
    # Let's restructure: build feature sets and evaluate each
    
    feature_sets = []
    
    # 1. Polynomial features degree 1 (linear)
    F = _poly_features(X, 1)
    feature_sets.append(("poly1", F))
    
    # 2. Polynomial features degree 2
    if time.time() < deadline:
        F = _poly_features(X, 2)
        feature_sets.append(("poly2", F))
    
    # 3. Polynomial features degree 3
    if time.time() < deadline:
        F = _poly_features(X, 3)
        feature_sets.append(("poly3", F))
    
    # 4. Sine/cosine features
    if time.time() < deadline:
        F = _sine_features(X)
        feature_sets.append(("sine", F))
    
    # 5. Log features
    if time.time() < deadline:
        F = _log_features(X)
        feature_sets.append(("log", F))
    
    # 6. Sqrt features
    if time.time() < deadline:
        F = _sqrt_features(X)
        feature_sets.append(("sqrt", F))
    
    # 7. Combination: poly2 + sine + log
    if time.time() < deadline:
        F1 = _poly_features(X, 2)
        F2 = _sine_features(X)
        F3 = _log_features(X)
        F = np.hstack([F1, F2, F3])
        feature_sets.append(("combo1", F))
    
    # 8. Combination: poly2 + sine + log + sqrt
    if time.time() < deadline:
        F1 = _poly_features(X, 2)
        F2 = _sine_features(X)
        F3 = _log_features(X)
        F4 = _sqrt_features(X)
        F = np.hstack([F1, F2, F3, F4])
        feature_sets.append(("combo2", F))
    
    # 9. Poly3 + sine
    if time.time() < deadline:
        F1 = _poly_features(X, 3)
        F2 = _sine_features(X)
        F = np.hstack([F1, F2])
        feature_sets.append(("poly3_sine", F))
    
    # 10. Poly2 + rational
    if time.time() < deadline:
        F1 = _poly_features(X, 2)
        F2 = _rational_features(X)
        F = np.hstack([F1, F2])
        feature_sets.append(("poly2_rat", F))
    
    # Evaluate each feature set
    for name, F in feature_sets:
        if time.time() > deadline:
            break
        n_features = F.shape[1]
        if n_features == 0 or not np.all(np.isfinite(F)):
            continue
        w, pred = _fit_linear(F, y)
        mse = np.mean((pred - y) ** 2)
        if np.isfinite(mse) and mse < best_train_mse:
            best_train_mse = mse
            # Store the feature generation function
            # We need to recreate this for prediction
            # Let's store the name and regenerate
            best_name = name
            best_w = w
    
    # Now create the predictor using the best feature set
    if best_predictor is None and best_name is not None:
        # Regenerate the best feature set
        if best_name == "poly1":
            F_train = _poly_features(X, 1)
            def _feature_func(X_test):
                return _poly_features(X_test, 1)
        elif best_name == "poly2":
            F_train = _poly_features(X, 2)
            def _feature_func(X_test):
                return _poly_features(X_test, 2)
        elif best_name == "poly3":
            F_train = _poly_features(X, 3)
            def _feature_func(X_test):
                return _poly_features(X_test, 3)
        elif best_name == "sine":
            F_train = _sine_features(X)
            def _feature_func(X_test):
                return _sine_features(X_test)
        elif best_name == "log":
            F_train = _log_features(X)
            def _feature_func(X_test):
                return _log_features(X_test)
        elif best_name == "sqrt":
            F_train = _sqrt_features(X)
            def _feature_func(X_test):
                return _sqrt_features(X_test)
        elif best_name == "combo1":
            F_train = np.hstack([_poly_features(X, 2), _sine_features(X), _log_features(X)])
            def _feature_func(X_test):
                return np.hstack([_poly_features(X_test, 2), _sine_features(X_test), _log_features(X_test)])
        elif best_name == "combo2":
            F_train = np.hstack([_poly_features(X, 2), _sine_features(X), _log_features(X), _sqrt_features(X)])
            def _feature_func(X_test):
                return np.hstack([_poly_features(X_test, 2), _sine_features(X_test), _log_features(X_test), _sqrt_features(X_test)])
        elif best_name == "poly3_sine":
            F_train = np.hstack([_poly_features(X, 3), _sine_features(X)])
            def _feature_func(X_test):
                return np.hstack([_poly_features(X_test, 3), _sine_features(X_test)])
        elif best_name == "poly2_rat":
            F_train = np.hstack([_poly_features(X, 2), _rational_features(X)])
            def _feature_func(X_test):
                return np.hstack([_poly_features(X_test, 2), _rational_features(X_test)])
        else:
            # Default to poly2
            F_train = _poly_features(X, 2)
            def _feature_func(X_test):
                return _poly_features(X_test, 2)
        
        w = best_w
        y_mean = np.mean(y)
        
        def _predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            F_test = _feature_func(X_test)
            if not np.all(np.isfinite(F_test)):
                return np.full(X_test.shape[0], y_mean)
            pred = F_test @ w
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        
        return _predict
    
    # Fallback: simple mean predictor
    y_mean = np.mean(y)
    def _predict_fallback(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        return np.full(X_test.shape[0], y_mean)
    
    return _predict_fallback

# EVOLVE-BLOCK-END
