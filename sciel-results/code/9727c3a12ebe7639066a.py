"""Symbolic regression via multi-model selection with polynomial, RBF, and GP fallbacks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

TIME_LIMIT = 290.0  # seconds, safety margin before 300s kill


def _make_poly_predict(X, y, degree):
    """Create polynomial regression predictor with given degree."""
    d = X.shape[1]
    n = len(X)
    
    # Generate polynomial features up to given degree
    def gen_features(X):
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        # Interaction and higher-order terms
        for deg in range(2, degree + 1):
            for i in range(d):
                for j in range(i, d):
                    if deg == 2:
                        if i == j:
                            cols.append(X[:, i] ** 2)
                        else:
                            cols.append(X[:, i] * X[:, j])
                    elif deg == 3:
                        if i == j:
                            cols.append(X[:, i] ** 3)
                        else:
                            cols.append(X[:, i] ** 2 * X[:, j])
                            if j > i:
                                cols.append(X[:, i] * X[:, j] ** 2)
                    elif deg == 4:
                        if i == j:
                            cols.append(X[:, i] ** 4)
                        else:
                            cols.append(X[:, i] ** 3 * X[:, j])
                            cols.append(X[:, i] ** 2 * X[:, j] ** 2)
                            if j > i:
                                cols.append(X[:, i] * X[:, j] ** 3)
        return np.column_stack(cols)
    
    F_train = gen_features(X)
    F_test_shape = (X.shape[1],)  # for reference
    
    # Add ridge regularization for stability
    ridge = 1e-4
    w, *_ = np.linalg.lstsq(F_train + ridge * np.eye(len(F_train)), y, rcond=None)
    
    def predict(X_new):
        F_new = gen_features(X_new)
        return F_new @ w
    
    return predict


def _make_rbf_predict(X, y, n_centers=20, sigma=None):
    """Create RBF (Radial Basis Function) predictor."""
    d = X.shape[1]
    n = len(X)
    
    if sigma is None:
        # Use median distance between random pairs
        idx = np.random.choice(n, min(50, n), replace=False)
        dists = cdist(X[idx], X[idx])
        sigma = np.median(dists[dists > 0]) * 2.0
        if sigma <= 0:
            sigma = 1.0
    
    # Select centers using k-means-like approach
    rng = np.random.RandomState(42)
    centers = X[rng.choice(n, min(n_centers, n), replace=False)]
    
    # Fit weights
    K = np.exp(-cdist(X, centers) ** 2 / (2 * sigma ** 2))
    ridge = 1e-3
    w, *_ = np.linalg.lstsq(K + ridge * np.eye(len(K)), y, rcond=None)
    
    def predict(X_new):
        K_new = np.exp(-cdist(X_new, centers) ** 2 / (2 * sigma ** 2))
        return K_new @ w
    
    return predict


def _make_gp_predict(X, y):
    """Create Gaussian Process regression predictor."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
    gp.fit(X, y)
    
    def predict(X_new):
        return gp.predict(X_new)
    
    return predict


def _make_svm_poly_predict(X, y):
    """Create SVM with polynomial kernel predictor."""
    from sklearn.svm import SVR
    
    svm = SVR(kernel='poly', degree=3, C=10.0, gamma='scale', epsilon=0.1)
    svm.fit(X, y)
    
    def predict(X_new):
        return svm.predict(X_new)
    
    return predict


def _make_svm_rbf_predict(X, y):
    """Create SVM with RBF kernel predictor."""
    from sklearn.svm import SVR
    
    svm = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)
    svm.fit(X, y)
    
    def predict(X_new):
        return svm.predict(X_new)
    
    return predict


def _cross_validate_predict(X, y, predict_fn, n_folds=5):
    """Simple cross-validation to estimate prediction error."""
    n = len(X)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    errors = []
    
    for i in range(n_folds):
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Recreate predictor on training fold
        try:
            pred_fn = predict_fn(X_train, y_train)
            preds = pred_fn(X_val)
            mse = np.mean((preds - y_val) ** 2)
            var_y = np.var(y_val)
            if var_y > 1e-10:
                errors.append(mse / var_y)
        except Exception:
            errors.append(10.0)
    
    return np.mean(errors)


def fit(X_train, y_train):
    start_time = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    d = X.shape[1]
    n = len(X)
    
    # Normalize inputs for better conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Candidate model factories
    candidates = []
    
    # Polynomial models of various degrees
    for degree in [2, 3, 4]:
        candidates.append((f'poly_deg{degree}', lambda Xn, yn, d=degree: _make_poly_predict(Xn, yn, d)))
    
    # RBF with different numbers of centers
    for nc in [10, 20, 50]:
        candidates.append((f'rbf_c{nc}', lambda Xn, yn, nc=nc: _make_rbf_predict(Xn, yn, n_centers=nc)))
    
    # SVM models
    try:
        candidates.append(('svm_poly', lambda Xn, yn: _make_svm_poly_predict(Xn, yn)))
        candidates.append(('svm_rbf', lambda Xn, yn: _make_svm_rbf_predict(Xn, yn)))
    except ImportError:
        pass
    
    # Gaussian Process
    try:
        candidates.append(('gp', lambda Xn, yn: _make_gp_predict(Xn, yn)))
    except ImportError:
        pass
    
    # Evaluate candidates using cross-validation on normalized data
    best_predictor = None
    best_score = float('inf')
    best_name = 'poly_deg2'
    
    for name, factory in candidates:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.7:
            break
        
        try:
            # Quick fit on full training set to check if it works
            pred_fn = factory(X_norm, y)
            test_pred = pred_fn(X_norm[:10])
            if not np.all(np.isfinite(test_pred)):
                continue
            
            # Cross-validate
            cv_score = _cross_validate_predict(X_norm, y, factory)
            
            if cv_score < best_score:
                best_score = cv_score
                best_predictor = pred_fn
                best_name = name
        except Exception:
            continue
    
    # If no model worked, fall back to simple polynomial
    if best_predictor is None:
        best_predictor = _make_poly_predict(X_norm, y, 2)
        best_name = 'fallback_poly'
    
    # Wrap predictor to denormalize inputs
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_mean) / X_std
        preds = best_predictor(X_new_norm)
        return preds
    
    return predict
# EVOLVE-BLOCK-END
