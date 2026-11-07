"""
Strategy: Adaptive Ensemble Regression with Multiple Basis Functions.
We generate several candidate feature spaces (polynomial, Fourier, RBF, spline-like)
and fit a linear model on each. We use cross-validation to select the best basis
for each equation, allowing adaptation to different functional forms (periodic,
polynomial, smooth local). A time guard ensures we finish before the deadline.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _features_polynomial(X, degree=5):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for idx in itertools.combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for j in idx:
                feat *= X[:, j]
            cols.append(feat)
    return np.column_stack(cols)

def _features_fourier(X, n_freqs=8):
    """Generate Fourier features (sin and cos of multiples of x)."""
    n, d = X.shape
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        for k in range(1, n_freqs + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _features_rbf(X, n_centers=20):
    """Generate RBF features with centers spread across the data range."""
    n, d = X.shape
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        min_val = np.min(x)
        max_val = np.max(x)
        centers = np.linspace(min_val, max_val, n_centers)
        # Use a fixed bandwidth based on data range
        bandwidth = (max_val - min_val) / n_centers * 1.5
        for c in centers:
            cols.append(np.exp(-((x - c) ** 2) / (2 * bandwidth ** 2)))
    return np.column_stack(cols)

def _features_spline(X, n_knots=15):
    """Generate B-spline-like features using piecewise linear basis."""
    n, d = X.shape
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        min_val = np.min(x)
        max_val = np.max(x)
        knots = np.linspace(min_val, max_val, n_knots)
        for i in range(len(knots) - 1):
            left = knots[i]
            right = knots[i + 1]
            # Hat function centered between left and right
            center = (left + right) / 2
            width = (right - left) / 2
            feat = np.clip(1 - np.abs(x - center) / width, 0, 1)
            cols.append(feat)
    return np.column_stack(cols)

def _features_log(X):
    """Generate log features for potentially multiplicative relationships."""
    n, d = X.shape
    cols = [np.ones(n)]
    for j in range(d):
        x = X[:, j]
        # Handle sign for log
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.sign(x) * np.log1p(np.abs(x)))
    return np.column_stack(cols)

def _fit_linear(features, y):
    """Fit linear model with regularization."""
    # Add small ridge regularization for stability
    reg = 1e-6
    n, p = features.shape
    A = features.T @ features + reg * np.eye(p)
    b = features.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(features, y, rcond=None)[0]
    return w

def _cross_validate(features, y, n_folds=5):
    """K-fold cross-validation to estimate generalization error."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    mse_sum = 0.0
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train = features[train_idx]
        y_train = y[train_idx]
        X_val = features[val_idx]
        y_val = y[val_idx]
        
        w = _fit_linear(X_train, y_train)
        y_pred = X_val @ w
        mse = np.mean((y_pred - y_val) ** 2)
        mse_sum += mse
    
    return mse_sum / n_folds

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0  # seconds
    
    X = X_train
    y = y_train
    n, d = X.shape
    
    # Generate candidate feature sets
    candidates = []
    
    # Polynomial of various degrees
    for deg in [2, 3, 4, 5]:
        try:
            F = _features_polynomial(X, degree=deg)
            candidates.append(('poly_deg%d' % deg, F))
        except Exception:
            pass
    
    # Fourier with various frequencies
    for nf in [4, 6, 8, 12]:
        try:
            F = _features_fourier(X, n_freqs=nf)
            candidates.append(('fourier_%d' % nf, F))
        except Exception:
            pass
    
    # RBF with various numbers of centers
    for nc in [10, 15, 20, 30]:
        try:
            F = _features_rbf(X, n_centers=nc)
            candidates.append(('rbf_%d' % nc, F))
        except Exception:
            pass
    
    # Spline-like
    for nk in [10, 15, 20]:
        try:
            F = _features_spline(X, n_knots=nk)
            candidates.append(('spline_%d' % nk, F))
        except Exception:
            pass
    
    # Log features
    try:
        F = _features_log(X)
        candidates.append(('log', F))
    except Exception:
        pass
    
    # Evaluate each candidate via cross-validation
    best_mse = np.inf
    best_w = None
    best_F = None
    
    for name, F in candidates:
        if time.time() > deadline - 10:
            break
        try:
            mse = _cross_validate(F, y, n_folds=5)
            if mse < best_mse:
                best_mse = mse
                best_F = F
                best_w = _fit_linear(F, y)
        except Exception:
            continue
    
    # If no candidate worked, fall back to simple linear
    if best_w is None:
        F = np.column_stack([np.ones(n), X])
        best_w = _fit_linear(F, y)
        best_F = F
    
    def predict(X_new):
        if time.time() > deadline - 1:
            # Emergency fallback: return mean of training y
            return np.full(X_new.shape[0], np.mean(y))
        F_new = _transform(X_new, best_F, X)
        return F_new @ best_w
    
    return predict

def _transform(X_new, F_train, X_train):
    """
    Transform new data using the same feature mapping as the training data.
    We infer the feature type from the training features and re-apply.
    """
    n_new, d_new = X_new.shape
    n_train, d_train = X_train.shape
    
    # Infer which feature set was used by comparing column counts
    n_cols = F_train.shape[1]
    
    # Try to match the feature generator
    # We'll just re-generate all candidates and pick the one with matching width
    # But that's expensive. Instead, we store the feature type info.
    
    # For simplicity and speed, we'll use a heuristic:
    # The best_F was generated from one of our known generators.
    # We can try each generator and see which one produces the right number of columns.
    
    # Actually, let's just re-generate the same features.
    # We need to know which generator was used. Let's store that info.
    
    # Since we don't have easy access to which one was best, let's use a different approach:
    # We'll fit the model in a way that we can easily transform.
    
    # For now, let's just assume we can regenerate. We'll modify the approach.
    
    # Actually, the cleanest way is to store the feature function.
    # Let's refactor to pass the feature function.
    
    # For this version, we'll just try all candidates again and pick the one
    # with matching width. This is a bit wasteful but correct.
    
    for name, F_func in _get_feature_functions():
        try:
            F_test = F_func(X_train)
            if F_test.shape[1] == n_cols:
                return F_func(X_new)
        except Exception:
            continue
    
    # Fallback: linear
    return np.column_stack([np.ones(n_new), X_new])

def _get_feature_functions():
    """Return list of (name, function) pairs for feature generation."""
    funcs = []
    for deg in [2, 3, 4, 5]:
        funcs.append(('poly_deg%d' % deg, lambda X, d=deg: _features_polynomial(X, degree=d)))
    for nf in [4, 6, 8, 12]:
        funcs.append(('fourier_%d' % nf, lambda X, n=nf: _features_fourier(X, n_freqs=n)))
    for nc in [10, 15, 20, 30]:
        funcs.append(('rbf_%d' % nc, lambda X, n=nc: _features_rbf(X, n_centers=n)))
    for nk in [10, 15, 20]:
        funcs.append(('spline_%d' % nk, lambda X, n=nk: _features_spline(X, n_knots=n)))
    funcs.append(('log', _features_log))
    return funcs
# EVOLVE-BLOCK-END
