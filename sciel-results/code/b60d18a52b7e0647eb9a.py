"""Ensemble-based regression with RBF kernel and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _poly_features(X, degree):
    """Polynomial features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)

def _trig_features(X, n_harmonics=5):
    """Trigonometric features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _mixed_features(X, poly_deg=3, n_harmonics=3):
    """Mixed polynomial and trigonometric features."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, poly_deg + 1):
        cols.append(x ** k)
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _rational_features(X, num_deg=2, den_deg=1):
    """Rational function features: p(x)/q(x) expanded."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, num_deg + 1):
        cols.append(x ** k)
    for k in range(1, den_deg + 1):
        cols.append(x ** k / (1 + x ** 2))
    return np.column_stack(cols)

def _exp_features(X):
    """Exponential and log features."""
    x = X[:, 0]
    cols = [np.ones(len(x)), x, x ** 2]
    x_clipped = np.clip(x, -10, 10)
    cols.append(np.exp(x_clipped))
    cols.append(np.exp(-x_clipped))
    x_safe = np.where(np.abs(x) > 1e-6, x, 1e-6)
    cols.append(np.log(np.abs(x_safe)))
    return np.column_stack(cols)

def _fit_model(features_func, X_train, y_train, max_degree=None):
    """Fit a linear model on given features, return (mse, predict_func)."""
    try:
        if max_degree is not None and hasattr(features_func, '__name__'):
            if 'poly' in features_func.__name__:
                Xf = features_func(X_train, max_degree)
            else:
                Xf = features_func(X_train)
        else:
            Xf = features_func(X_train)
        
        w, residuals, rank, sv = np.linalg.lstsq(Xf, y_train, rcond=None)
        y_pred = Xf @ w
        mse = np.mean((y_pred - y_train) ** 2)
        
        def predict(X):
            if max_degree is not None and hasattr(features_func, '__name__'):
                if 'poly' in features_func.__name__:
                    Xf_test = features_func(X, max_degree)
                else:
                    Xf_test = features_func(X)
            else:
                Xf_test = features_func(X)
            return Xf_test @ w
        
        return mse, predict
    except Exception:
        return float('inf'), None

def _rbf_fit(X_train, y_train, bandwidths=None):
    """Fit RBF kernel regression with multiple bandwidths, return best (mse, predict)."""
    try:
        x = X_train[:, 0].reshape(-1, 1)
        n = len(x)
        
        if bandwidths is None:
            diffs = np.abs(x - x.T)
            np.fill_diagonal(diffs, np.inf)
            med_dist = np.median(diffs[diffs > 0])
            bandwidths = [med_dist * f for f in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
        
        best_mse = float('inf')
        best_predict = None
        
        for bw in bandwidths:
            if _time_left() < 3.0:
                break
            try:
                diff = x - x.T
                K = np.exp(-diff ** 2 / (2 * bw ** 2))
                K += 1e-8 * np.eye(n)
                
                w = np.linalg.solve(K, y_train)
                y_pred = K @ w
                mse = np.mean((y_pred - y_train) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    w_copy = w.copy()
                    bw_copy = bw
                    x_train_copy = x.copy()
                    
                    def predict_rbf(X, w=w_copy, bw=bw_copy, x_train=x_train_copy):
                        x_new = X[:, 0].reshape(-1, 1)
                        diff = x_new - x_train.T
                        K_new = np.exp(-diff ** 2 / (2 * bw ** 2))
                        return K_new @ w
                    
                    best_predict = predict_rbf
            except Exception:
                continue
        
        return best_mse, best_predict
    except Exception:
        return float('inf'), None

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    d = X_train.shape[1]
    if d != 1:
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    # Portfolio of candidate models
    candidates = []
    
    # Polynomial models of various degrees
    for deg in [2, 3, 4, 5, 6, 7, 8]:
        candidates.append(('poly', deg))
    
    # Trigonometric models
    for n in [3, 5, 7, 10]:
        candidates.append(('trig', n))
    
    # Mixed models
    for p, t in [(2,2), (3,3), (4,4)]:
        candidates.append(('mixed', (p, t)))
    
    # Rational models
    for nd, dd in [(2,1), (3,1), (3,2)]:
        candidates.append(('rational', (nd, dd)))
    
    # Exponential models
    candidates.append(('exp', None))
    
    # RBF kernel models
    candidates.append(('rbf', None))
    
    # Fit all candidates and collect results
    valid_models = []
    
    for model_type, param in candidates:
        if _time_left() < 5.0:
            break
        
        if model_type == 'poly':
            mse, predict = _fit_model(_poly_features, X_train, y_train, max_degree=param)
        elif model_type == 'trig':
            def trig_fit(X, n=param):
                return _trig_features(X, n_harmonics=n)
            Xf = trig_fit(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w, n=param: _trig_features(X, n_harmonics=n) @ w
        elif model_type == 'mixed':
            p, t = param
            def mixed_fit(X, p=p, t=t):
                return _mixed_features(X, poly_deg=p, n_harmonics=t)
            Xf = mixed_fit(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w, p=p, t=t: _mixed_features(X, poly_deg=p, n_harmonics=t) @ w
        elif model_type == 'rational':
            nd, dd = param
            def rat_fit(X, nd=nd, dd=dd):
                return _rational_features(X, num_deg=nd, den_deg=dd)
            Xf = rat_fit(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w, nd=nd, dd=dd: _rational_features(X, num_deg=nd, den_deg=dd) @ w
        elif model_type == 'exp':
            Xf = _exp_features(X_train)
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
            y_pred = Xf @ w
            mse = np.mean((y_pred - y_train) ** 2)
            predict = lambda X, w=w: _exp_features(X) @ w
        elif model_type == 'rbf':
            mse, predict = _rbf_fit(X_train, y_train)
        else:
            continue
        
        if predict is not None and np.isfinite(mse):
            valid_models.append((mse, predict))
    
    # If no model worked, fallback to degree-3 polynomial
    if not valid_models:
        Xf = _poly_features(X_train, 3)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        return lambda X, w=w: _poly_features(X, 3) @ w
    
    # Sort by MSE and select top K for ensemble
    valid_models.sort(key=lambda x: x[0])
    
    # Select top 5 models for ensemble to reduce variance
    top_k = min(5, len(valid_models))
    ensemble_models = [m[1] for m in valid_models[:top_k]]
    
    def predict_ensemble(X):
        preds = np.zeros(len(X))
        for model in ensemble_models:
            preds += model(X)
        preds /= len(ensemble_models)
        # Ensure finite values
        preds = np.nan_to_num(preds, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
        return preds
    
    return predict_ensemble
# EVOLVE-BLOCK-END
