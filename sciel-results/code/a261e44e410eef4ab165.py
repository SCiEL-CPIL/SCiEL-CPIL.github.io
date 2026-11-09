"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement
import time

def _poly_features(X, d, max_degree):
    """Generate polynomial features up to max_degree for d inputs."""
    n = X.shape[0]
    cols = [np.ones(n)]
    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(d), degree):
            term = np.ones(n)
            for idx in combo:
                term = term * X[:, idx]
            cols.append(term)
    return np.column_stack(cols)

def _sine_cosine_features(X, d, max_freq=3):
    """Generate sine/cosine features."""
    n = X.shape[0]
    cols = []
    for j in range(d):
        for freq in range(1, max_freq + 1):
            cols.append(np.sin(freq * X[:, j]))
            cols.append(np.cos(freq * X[:, j]))
    return np.column_stack(cols)

def _exp_reciprocal_features(X, d):
    """Generate exp and reciprocal features."""
    n = X.shape[0]
    cols = []
    for j in range(d):
        cols.append(np.exp(X[:, j]))
        cols.append(1.0 / (X[:, j] + 1e-8))  # inputs >= 1, so safe
    return np.column_stack(cols)

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 320.0  # leave 10s safety margin
    
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    # Try increasing complexity levels
    best_predict = None
    best_score = np.inf
    
    # Level 1: Degree-2 polynomial
    try:
        feats = _poly_features(X_train, d, max_degree=2)
        w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
        pred = feats @ w
        score = np.mean((pred - y_train) ** 2) / max(np.var(y_train), 1e-12)
        if score < best_score and np.all(np.isfinite(pred)):
            best_score = score
            # Capture for prediction
            def make_predict(feats_func, w_val):
                def predict(X):
                    f = feats_func(X)
                    return f @ w_val
                return predict
            best_predict = make_predict(lambda X: _poly_features(X, d, max_degree=2), w)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_predict if best_predict else (lambda X: np.full(X.shape[0], np.mean(y_train)))
    
    # Level 2: Degree-3 polynomial
    try:
        feats = _poly_features(X_train, d, max_degree=3)
        w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
        pred = feats @ w
        score = np.mean((pred - y_train) ** 2) / max(np.var(y_train), 1e-12)
        if score < best_score and np.all(np.isfinite(pred)):
            best_score = score
            def make_predict(feats_func, w_val):
                def predict(X):
                    f = feats_func(X)
                    return f @ w_val
                return predict
            best_predict = make_predict(lambda X: _poly_features(X, d, max_degree=3), w)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_predict if best_predict else (lambda X: np.full(X.shape[0], np.mean(y_train)))
    
    # Level 3: Degree-4 polynomial
    try:
        feats = _poly_features(X_train, d, max_degree=4)
        w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
        pred = feats @ w
        score = np.mean((pred - y_train) ** 2) / max(np.var(y_train), 1e-12)
        if score < best_score and np.all(np.isfinite(pred)):
            best_score = score
            def make_predict(feats_func, w_val):
                def predict(X):
                    f = feats_func(X)
                    return f @ w_val
                return predict
            best_predict = make_predict(lambda X: _poly_features(X, d, max_degree=4), w)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_predict if best_predict else (lambda X: np.full(X.shape[0], np.mean(y_train)))
    
    # Level 4: Add trigonometric features
    try:
        poly_feats = _poly_features(X_train, d, max_degree=2)
        trig_feats = _sine_cosine_features(X_train, d, max_freq=2)
        feats = np.column_stack([poly_feats, trig_feats])
        w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
        pred = feats @ w
        score = np.mean((pred - y_train) ** 2) / max(np.var(y_train), 1e-12)
        if score < best_score and np.all(np.isfinite(pred)):
            best_score = score
            def make_predict(w_val):
                def predict(X):
                    p = _poly_features(X, d, max_degree=2)
                    t = _sine_cosine_features(X, d, max_freq=2)
                    f = np.column_stack([p, t])
                    return f @ w_val
                return predict
            best_predict = make_predict(w)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_predict if best_predict else (lambda X: np.full(X.shape[0], np.mean(y_train)))
    
    # Level 5: Add exp/reciprocal features
    try:
        poly_feats = _poly_features(X_train, d, max_degree=2)
        exp_feats = _exp_reciprocal_features(X_train, d)
        feats = np.column_stack([poly_feats, exp_feats])
        w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
        pred = feats @ w
        score = np.mean((pred - y_train) ** 2) / max(np.var(y_train), 1e-12)
        if score < best_score and np.all(np.isfinite(pred)):
            best_score = score
            def make_predict(w_val):
                def predict(X):
                    p = _poly_features(X, d, max_degree=2)
                    e = _exp_reciprocal_features(X, d)
                    f = np.column_stack([p, e])
                    return f @ w_val
                return predict
            best_predict = make_predict(w)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_predict if best_predict else (lambda X: np.full(X.shape[0], np.mean(y_train)))
    
    # Level 6: Degree-4 + trig
    try:
        poly_feats = _poly_features(X_train, d, max_degree=3)
        trig_feats = _sine_cosine_features(X_train, d, max_freq=2)
        feats = np.column_stack([poly_feats, trig_feats])
        w, *_ = np.linalg.lstsq(feats, y_train, rcond=None)
        pred = feats @ w
        score = np.mean((pred - y_train) ** 2) / max(np.var(y_train), 1e-12)
        if score < best_score and np.all(np.isfinite(pred)):
            best_score = score
            def make_predict(w_val):
                def predict(X):
                    p = _poly_features(X, d, max_degree=3)
                    t = _sine_cosine_features(X, d, max_freq=2)
                    f = np.column_stack([p, t])
                    return f @ w_val
                return predict
            best_predict = make_predict(w)
    except Exception:
        pass
    
    if best_predict is None:
        return lambda X: np.full(X.shape[0], np.mean(y_train))
    
    return best_predict
# EVOLVE-BLOCK-END
