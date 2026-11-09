"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 280.0  # Leave 50s margin for 330s limit
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    # If y is constant, return constant
    if y_var < 1e-10:
        return lambda X: np.full(len(X), y_mean)
    
    # Normalize inputs to [0, 1] for better conditioning
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-10] = 1.0
    
    def normalize(X):
        return (X - x_min) / x_range
    
    X_norm = normalize(X_train)
    
    # Try multiple feature sets and pick the best via cross-validation
    
    def cv_score(features_func, n_folds=5):
        """Cross-validation score (lower is better)"""
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        scores = []
        for i in range(n_folds):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            
            X_tr = X_norm[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_norm[val_idx]
            y_val = y_train[val_idx]
            
            try:
                F_tr = features_func(X_tr)
                F_val = features_func(X_val)
                
                # Add small ridge for stability
                lam = 1e-6
                w = np.linalg.lstsq(F_tr + lam * np.eye(F_tr.shape[1]) * np.mean(F_tr, axis=0), y_tr, rcond=None)[0]
                pred_val = F_val @ w
                score = np.mean((pred_val - y_val) ** 2) / y_var
                scores.append(score)
            except Exception:
                scores.append(1e6)
        return np.mean(scores)
    
    def build_poly_features(X, max_degree):
        """Polynomial features up to max_degree"""
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        # Interaction and higher-order terms
        for degree in range(2, max_degree + 1):
            for combo in product(range(X.shape[1]), repeat=degree):
                if sorted(combo) != list(combo):
                    continue  # Only include unique combinations (avoid duplicates)
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        return np.column_stack(cols)
    
    def build_chebyshev_features(X, max_degree):
        """Chebyshev polynomial features"""
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            for k in range(1, max_degree + 1):
                cols.append(np.polynomial.chebyshev.chebval(2*X[:, j] - 1, np.eye(max_degree+1, k)))
        return np.column_stack(cols)
    
    def build_sinusoidal_features(X, max_freq=5):
        """Sinusoidal features"""
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            for freq in range(1, max_freq + 1):
                cols.append(np.sin(freq * X[:, j]))
                cols.append(np.cos(freq * X[:, j]))
        return np.column_stack(cols)
    
    def build_rbf_features(X, n_centers=20, gamma=None):
        """RBF features using random centers"""
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # Sample centers from training data
        idx = np.random.choice(n, n_centers, replace=False)
        centers = X[idx]
        cols = []
        for center in centers:
            dist_sq = np.sum((X - center) ** 2, axis=1)
            cols.append(np.exp(-gamma * dist_sq))
        return np.column_stack(cols)
    
    # Candidate feature generators
    candidates = []
    
    # Polynomial features of various degrees
    for degree in [2, 3, 4]:
        candidates.append((f"poly_{degree}", lambda X, d=degree: build_poly_features(X, d)))
    
    # Chebyshev features
    for degree in [3, 5]:
        candidates.append((f"cheb_{degree}", lambda X, d=degree: build_chebyshev_features(X, d)))
    
    # Sinusoidal features
    candidates.append(("sin_3", lambda X: build_sinusoidal_features(X, 3)))
    candidates.append(("sin_5", lambda X: build_sinusoidal_features(X, 5)))
    
    # RBF features with different gammas
    for gamma in [0.5, 1.0, 2.0]:
        candidates.append((f"rbf_g{gamma}", lambda X, g=gamma: build_rbf_features(X, 30, g)))
    
    # Combined features: poly2 + sin
    def combined_features(X):
        return np.hstack([build_poly_features(X, 2), build_sinusoidal_features(X, 3)])
    candidates.append(("poly2_sin", combined_features))
    
    # Evaluate candidates with cross-validation
    best_score = 1e10
    best_func = None
    best_name = "none"
    
    for name, func in candidates:
        if time.time() > DEADLINE - 30:
            break
        try:
            score = cv_score(func)
            if score < best_score:
                best_score = score
                best_func = func
                best_name = name
        except Exception:
            continue
    
    # If no good candidate found, use a simple linear model as fallback
    if best_func is None:
        best_func = lambda X: np.column_stack([np.ones(len(X)), X])
    
    # Final fit on all data with the best feature set
    try:
        F_train = best_func(X_norm)
        lam = 1e-5
        w = np.linalg.lstsq(F_train + lam * np.eye(F_train.shape[1]) * np.mean(F_train, axis=0), y_train, rcond=None)[0]
    except Exception:
        # Fallback to linear
        F_train = np.column_stack([np.ones(len(X_train)), X_norm])
        w = np.linalg.lstsq(F_train, y_train, rcond=None)[0]
    
    def predict(X):
        X_n = normalize(X)
        F = best_func(X_n)
        return F @ w
    
    return predict
# EVOLVE-BLOCK-END
