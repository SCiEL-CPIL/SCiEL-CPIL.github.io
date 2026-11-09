# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin for 330s kill
    
    n, d = X_train.shape
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_mean = np.mean(y_train)
        return lambda X: np.full(X.shape[0], y_mean)
    
    def time_ok():
        return time.time() < deadline
    
    def build_poly_features(X, degree):
        """Build polynomial features up to given degree."""
        cols = [np.ones(len(X))]
        # Degree 1
        for j in range(d):
            cols.append(X[:, j])
        # Higher degrees
        for deg in range(2, degree + 1):
            for idx in product(range(d), repeat=deg):
                # Only include sorted indices to avoid duplicates
                sorted_idx = tuple(sorted(idx))
                # Check if this combination was already added
                term = np.ones(len(X))
                for k in range(deg):
                    term = term * X[:, sorted_idx[k]]
                cols.append(term)
        return np.column_stack(cols)
    
    def build_trig_features(X):
        """Build trigonometric features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
        return np.column_stack(cols)
    
    def build_rbf_features(X, centers, bandwidths):
        """Build RBF features."""
        cols = [np.ones(len(X))]
        for c in centers:
            for b in bandwidths:
                dist_sq = np.sum((X - c) ** 2, axis=1)
                cols.append(np.exp(-b * dist_sq))
        return np.column_stack(cols)
    
    def fit_and_score(X, y, features):
        """Fit linear model and return training MSE."""
        try:
            w, residuals, rank, sv = np.linalg.lstsq(features, y, rcond=None)
            pred = features @ w
            mse = np.mean((pred - y) ** 2)
            return mse, w
        except:
            return np.inf, None
    
    # Strategy: try multiple feature sets, pick best by training MSE
    # Since data is noiseless, training MSE should be very small for the right model
    
    best_mse = np.inf
    best_predictor = None
    
    # 1. Try polynomial features of various degrees
    for degree in [2, 3, 4, 5]:
        if not time_ok():
            break
        try:
            feat = build_poly_features(X_train, degree)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_poly_features, dg=degree, ww=w: f(X, dg) @ ww
        except:
            pass
    
    # 2. Try trigonometric features
    if time_ok():
        try:
            feat = build_trig_features(X_train)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_trig_features, ww=w: f(X) @ ww
        except:
            pass
    
    # 3. Try RBF features with a few centers
    if time_ok():
        # Use some training points as centers
        n_centers = min(10, n)
        indices = np.random.choice(n, n_centers, replace=False)
        centers = X_train[indices]
        bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
        try:
            feat = build_rbf_features(X_train, centers, bandwidths)
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f=build_rbf_features, cs=centers, bs=bandwidths, ww=w: f(X, cs, bs) @ ww
        except:
            pass
    
    # 4. Try a combined feature set
    if time_ok():
        try:
            feat1 = build_poly_features(X_train, 3)
            feat2 = build_trig_features(X_train)
            feat = np.column_stack([feat1, feat2])
            mse, w = fit_and_score(X_train, y_train, feat)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda X, f1=build_poly_features, f2=build_trig_features, dg=3, ww=w: np.column_stack([f1(X, dg), f2(X)]) @ ww
        except:
            pass
    
    # If no good model found, fall back to simple linear
    if best_predictor is None:
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_train]), y_train, rcond=None)
        best_predictor = lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
    
    return best_predictor
# EVOLVE-BLOCK-END
