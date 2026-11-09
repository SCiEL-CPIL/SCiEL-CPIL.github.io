"""Search-based symbolic regression with time budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Baseline: degree-2 polynomial
    def make_poly_features(X, degree=2):
        cols = [np.ones(len(X))]
        # Linear terms
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        # Quadratic terms
        if degree >= 2:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    # Fit baseline polynomial
    def fit_linear_model(features_train, y):
        try:
            w, *_ = np.linalg.lstsq(features_train, y, rcond=None)
            return w
        except:
            return None
    
    poly_feats = make_poly_features(X_train, 2)
    w_poly = fit_linear_model(poly_feats, y_train)
    best_pred = poly_feats @ w_poly if w_poly is not None else np.full(n, y_mean)
    best_nrmse = np.mean((best_pred - y_train)**2) / y_var
    best_features_func = lambda X: make_poly_features(X, 2)
    best_w = w_poly
    
    # Try higher degree polynomials
    for degree in [3, 4]:
        if time.time() > deadline - 30:
            break
        try:
            cols = [np.ones(n)]
            for j in range(d):
                cols.append(X_train[:, j])
            # Add higher order terms
            for k in range(2, degree + 1):
                # All monomials of degree k
                for idx in product(range(d), repeat=k):
                    if sorted(idx) == list(range(min(k, d))) and len(set(idx)) == k:
                        # Skip if we already have lower degree versions
                        pass
                    # Actually, let's just add all products
                    feat = np.ones(n)
                    for i in idx:
                        feat = feat * X_train[:, i]
                    # Only add if not duplicate
                    cols.append(feat)
            
            # This can get too large, so let's be more selective
            # Just add specific higher-order terms
            cols = [np.ones(n)]
            for j in range(d):
                cols.append(X_train[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X_train[:, i] * X_train[:, j])
            if degree >= 3:
                for i in range(d):
                    cols.append(X_train[:, i]**3)
                for i in range(d):
                    for j in range(i, d):
                        cols.append(X_train[:, i]**2 * X_train[:, j])
                        cols.append(X_train[:, i] * X_train[:, j]**2)
            if degree >= 4:
                for i in range(d):
                    cols.append(X_train[:, i]**4)
            
            feat_matrix = np.column_stack(cols)
            w_higher = fit_linear_model(feat_matrix, y_train)
            if w_higher is not None:
                pred = feat_matrix @ w_higher
                nrmse = np.mean((pred - y_train)**2) / y_var
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
                    best_w = w_higher
                    
                    def make_higher_features(X, deg=degree):
                        cols = [np.ones(len(X))]
                        for j in range(X.shape[1]):
                            cols.append(X[:, j])
                        for i in range(X.shape[1]):
                            for j in range(i, X.shape[1]):
                                cols.append(X[:, i] * X[:, j])
                        if deg >= 3:
                            for i in range(X.shape[1]):
                                cols.append(X[:, i]**3)
                            for i in range(X.shape[1]):
                                for j in range(i, X.shape[1]):
                                    cols.append(X[:, i]**2 * X[:, j])
                                    cols.append(X[:, i] * X[:, j]**2)
                        if deg >= 4:
                            for i in range(X.shape[1]):
                                cols.append(X[:, i]**4)
                        return np.column_stack(cols)
                    
                    best_features_func = lambda X, deg=degree: make_higher_features(X, deg)
        except:
            pass
    
    # Try some non-linear basis functions
    if time.time() < deadline - 20:
        # Add basis: x, x^2, x^3, sin(x), cos(x), exp(x), log(x), 1/x
        def make_rich_features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                x = X[:, j]
                cols.append(x)
                cols.append(x**2)
                cols.append(x**3)
                cols.append(np.sin(x))
                cols.append(np.cos(x))
                # Safe log
                cols.append(np.log(np.maximum(x, 1e-10)))
                # Safe reciprocal
                cols.append(1.0 / np.maximum(x, 1e-10))
            # Cross terms for pairs
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
                    cols.append(np.sin(X[:, i] * X[:, j]))
            return np.column_stack(cols)
        
        try:
            rich_feats = make_rich_features(X_train)
            # Use ridge regression for stability
            lam = 1e-6
            A = rich_feats.T @ rich_feats + lam * np.eye(rich_feats.shape[1])
            b = rich_feats.T @ y_train
            w_rich = np.linalg.solve(A, b)
            pred_rich = rich_feats @ w_rich
            nrmse_rich = np.mean((pred_rich - y_train)**2) / y_var
            if nrmse_rich < best_nrmse:
                best_nrmse = nrmse_rich
                best_w = w_rich
                best_features_func = make_rich_features
        except:
            pass
    
    # Try kernel ridge regression with RBF kernel
    if time.time() < deadline - 10:
        try:
            # Compute Gram matrix
            from scipy.spatial.distance import cdist
            dists = cdist(X_train, X_train, metric='sqeuclidean')
            
            # Try several bandwidths
            for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
                if time.time() > deadline - 5:
                    break
                K = np.exp(-gamma * dists)
                lam = 1e-3
                K_reg = K + lam * np.eye(n)
                alpha = np.linalg.solve(K_reg, y_train)
                
                # Evaluate on training
                pred_k = K @ alpha
                nrmse_k = np.mean((pred_k - y_train)**2) / y_var
                if nrmse_k < best_nrmse:
                    best_nrmse = nrmse_k
                    
                    def make_krbf_pred(X_test, gamma=gamma, alpha=alpha, X_tr=X_train):
                        dists_test = cdist(X_test, X_tr, metric='sqeuclidean')
                        K_test = np.exp(-gamma * dists_test)
                        return K_test @ alpha
                    
                    best_features_func = None
                    best_w = None
                    predict_fn = make_krbf_pred
                    return predict_fn
        except:
            pass
    
    # Return the best model found
    if best_w is not None:
        return lambda X: best_features_func(X) @ best_w
    else:
        return lambda X: np.full(X.shape[0], y_mean)
# EVOLVE-BLOCK-END
