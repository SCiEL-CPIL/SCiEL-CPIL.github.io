"""Symbolic regression using an ensemble of Regularized Polynomial and Kernel Ridge Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    t_start = time.time()
    DEADLINE = 300.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    n, d = X.shape
    y_flat = y.flatten()

    # Standardize X
    mu_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    X_std = (X - mu_x) / std_x
    
    mu_y = y_flat.mean()
    std_y = y_flat.std()
    if std_y < 1e-8:
        std_y = 1.0
    y_std = (y_flat - mu_y) / std_y

    # Feature generator for polynomials
    def poly_features(X, degree):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        if degree >= 2:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    # Kernel function
    def rbf_kernel(X1, X2, gamma):
        sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T)
        sq_dist = np.maximum(sq_dist, 0)
        return np.exp(-gamma * sq_dist)

    # Model Fitters
    models = []
    
    # 1. Regularized Polynomial Regression
    for degree in [1, 2, 3]:
        X_feat = poly_features(X_std, degree)
        # Add regularization
        lam = 1e-2
        try:
            # Solve (X^T X + lam I) w = X^T y
            A = X_feat.T.dot(X_feat) + lam * np.eye(X_feat.shape[1])
            b = X_feat.T.dot(y_std)
            w = solve(A, b)
            models.append(('poly', degree, w, X_feat.shape[1]))
        except:
            pass

    # 2. Kernel Ridge Regression
    # Optimize gamma and alpha via small grid search
    best_krr_gamma = None
    best_krr_alpha = None
    best_krr_cv_err = float('inf')
    
    gammas = [0.01, 0.1, 1.0, 10.0]
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Simple 5-fold CV
    indices = np.arange(n)
    fold_size = n // 5
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(5)]
    folds[-1] = np.concatenate([folds[-1], indices[5*fold_size:]]) # Handle remainder
    
    for gamma in gammas:
        for alpha in alphas:
            cv_errors = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                
                X_tr = X_std[train_idx]
                y_tr = y_std[train_idx]
                X_val = X_std[val_idx]
                y_val = y_std[val_idx]
                
                try:
                    K_tr = rbf_kernel(X_tr, X_tr, gamma)
                    K_tr += alpha * np.eye(len(X_tr))
                    # Solve for weights
                    w_krr = solve(K_tr, y_tr)
                    
                    K_val = rbf_kernel(X_val, X_tr, gamma)
                    pred_val = K_val.dot(w_krr)
                    err = np.mean((pred_val - y_val)**2)
                    cv_errors.append(err)
                except:
                    cv_errors.append(float('inf'))
            
            mean_err = np.mean(cv_errors)
            if mean_err < best_krr_cv_err:
                best_krr_cv_err = mean_err
                best_krr_gamma = gamma
                best_krr_alpha = alpha
                
    if best_krr_gamma is not None and time.time() < DEADLINE - 10:
        K_tr = rbf_kernel(X_std, X_std, best_krr_gamma)
        K_tr += best_krr_alpha * np.eye(n)
        try:
            w_krr = solve(K_tr, y_std)
            models.append(('krr', best_krr_gamma, w_krr, None))
        except:
            pass

    # If no models were fitted (unlikely), fallback to constant
    if not models:
        def predict_const(X_test):
            return np.full(len(X_test), mu_y)
        return predict_const

    # Select best model based on training error (proxy for test error if CV was used for selection)
    # Or just average all models? Averaging often helps robustness.
    # Let's evaluate all models on a held-out subset of training data to pick the best, or just average.
    # Given small N, averaging is safer.
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        X_test_std = (X_test - mu_x) / std_x
        preds = []
        
        for model in models:
            try:
                if model[0] == 'poly':
                    _, degree, w, _ = model
                    X_feat_test = poly_features(X_test_std, degree)
                    pred = X_feat_test.dot(w)
                elif model[0] == 'krr':
                    _, gamma, w, _ = model
                    K_test = rbf_kernel(X_test_std, X_std, gamma)
                    pred = K_test.dot(w)
                else:
                    continue
                preds.append(pred)
            except:
                pass
        
        if not preds:
            return np.full(len(X_test), mu_y)
            
        # Average predictions
        pred_avg = np.mean(preds, axis=0)
        y_pred = pred_avg * std_y + mu_y
        y_pred = np.where(np.isfinite(y_pred), y_pred, mu_y)
        return y_pred

    return predict
# EVOLVE-BLOCK-END
