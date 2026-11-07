import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # Leave 20s buffer for safety

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape

    # Handle edge case: constant y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        def predict_const(X_test):
            X_test = np.array(X_test, dtype=float)
            if X_test.ndim == 1:
                return np.array([y_mean])
            return np.full(X_test.shape[0], y_mean)
        return predict_const

    # Normalize X to zero mean, unit variance
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X - x_mean) / x_std

    # Normalize y
    y_norm = (y - y_mean) / y_std

    def _time_left():
        return deadline - time.time()

    # Generate a rich feature dictionary
    def generate_features(X):
        """Generate a rich set of features from input X."""
        features = [X]  # Start with original features
        feature_names = [f'x{i}' for i in range(X.shape[1])]
        
        # Polynomials up to degree 3
        for deg in range(2, 4):
            if deg == 2:
                for i in range(X.shape[1]):
                    features.append(X[:, i:i+1]**2)
                    feature_names.append(f'x{i}^2')
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        features.append(X[:, i:i+1] * X[:, j:j+1])
                        feature_names.append(f'x{i}*x{j}')
            elif deg == 3:
                for i in range(X.shape[1]):
                    features.append(X[:, i:i+1]**3)
                    feature_names.append(f'x{i}^3')
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        features.append(X[:, i:i+1]**2 * X[:, j:j+1])
                        feature_names.append(f'x{i}^2*x{j}')
                        features.append(X[:, i:i+1] * X[:, j:j+1]**2)
                        feature_names.append(f'x{i}*x{j}^2')
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        for k in range(j+1, X.shape[1]):
                            features.append(X[:, i:i+1] * X[:, j:j+1] * X[:, k:k+1])
                            feature_names.append(f'x{i}*x{j}*x{k}')
        
        # Trigonometric features
        for i in range(X.shape[1]):
            features.append(np.sin(X[:, i:i+1]))
            feature_names.append(f'sin(x{i})')
            features.append(np.cos(X[:, i:i+1]))
            feature_names.append(f'cos(x{i})')
            features.append(np.sin(2*X[:, i:i+1]))
            feature_names.append(f'sin(2*x{i})')
            features.append(np.cos(2*X[:, i:i+1]))
            feature_names.append(f'cos(2*x{i})')
        
        # Exponential and log features
        for i in range(X.shape[1]):
            features.append(np.exp(-X[:, i:i+1]**2))
            feature_names.append(f'exp(-x{i}^2)')
            features.append(np.log1p(np.abs(X[:, i:i+1])))
            feature_names.append(f'log1p(|x{i}|)')
            features.append(1.0 / (1.0 + np.exp(-X[:, i:i+1])))
            feature_names.append(f'sigmoid(x{i})')
        
        # Interactions with trig
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                features.append(np.sin(X[:, i:i+1]) * X[:, j:j+1])
                feature_names.append(f'sin(x{i})*x{j}')
                features.append(np.cos(X[:, i:i+1]) * X[:, j:j+1])
                feature_names.append(f'cos(x{i})*x{j}')
        
        # Combine all features
        if len(features) > 1:
            X_features = np.hstack(features)
        else:
            X_features = features[0]
        
        return X_features, feature_names

    # Generate features
    X_feat, feat_names = generate_features(X_norm)
    n_feat = X_feat.shape[1]
    
    # Standardize features
    feat_mean = X_feat.mean(axis=0)
    feat_std = X_feat.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_feat_std = (X_feat - feat_mean) / feat_std

    # Strategy: Use ridge regression with LASSO-like sparsity via iterative feature selection
    # First, try full ridge regression
    def ridge_solve(X, y, alpha):
        """Solve ridge regression problem."""
        try:
            # X^T X + alpha*I
            XtX = X.T @ X
            XtX += alpha * np.eye(X.shape[1])
            Xty = X.T @ y
            coef = np.linalg.solve(XtX, Xty)
            return coef
        except:
            return np.zeros(X.shape[1])

    # Compute LOOCV for ridge regression
    def loocv_ridge(X, y, alpha):
        """Compute LOOCV MSE for ridge regression."""
        n = X.shape[0]
        try:
            # H = X (X^T X + alpha I)^{-1} X^T
            XtX = X.T @ X + alpha * np.eye(X.shape[1])
            XtX_inv = np.linalg.inv(XtX)
            H = X @ XtX_inv @ X.T
            H_diag = np.diag(H)
            
            y_pred = H @ y
            denom = 1 - H_diag
            denom = np.maximum(np.abs(denom), 1e-8)
            
            errors = (y - y_pred) / denom
            mse = np.mean(errors**2)
            return mse
        except:
            return np.inf

    # Grid search over alpha for ridge regression
    alphas = np.logspace(-4, 4, 20)
    best_mse = np.inf
    best_alpha = 1.0
    
    for alpha in alphas:
        if _time_left() < 15:
            break
        mse = loocv_ridge(X_feat_std, y_norm, alpha)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    # Fit final ridge model
    coef = ridge_solve(X_feat_std, y_norm, best_alpha)
    
    # Check if we have time for feature selection
    if _time_left() > 60:
        # Simple forward selection
        selected = []
        remaining = list(range(n_feat))
        best_mse_sel = best_mse
        coef_sel = coef.copy()
        
        max_features = min(20, n_feat)
        for step in range(max_features):
            if _time_left() < 30:
                break
            
            best_feat = None
            best_mse_step = best_mse_sel
            
            for feat in remaining:
                test_selected = selected + [feat]
                X_sub = X_feat_std[:, test_selected]
                
                # Solve ridge on subset
                alpha_sub = best_alpha
                try:
                    XtX = X_sub.T @ X_sub + alpha_sub * np.eye(len(test_selected))
                    Xty = X_sub.T @ y_norm
                    coef_sub = np.linalg.solve(XtX, Xty)
                    
                    # Compute training MSE
                    y_pred_sub = X_sub @ coef_sub
                    mse_sub = np.mean((y_norm - y_pred_sub)**2)
                    
                    if mse_sub < best_mse_step:
                        best_mse_step = mse_sub
                        best_feat = feat
                        best_coef_step = coef_sub
                except:
                    continue
            
            if best_feat is None:
                break
            
            selected.append(best_feat)
            remaining.remove(best_feat)
            best_mse_sel = best_mse_step
            coef_sel = best_coef_step
        
        # Use selected features if they improved
        if len(selected) > 0 and best_mse_sel < best_mse * 0.9:
            X_feat_final = X_feat_std[:, selected]
            coef_final = coef_sel
            feat_mean_final = feat_mean[selected]
            feat_std_final = feat_std[selected]
        else:
            X_feat_final = X_feat_std
            coef_final = coef
            feat_mean_final = feat_mean
            feat_std_final = feat_std
    else:
        X_feat_final = X_feat_std
        coef_final = coef
        feat_mean_final = feat_mean
        feat_std_final = feat_std

    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test_norm = (X_test - x_mean) / x_std
        X_test_feat, _ = generate_features(X_test_norm)
        X_test_feat_std = (X_test_feat - feat_mean_final) / feat_std_final
        
        # If we used a subset of features, extract those
        if X_test_feat_std.shape[1] != coef_final.shape[0]:
            # This shouldn't happen, but handle it
            X_test_feat_std = X_test_feat_std[:, :coef_final.shape[0]]
        
        y_pred_norm = X_test_feat_std @ coef_final
        y_pred = y_pred_norm * y_std + y_mean
        
        return y_pred

    return predict
