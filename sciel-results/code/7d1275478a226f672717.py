import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin from 330s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for kernel stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Center y
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Split for validation
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_norm = X_norm[train_idx]
    y_tr = y_centered[train_idx]
    X_val_norm = X_norm[val_idx]
    y_val = y_centered[val_idx]
    
    # Model 1: Gaussian Process with Matern-5/2 kernel
    def matern_kernel(X1, X2, length_scale, signal_var, nu=2.5):
        """Compute Matern kernel with given hyperparameters."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        r = np.sqrt(sq_dists)
        if nu == 2.5:
            K = signal_var * (1 + np.sqrt(3) * r / length_scale) * np.exp(-np.sqrt(3) * r / length_scale)
        elif nu == 1.5:
            K = signal_var * (1 + np.sqrt(3) * r / length_scale) * np.exp(-np.sqrt(3) * r / length_scale)
        elif nu == 3.5:
            K = signal_var * (1 + np.sqrt(5) * r / length_scale + 5 * r**2 / (3 * length_scale**2)) * np.exp(-np.sqrt(5) * r / length_scale)
        else:
            K = signal_var * np.exp(-r / length_scale)
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    def gp_nll(params):
        """Negative log marginal likelihood for GP."""
        length_scale = np.exp(params[0])
        signal_var = np.exp(params[1])
        noise_var = np.exp(params[2])
        
        K = matern_kernel(X_tr_norm, X_tr_norm, length_scale, signal_var) + noise_var * np.eye(len(X_tr_norm))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            nll = 0.5 * y_tr @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y_tr) * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    # Hyperparameter search for GP
    best_gp_params = None
    best_gp_nll = np.inf
    
    length_scale_grid = np.logspace(-1, 1, 12)
    signal_var_grid = np.logspace(-1, 2, 8)
    noise_var_grid = np.logspace(-4, -1, 8)
    
    for ls in length_scale_grid:
        if time.time() > deadline - 80:
            break
        for sv in signal_var_grid:
            for nv in noise_var_grid:
                params = np.array([np.log(ls), np.log(sv), np.log(nv)])
                nll = gp_nll(params)
                if nll < best_gp_nll:
                    best_gp_nll = nll
                    best_gp_params = params
    
    if best_gp_params is not None:
        try:
            result = minimize(gp_nll, best_gp_params, method='L-BFGS-B', 
                             bounds=[(-5, 5), (-2, 5), (-8, -1)], maxiter=50)
            if result.fun < best_gp_nll:
                best_gp_nll = result.fun
                best_gp_params = result.x
        except:
            pass
    
    if best_gp_params is None or not np.isfinite(best_gp_params).all():
        best_gp_params = np.array([0.0, 0.0, -3.0])
    
    gp_ls = np.exp(best_gp_params[0])
    gp_sv = np.exp(best_gp_params[1])
    gp_nv = np.exp(best_gp_params[2])
    
    # Validate GP on holdout
    K_val_tr = matern_kernel(X_val_norm, X_tr_norm, gp_ls, gp_sv)
    K_tr_tr = matern_kernel(X_tr_norm, X_tr_norm, gp_ls, gp_sv) + gp_nv * np.eye(len(X_tr_norm))
    
    try:
        w = solve(K_tr_tr, y_tr, assume_a='pos')
        y_pred_val_gp = K_val_tr @ w
        val_mse_gp = np.mean((y_pred_val_gp - y_val) ** 2)
    except:
        val_mse_gp = np.inf
    
    # Model 2: High-degree Polynomial Kernel Ridge Regression
    def poly_kernel(X1, X2, degree, alpha=1.0):
        """Polynomial kernel."""
        K = (alpha * (X1 @ X2.T) + 1) ** degree
        if len(X1) == len(X2):
            K += 1e-8 * np.eye(len(X1))
        return K
    
    # Try different polynomial degrees and regularization
    best_poly_mse = np.inf
    best_poly_degree = 3
    best_poly_alpha = 1.0
    
    for degree in [2, 3, 4, 5]:
        if time.time() > deadline - 60:
            break
        for reg in np.logspace(-4, 2, 10):
            try:
                K = poly_kernel(X_tr_norm, X_tr_norm, degree) + reg * np.eye(len(X_tr_norm))
                w = solve(K, y_tr, assume_a='pos')
                K_val = poly_kernel(X_val_norm, X_tr_norm, degree)
                y_pred = K_val @ w
                mse = np.mean((y_pred - y_val) ** 2)
                if mse < best_poly_mse:
                    best_poly_mse = mse
                    best_poly_degree = degree
                    best_poly_alpha = reg
            except:
                continue
    
    # Model 3: Rational Function Approximator
    # Fit y = (sum_i a_i * phi_i(x)) / (sum_j b_j * psi_j(x)) using iterative approach
    def rational_fit(X_tr, y_tr, X_val, y_val, max_iter=20):
        """Fit rational function using iterative weighted least squares."""
        n = X_tr.shape[0]
        d = X_tr.shape[1]
        
        # Generate basis functions
        def gen_basis(X):
            feats = [np.ones(X.shape[0])]
            for i in range(d):
                feats.append(X[:, i])
            for i in range(d):
                for j in range(i, d):
                    feats.append(X[:, i] * X[:, j])
            # Add some non-linear terms
            for i in range(d):
                feats.append(np.exp(-X[:, i]))
                feats.append(np.log(X[:, i] + 1e-6))
            return np.column_stack(feats)
        
        X_b = gen_basis(X_tr)
        X_val_b = gen_basis(X_val)
        
        # Initialize denominator as 1
        denom = np.ones(n)
        
        best_mse = np.inf
        best_num_w = None
        best_den_w = None
        
        for it in range(max_iter):
            if time.time() > deadline - 40:
                break
            try:
                # Fit numerator: y * denom = num
                y_weighted = y_tr * denom
                A = X_b.T @ X_b + 1e-4 * np.eye(X_b.shape[1])
                b = X_b.T @ y_weighted
                num_w = solve(A, b, assume_a='pos')
                
                # Predict numerator
                num_pred = X_b @ num_w
                num_val_pred = X_val_b @ num_w
                
                # Fit denominator: 1 = denom (using current numerator prediction)
                # Use iterative approach: minimize sum((y_i * denom_i - num_i)^2)
                # Approximate: fit denom such that num_pred / denom ≈ y
                # This is tricky, so use a simpler approach
                
                # Alternative: use Nelder-Mead on log-likelihood
                # For now, just use polynomial model as fallback
                y_pred_val = num_val_pred / np.ones(X_val.shape[0])
                mse = np.mean((y_pred_val - y_val) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_num_w = num_w
                    best_den_w = np.ones(X_val.shape[0])
                    
            except:
                break
        
        return best_mse, best_num_w, best_den_w, X_b, X_val_b
    
    rational_mse, rat_num_w, rat_den_w, rat_X_b, rat_X_val_b = rational_fit(X_tr_norm, y_tr, X_val_norm, y_val)
    
    # Model 4: Rich feature Ridge Regression (from original)
    def generate_features(X):
        n = X.shape[0]
        d = X.shape[1]
        feats = [np.ones(n)]
        
        for i in range(d):
            feats.append(X[:, i])
        
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
        
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    if i <= j <= k:
                        feats.append(X[:, i] * X[:, j] * X[:, k])
        
        for i in range(d):
            feats.append(np.sin(X[:, i]))
            feats.append(np.cos(X[:, i]))
            feats.append(np.sin(2 * np.pi * X[:, i]))
            feats.append(np.cos(2 * np.pi * X[:, i]))
        
        for i in range(d):
            feats.append(np.exp(-2 * X[:, i]))
            feats.append(np.exp(-X[:, i]))
            feats.append(np.log(X[:, i] + 1e-6))
            feats.append(1.0 / (X[:, i] + 1e-6))
        
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    feats.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                    feats.append(np.sin(X[:, i]) * np.sin(X[:, j]))
        
        X_feat = np.column_stack(feats)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        return X_feat
    
    X_tr_feat = generate_features(X_tr_norm)
    X_val_feat = generate_features(X_val_norm)
    
    n_features = X_tr_feat.shape[1]
    best_ridge_mse = np.inf
    best_ridge_alpha = 1.0
    
    for alpha in np.logspace(-6, 2, 12):
        if time.time() > deadline - 30:
            break
        try:
            A = X_tr_feat.T @ X_tr_feat + alpha * np.eye(n_features)
            b = X_tr_feat.T @ y_tr
            w_ridge = solve(A, b, assume_a='pos')
            y_pred_ridge = X_val_feat @ w_ridge
            ridge_mse = np.mean((y_pred_ridge - y_val) ** 2)
            if ridge_mse < best_ridge_mse:
                best_ridge_mse = ridge_mse
                best_ridge_alpha = alpha
        except:
            continue
    
    # Select best model based on validation MSE
    models = {
        'gp': val_mse_gp,
        'poly': best_poly_mse,
        'rational': rational_mse,
        'ridge': best_ridge_mse
    }
    
    valid_models = {k: v for k, v in models.items() if np.isfinite(v)}
    
    if not valid_models:
        # Fallback to GP with default params
        use_model = 'gp'
    else:
        use_model = min(valid_models, key=valid_models.get)
    
    # Refit selected model on ALL training data
    if use_model == 'gp':
        K_full = matern_kernel(X_norm, X_norm, gp_ls, gp_sv) + gp_nv * np.eye(n)
        w_full = solve(K_full, y_centered, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = matern_kernel(X_test_norm, X_norm, gp_ls, gp_sv)
            return K_test @ w_full + y_mean
    
    elif use_model == 'poly':
        K_full = poly_kernel(X_norm, X_norm, best_poly_degree) + best_poly_alpha * np.eye(n)
        w_full = solve(K_full, y_centered, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = poly_kernel(X_test_norm, X_norm, best_poly_degree)
            return K_test @ w_full + y_mean
    
    elif use_model == 'ridge':
        X_all_feat = generate_features(X_norm)
        X_all_feat = np.nan_to_num(X_all_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        A_full = X_all_feat.T @ X_all_feat + best_ridge_alpha * np.eye(n_features)
        b_full = X_all_feat.T @ y_centered
        w_full = solve(A_full, b_full, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_features(X_test_norm)
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            return X_test_feat @ w_full + y_mean
    
    else:  # rational fallback to ridge
        X_all_feat = generate_features(X_norm)
        X_all_feat = np.nan_to_num(X_all_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        A_full = X_all_feat.T @ X_all_feat + best_ridge_alpha * np.eye(n_features)
        b_full = X_all_feat.T @ y_centered
        w_full = solve(A_full, b_full, assume_a='pos')
        
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat = generate_features(X_test_norm)
            X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=1e6, neginf=-1e6)
            return X_test_feat @ w_full + y_mean
    
    return predict
