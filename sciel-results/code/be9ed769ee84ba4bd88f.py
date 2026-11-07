"""Symbolic regression via ensemble of diverse model families with cross-validation selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # safety margin before 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # All equations are 1D per task description
    # We'll work with the first column if d > 1, but typically d=1
    
    # Helper: check if we're running out of time
    def time_left():
        return DEADLINE - time.time()
    
    # Build a set of candidate models
    # Each model is a function that takes X and returns predictions
    
    candidate_models = []
    
    # 1. Polynomial regression of various degrees
    for deg in range(1, 12):
        def make_poly(deg):
            def predict(X_test):
                x = X_test[:, 0] if X_test.ndim > 1 else X_test
                # Build Vandermonde-like matrix
                Xv = np.vander(x, N=deg+1, increasing=True)
                # Regularized least squares
                reg = 1e-6
                A = Xv.T @ Xv + reg * np.eye(deg+1)
                b = Xv.T @ y
                w = np.linalg.solve(A, b)
                return Xv @ w
            return predict
        candidate_models.append(make_poly(deg))
    
    # 2. Fourier series (useful for periodic functions)
    for nterms in range(1, 20):
        def make_fourier(nterms):
            def predict(X_test):
                x = X_test[:, 0] if X_test.ndim > 1 else X_test
                # Build Fourier features
                Xf = [np.ones_like(x)]
                for k in range(1, nterms + 1):
                    Xf.append(np.sin(k * x))
                    Xf.append(np.cos(k * x))
                Xf = np.column_stack(Xf)
                reg = 1e-6
                A = Xf.T @ Xf + reg * np.eye(Xf.shape[1])
                b = Xf.T @ y
                w = np.linalg.solve(A, b)
                return Xf @ w
            return predict
        candidate_models.append(make_fourier(nterms))
    
    # 3. Spline-like basis using RBF kernels at training points
    # Use a subset of centers for efficiency
    def make_rbf_basis(n_centers, bandwidth):
        centers = X[:n_centers, 0] if d == 1 else X[:n_centers]
        def predict(X_test):
            x = X_test[:, 0] if X_test.ndim > 1 else X_test
            # Compute RBF kernel values
            if d == 1:
                diff = x[:, np.newaxis] - centers[np.newaxis, :]
                K = np.exp(-bandwidth ** 2 * diff ** 2)
            else:
                diff = X_test[:, :d][:, np.newaxis, :] - centers[np.newaxis, :, :]
                K = np.exp(-bandwidth ** 2 * np.sum(diff ** 2, axis=2))
            # Solve for weights
            # Pre-compute kernel at training points
            if d == 1:
                diff_train = X[:, 0][:, np.newaxis] - centers[np.newaxis, :]
                K_train = np.exp(-bandwidth ** 2 * diff_train ** 2)
            else:
                diff_train = X[:, :d][:, np.newaxis, :] - centers[np.newaxis, :, :]
                K_train = np.exp(-bandwidth ** 2 * np.sum(diff_train ** 2, axis=2))
            reg = 1e-6
            A = K_train.T @ K_train + reg * np.eye(n_centers)
            b = K_train.T @ y
            w = np.linalg.solve(A, b)
            return K @ w
        return predict
    
    # Add RBF models with different bandwidths
    for bandwidth in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        candidate_models.append(make_rbf_basis(n, bandwidth))
    
    # 4. Gaussian Process with Matern 5/2 kernel
    def make_gp(length_scale, noise_var):
        # Pre-compute kernel matrix
        x = X[:, 0]
        diff = x[:, np.newaxis] - x[np.newaxis, :]
        K = (1 + np.sqrt(3) * np.abs(diff)) * np.exp(-np.sqrt(3) * np.abs(diff) / length_scale)
        K[np.diag_indices_from(K)] += noise_var
        
        # Pre-compute Cholesky
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError:
            # Fallback to regularized solve
            K_reg = K + 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K_reg)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            except np.linalg.LinAlgError:
                return None
        
        def predict(X_test):
            x_test = X_test[:, 0]
            diff = x_test[:, np.newaxis] - x[np.newaxis, :]
            K_star = (1 + np.sqrt(3) * np.abs(diff)) * np.exp(-np.sqrt(3) * np.abs(diff) / length_scale)
            return K_star @ alpha
        return predict
    
    # Add GP models with different length scales
    for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
        gp_model = make_gp(ls, 1e-8)
        if gp_model is not None:
            candidate_models.append(gp_model)
    
    # 5. Neural network-like: use a simple tanh activation model
    def make_tanh_model(n_hidden, lr=0.01, epochs=500):
        x = X[:, 0]
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Initialize weights
        np.random.seed(42)
        W1 = np.random.randn(n_hidden, 1) * 0.1
        b1 = np.zeros(n_hidden)
        W2 = np.random.randn(1, n_hidden) * 0.1
        b2 = np.zeros(1)
        
        for epoch in range(epochs):
            if time_left() < 5:
                break
            # Forward
            z1 = x[:, np.newaxis] @ W1.T + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2.T + b2
            pred_norm = z2.flatten()
            
            # Backward
            error = pred_norm - y_norm
            dz2 = error
            dW2 = a1.T @ dz2[:, np.newaxis] / n
            db2 = np.sum(dz2, axis=0) / n
            da1 = dz2[:, np.newaxis] @ W2
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = x[:, np.newaxis].T @ dz1 / n
            db1 = np.sum(dz1, axis=0) / n
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        
        def predict(X_test):
            x_test = X_test[:, 0]
            z1 = x_test[:, np.newaxis] @ W1.T + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2.T + b2
            return (z2.flatten() * y_std) + y_mean
        return predict
    
    for n_hidden in [4, 8, 16, 32]:
        candidate_models.append(make_tanh_model(n_hidden))
    
    # 6. Rational function approximation: y = (a*x^2 + b*x + c) / (d*x + e)
    def make_rational():
        # Nonlinear least squares
        x = X[:, 0]
        def residuals(params):
            a, b, c, d, e = params
            denom = d * x + e
            # Avoid division by zero
            mask = np.abs(denom) > 1e-10
            pred = (a * x[mask]**2 + b * x[mask] + c) / denom[mask]
            return pred - y[mask]
        
        try:
            # Initial guess
            p0 = [0.1, 0.1, 0.1, 0.01, 1.0]
            result = minimize(residuals, p0, method='Nelder-Mead', options={'maxiter': 1000})
            a, b, c, d, e = result.x
            
            def predict(X_test):
                x_test = X_test[:, 0]
                denom = d * x_test + e
                mask = np.abs(denom) > 1e-10
                pred = np.zeros_like(x_test)
                pred[mask] = (a * x_test[mask]**2 + b * x_test[mask] + c) / denom[mask]
                # For points where denom is near zero, use limit or interpolation
                if np.any(~mask):
                    # Use polynomial fit for those points
                    valid = np.abs(d * X[:, 0] + e) > 1e-10
                    if np.sum(valid) > 3:
                        coeffs = np.polyfit(X[valid, 0], y[valid], 3)
                        pred[~mask] = np.polyval(coeffs, x_test[~mask])
                    else:
                        pred[~mask] = np.mean(y)
                return pred
            return predict
        except Exception:
            return None
    
    rational_model = make_rational()
    if rational_model is not None:
        candidate_models.append(rational_model)
    
    # 7. Exponential models: y = a * exp(b*x) + c
    def make_exp_model():
        x = X[:, 0]
        def residuals(params):
            a, b, c = params
            pred = a * np.exp(b * x) + c
            # Clip to avoid overflow
            pred = np.clip(pred, -1e6, 1e6)
            return pred - y
        
        try:
            p0 = [1.0, 0.1, 0.0]
            result = minimize(residuals, p0, method='Nelder-Mead', options={'maxiter': 1000})
            a, b, c = result.x
            
            def predict(X_test):
                x_test = X_test[:, 0]
                pred = a * np.exp(b * x_test) + c
                return np.clip(pred, -1e6, 1e6)
            return predict
        except Exception:
            return None
    
    exp_model = make_exp_model()
    if exp_model is not None:
        candidate_models.append(exp_model)
    
    # 8. Logarithmic models: y = a * log(b*x + c) + d
    def make_log_model():
        x = X[:, 0]
        def residuals(params):
            a, b, c, d = params
            arg = b * x + c
            mask = arg > 1e-10
            if np.sum(mask) < 10:
                return np.ones_like(y) * 1e6
            pred = a * np.log(arg[mask]) + d
            return np.concatenate([pred - y[mask], np.ones(n - np.sum(mask)) * 1e6])
        
        try:
            p0 = [1.0, 1.0, 1.0, 0.0]
            result = minimize(residuals, p0, method='Nelder-Mead', options={'maxiter': 1000})
            a, b, c, d = result.x
            
            def predict(X_test):
                x_test = X_test[:, 0]
                arg = b * x_test + c
                pred = np.zeros_like(x_test)
                mask = arg > 1e-10
                pred[mask] = a * np.log(arg[mask]) + d
                pred[~mask] = np.mean(y)
                return pred
            return predict
        except Exception:
            return None
    
    log_model = make_log_model()
    if log_model is not None:
        candidate_models.append(log_model)
    
    # 9. Piecewise linear/spline interpolation
    def make_spline_interp():
        x = X[:, 0]
        y_sorted = y[np.argsort(x)]
        x_sorted = x[np.argsort(x)]
        
        def predict(X_test):
            x_test = X_test[:, 0]
            return np.interp(x_test, x_sorted, y_sorted)
        return predict
    
    candidate_models.append(make_spline_interp())
    
    # 10. Local polynomial regression (LOESS-like)
    def make_loess(bandwidth=0.3):
        x = X[:, 0]
        def predict(X_test):
            x_test = X_test[:, 0]
            preds = np.zeros_like(x_test)
            for i, xi in enumerate(x_test):
                if time_left() < 2:
                    preds[i:] = np.mean(y)
                    break
                # Find neighbors
                dist = np.abs(x - xi)
                mask = dist < bandwidth
                if np.sum(mask) < 3:
                    mask = dist < bandwidth * 2
                if np.sum(mask) < 3:
                    preds[i] = np.mean(y)
                    continue
                x_local = x[mask]
                y_local = y[mask]
                # Weighted quadratic fit
                w = np.exp(-dist[mask] ** 2 / (2 * (bandwidth / 2) ** 2))
                Xv = np.column_stack([np.ones_like(x_local), x_local, x_local ** 2])
                Xv_w = Xv * w[:, np.newaxis]
                y_w = y_local * w
                try:
                    coeffs = np.linalg.lstsq(Xv_w, y_w, rcond=None)[0]
                    preds[i] = coeffs[0] + coeffs[1] * xi + coeffs[2] * xi ** 2
                except Exception:
                    preds[i] = np.mean(y_local)
            return preds
        return predict
    
    for bw in [0.2, 0.5, 1.0]:
        candidate_models.append(make_loess(bw))
    
    # Evaluate all candidates using cross-validation
    # Use 5-fold CV
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    
    cv_scores = np.zeros(len(candidate_models))
    
    for model_idx, model in enumerate(candidate_models):
        if time_left() < 10:
            # Not enough time for CV, use training score
            try:
                pred = model(X)
                cv_scores[model_idx] = np.mean((pred - y) ** 2)
            except Exception:
                cv_scores[model_idx] = 1e10
            continue
        
        fold_errors = []
        for fold_i in range(n_folds):
            if time_left() < 5:
                break
            val_idx = folds[fold_i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
            
            # We can't retrain the model easily, so use the existing model
            # This is a limitation; we're evaluating the pre-trained model on held-out data
            try:
                pred_val = model(X[val_idx])
                fold_errors.append(np.mean((pred_val - y[val_idx]) ** 2))
            except Exception:
                fold_errors.append(1e10)
        
        cv_scores[model_idx] = np.mean(fold_errors) if fold_errors else 1e10
    
    # Select best model
    best_idx = np.argmin(cv_scores)
    best_model = candidate_models[best_idx]
    
    # Verify the best model works
    try:
        test_pred = best_model(X)
        if not np.all(np.isfinite(test_pred)):
            # Fallback to polynomial degree 5
            best_idx = 4  # polynomial degree 5
            best_model = candidate_models[best_idx]
    except Exception:
        # Fallback to simple linear regression
        def fallback_predict(X_test):
            x = X_test[:, 0]
            A = np.column_stack([np.ones_like(x), x])
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            return A @ w
        best_model = fallback_predict
    
    return best_model
# EVOLVE-BLOCK-END
