import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 315.0  # Leave 15s buffer before 330s kill
    
    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape
    
    # Helper: check if we should stop
    def _time_left():
        return deadline - time.time()
    
    def _time_used():
        return time.time() - start_time
    
    # Build a rich basis of candidate features
    def _build_basis(X, d):
        n = X.shape[0]
        features = [np.ones(n)]
        feature_names = ['const']
        
        # Univariate features for each input
        for j in range(d):
            x = X[:, j]
            # Polynomials up to degree 6
            for k in range(1, 7):
                features.append(x**k)
                feature_names.append(f'x{j}^{k}')
            
            # Trigonometric
            features.append(np.sin(x))
            feature_names.append(f'sin(x{j})')
            features.append(np.cos(x))
            feature_names.append(f'cos(x{j})')
            features.append(np.sin(2*x))
            feature_names.append(f'sin(2x{j})')
            features.append(np.cos(2*x))
            feature_names.append(f'cos(2x{j})')
            
            # Exponential and log (with safety)
            exp_x = np.exp(np.clip(x, -50, 50))
            features.append(exp_x)
            feature_names.append(f'exp(x{j})')
            
            log_x = np.log(np.clip(x, 1e-10, None))
            features.append(log_x)
            feature_names.append(f'log(x{j})')
            
            # Hyperbolic
            features.append(np.tanh(x))
            feature_names.append(f'tanh(x{j})')
            features.append(np.sinh(np.clip(x, -50, 50)))
            feature_names.append(f'sinh(x{j})')
            
            # Square root
            features.append(np.sqrt(np.clip(x, 0, None)))
            feature_names.append(f'sqrt(x{j})')
            
            # Absolute value
            features.append(np.abs(x))
            feature_names.append(f'abs(x{j})')
            
            # Inverse
            inv_x = 1.0 / np.clip(x, 1e-10, None)
            features.append(inv_x)
            feature_names.append(f'1/x{j}')
            
            # Rational: x / (1 + x)
            features.append(x / (1.0 + x))
            feature_names.append(f'x{j}/(1+x{j})')
            
            # More rational functions
            features.append(x / (1.0 + x**2))
            feature_names.append(f'x{j}/(1+x{j}^2)')
            features.append(x / (1.0 + np.abs(x)))
            feature_names.append(f'x{j}/(1+|x{j}|)')
        
        # Bivariate interactions
        if d > 1:
            for i in range(d):
                for j in range(i+1, d):
                    xi = X[:, i]
                    xj = X[:, j]
                    
                    # Products
                    features.append(xi * xj)
                    feature_names.append(f'x{i}*x{j}')
                    
                    # Squared products
                    features.append((xi * xj)**2)
                    feature_names.append(f'(x{i}*x{j})^2')
                    
                    # Sums
                    s = xi + xj
                    features.append(np.sin(s))
                    feature_names.append(f'sin(x{i}+x{j})')
                    features.append(np.cos(s))
                    feature_names.append(f'cos(x{i}+x{j})')
                    features.append(np.tanh(s))
                    feature_names.append(f'tanh(x{i}+x{j})')
                    features.append(s**2)
                    feature_names.append(f'(x{i}+x{j})^2')
                    
                    # Differences
                    diff = xi - xj
                    features.append(np.sin(diff))
                    feature_names.append(f'sin(x{i}-x{j})')
                    features.append(np.cos(diff))
                    feature_names.append(f'cos(x{i}-x{j})')
                    features.append(diff**2)
                    feature_names.append(f'(x{i}-x{j})^2')
                    
                    # Ratios
                    ratio = xi / np.clip(xj, 1e-10, None)
                    features.append(ratio)
                    feature_names.append(f'x{i}/x{j}')
                    inv_ratio = xj / np.clip(xi, 1e-10, None)
                    features.append(inv_ratio)
                    feature_names.append(f'x{j}/x{i}')
                    
                    # Product with trig
                    features.append(xi * xj * np.sin(xi + xj))
                    feature_names.append(f'x{i}*x{j}*sin(x{i}+x{j})')
                    features.append(xi * xj * np.cos(xi + xj))
                    feature_names.append(f'x{i}*x{j}*cos(x{i}+x{j})')
                    
                    # Exp of product
                    features.append(np.exp(np.clip(xi * xj, -50, 50)))
                    feature_names.append(f'exp(x{i}*x{j})')
                    
                    # Log of product
                    features.append(np.log(np.clip(xi * xj, 1e-10, None)))
                    feature_names.append(f'log(x{i}*x{j})')
                    
                    # Sqrt of product
                    features.append(np.sqrt(np.clip(xi * xj, 0, None)))
                    feature_names.append(f'sqrt(x{i}*x{j})')
                    
                    # Max and min
                    features.append(np.maximum(xi, xj))
                    feature_names.append(f'max(x{i},x{j})')
                    features.append(np.minimum(xi, xj))
                    feature_names.append(f'min(x{i},x{j})')
                    
                    # Sum of squares
                    features.append(xi**2 + xj**2)
                    feature_names.append(f'x{i}^2+x{j}^2')
        
        # Trivariate interactions
        if d > 2:
            x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
            prod3 = x0 * x1 * x2
            features.append(prod3)
            feature_names.append('x0*x1*x2')
            
            s3 = x0 + x1 + x2
            features.append(np.sin(s3))
            feature_names.append('sin(sum3)')
            features.append(np.cos(s3))
            feature_names.append('cos(sum3)')
            features.append(s3**2)
            feature_names.append('sum3^2')
            
            # Pairwise products summed
            features.append(x0*x1 + x0*x2 + x1*x2)
            feature_names.append('sum_pair_prods')
            
            # Products of pairs
            features.append(x0*x1 * x2)
            feature_names.append('x0*x1*x2')
            
            # Ratios
            features.append(x0 / np.clip(x1, 1e-10, None))
            feature_names.append('x0/x1')
            features.append(x0 / np.clip(x2, 1e-10, None))
            feature_names.append('x0/x2')
            features.append(x1 / np.clip(x2, 1e-10, None))
            feature_names.append('x1/x2')
            
            # Sqrt of products
            features.append(np.sqrt(np.clip(x0*x1, 0, None)))
            feature_names.append('sqrt(x0*x1)')
            features.append(np.sqrt(np.clip(x0*x2, 0, None)))
            feature_names.append('sqrt(x0*x2)')
            features.append(np.sqrt(np.clip(x1*x2, 0, None)))
            feature_names.append('sqrt(x1*x2)')
        
        # Stack and clean
        B = np.column_stack(features)
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove constant columns (all same value)
        col_vars = np.var(B, axis=0)
        valid = col_vars > 1e-12
        B = B[:, valid]
        
        return B, feature_names
    
    B, feature_names = _build_basis(X, d)
    num_features = B.shape[1]
    
    # Fit with ridge regression
    def _fit_ridge(B, y, lam=1e-4):
        n, p = B.shape
        A = B.T @ B + lam * np.eye(p)
        b = B.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w
    
    # Fit with Lasso-like approach using coordinate descent
    def _fit_lasso(B, y, alpha=0.01, max_iter=1000):
        n, p = B.shape
        w = np.zeros(p)
        B_norm = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-8)
        
        for _ in range(max_iter):
            for j in range(p):
                w_old = w[j]
                rho = B_norm[:, j].T @ (y - B_norm @ w + w_old * B_norm[:, j])
                if rho > alpha:
                    w[j] = (rho - alpha) / (B_norm[:, j].T @ B_norm[:, j])
                elif rho < -alpha:
                    w[j] = (rho + alpha) / (B_norm[:, j].T @ B_norm[:, j])
                else:
                    w[j] = 0.0
        
        return w
    
    # Compute MSE
    def _compute_mse(pred, y):
        return np.mean((pred - y)**2)
    
    # Forward feature selection with better efficiency
    def _select_features(B, y, max_features=20, time_budget=300):
        n, p = B.shape
        selected = [0]  # Start with constant
        remaining = list(range(1, p))
        
        # Initial fit with just constant
        w = _fit_ridge(B[:, [0]], y)
        pred = B[:, [0]] @ w
        best_mse = _compute_mse(pred, y)
        
        # Pre-compute some statistics for faster evaluation
        B_centered = B - B.mean(axis=0)
        
        step = 0
        while remaining and step < max_features and _time_left() > 15:
            best_idx = -1
            best_new_mse = best_mse
            
            # Try each remaining feature
            for idx in remaining:
                new_sel = selected + [idx]
                B_sel = B[:, new_sel]
                try:
                    w = _fit_ridge(B_sel, y)
                    pred = B_sel @ w
                    mse = _compute_mse(pred, y)
                    if mse < best_new_mse:
                        best_new_mse = mse
                        best_idx = idx
                except:
                    continue
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            remaining.remove(best_idx)
            best_mse = best_new_mse
            step += 1
            
            # Early stopping if MSE is very small
            if best_mse < 1e-10:
                break
        
        return selected, best_mse
    
    # Try neural network fit with better architecture
    def _try_nn_fit(X, y, time_budget=45):
        if _time_left() < time_budget:
            return None
        
        # Simple 2-layer neural net using scipy minimize
        n, d = X.shape
        
        # Normalize inputs and output
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-8
        X_norm = (X - x_mean) / x_std
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Initialize weights
        np.random.seed(42)
        hidden_size = max(15, min(60, n // 4))
        W1 = np.random.randn(d, hidden_size) * np.sqrt(2.0 / d)
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros(1)
        
        params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
        
        def loss_fn(params):
            if _time_left() < 5:
                return 1e10
            W1 = params[:d*hidden_size].reshape(d, hidden_size)
            b1 = params[d*hidden_size:d*hidden_size+hidden_size]
            W2 = params[d*hidden_size+hidden_size:-1].reshape(hidden_size, 1)
            b2 = params[-1:]
            
            h = np.tanh(X_norm @ W1 + b1)
            pred = h @ W2 + b2
            pred = pred.flatten()
            mse = np.mean((pred - y_norm)**2)
            # Add L2 regularization
            reg = 1e-4 * (np.sum(W1**2) + np.sum(W2**2))
            return mse + reg
        
        try:
            result = minimize(loss_fn, params, method='L-BFGS-B', 
                             options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-7})
            
            if result.success or result.fun < loss_fn(params):
                W1 = result.x[:d*hidden_size].reshape(d, hidden_size)
                b1 = result.x[d*hidden_size:d*hidden_size+hidden_size]
                W2 = result.x[d*hidden_size+hidden_size:-1].reshape(hidden_size, 1)
                b2 = result.x[-1:]
                
                # Test MSE on training data
                h = np.tanh(X_norm @ W1 + b1)
                pred = (h @ W2 + b2).flatten()
                pred_orig = pred * y_std + y_mean
                mse = np.mean((pred_orig - y)**2)
                
                return {
                    'type': 'nn',
                    'mse': mse,
                    'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
                    'x_mean': x_mean, 'x_std': x_std,
                    'y_mean': y_mean, 'y_std': y_std
                }
        except:
            pass
        
        return None
    
    # Run feature selection
    selected_features, best_mse = _select_features(B, y, max_features=25)
    
    # Final ridge fit with selected features
    B_sel = B[:, selected_features]
    w = _fit_ridge(B_sel, y)
    
    # Try NN and see if it's better
    nn_result = _try_nn_fit(X, y)
    
    use_nn = (nn_result is not None and nn_result['mse'] < best_mse * 0.3)
    
    if use_nn:
        W1, b1, W2, b2 = nn_result['W1'], nn_result['b1'], nn_result['W2'], nn_result['b2']
        x_mean, x_std = nn_result['x_mean'], nn_result['x_std']
        y_mean, y_std = nn_result['y_mean'], nn_result['y_std']
        
        def predict(X_test):
            X_test = np.array(X_test, dtype=float)
            X_norm = (X_test - x_mean) / x_std
            h = np.tanh(X_norm @ W1 + b1)
            pred = (h @ W2 + b2).flatten()
            pred = pred * y_std + y_mean
            return pred
    else:
        def predict(X_test):
            X_test = np.array(X_test, dtype=float)
            B_test, _ = _build_basis(X_test, d)
            B_test_sel = B_test[:, selected_features]
            return B_test_sel @ w
    
    return predict
