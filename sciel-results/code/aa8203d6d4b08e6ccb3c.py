"""
Structural Change: Multi-kernel Gaussian Process Regression with Bayesian Hyperparameter Search.
Instead of a single RBF kernel with fixed bandwidth selection via validation split, this strategy:
1. Uses a composite kernel: RBF + Polynomial + Sigmoid (tanh) to capture diverse functional forms.
2. Employs a multi-start grid search over hyperparameters (length scales, amplitudes, polynomial degree).
3. Uses 5-fold cross-validation for robust model selection instead of a single train/val split.
4. Regularizes with a small noise term for numerical stability.
5. Falls back to polynomial regression if GP fitting fails.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    """
    Fits a multi-kernel Gaussian Process regression model with Bayesian hyperparameter search.
    Returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # Helper to compute squared distances
    def compute_sq_dists(A, B):
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        B_sq = np.sum(B**2, axis=1, keepdims=True)
        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
        return np.maximum(sq_dists, 0)
    
    # Composite kernel: RBF + Polynomial + Sigmoid
    def composite_kernel(X1, X2, hyperparams):
        """
        hyperparams: dict with keys:
        - rbf_length_scale: float
        - rbf_amplitude: float
        - poly_degree: int
        - poly_amplitude: float
        - sigmoid_length_scale: float
        - sigmoid_amplitude: float
        """
        sq_dists = compute_sq_dists(X1, X2)
        
        # RBF component
        rbf_ls = hyperparams['rbf_length_scale']
        rbf_amp = hyperparams['rbf_amplitude']
        K_rbf = rbf_amp * np.exp(-0.5 * sq_dists / (rbf_ls ** 2))
        
        # Polynomial component
        poly_deg = int(hyperparams['poly_degree'])
        poly_amp = hyperparams['poly_amplitude']
        # Compute polynomial kernel: (X1^T X2 / d + 1)^degree
        # Normalize by feature count to keep values reasonable
        dot_prod = (X1 @ X2.T) / n_features
        K_poly = poly_amp * (dot_prod + 1.0) ** poly_deg
        
        # Sigmoid component
        sig_ls = hyperparams['sigmoid_length_scale']
        sig_amp = hyperparams['sigmoid_amplitude']
        K_sig = sig_amp * np.tanh(5.0 * (dot_prod / sig_ls + 0.5))
        
        return K_rbf + K_poly + K_sig
    
    # 5-fold cross-validation for hyperparameter selection
    def cv_score(hyperparams):
        """Compute mean validation MSE over 5 folds."""
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // 5
        total_mse = 0.0
        n_folds = 0
        
        for fold in range(5):
            if time.time() > deadline - 30:
                break
                
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < 4 else n_samples
            
            val_idx = indices[val_start:val_end]
            tr_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_tr = X_norm[tr_idx]
            y_tr = y[tr_idx]
            X_val = X_norm[val_idx]
            y_val = y[val_idx]
            
            try:
                K_tr = composite_kernel(X_tr, X_tr, hyperparams)
                K_tr += 1e-6 * np.eye(len(y_tr))
                
                alpha = np.linalg.solve(K_tr, y_tr)
                
                K_val_tr = composite_kernel(X_val, X_tr, hyperparams)
                y_val_pred = K_val_tr @ alpha
                
                mse = np.mean((y_val_pred - y_val) ** 2)
                total_mse += mse
                n_folds += 1
            except np.linalg.LinAlgError:
                return float('inf')
        
        if n_folds == 0:
            return float('inf')
        
        return total_mse / n_folds
    
    # Define hyperparameter grid
    rbf_lengths = [0.1, 0.3, 0.5, 1.0, 2.0]
    rbf_amplitudes = [0.5, 1.0, 2.0]
    poly_degrees = [1, 2, 3]
    poly_amplitudes = [0.1, 0.5, 1.0]
    sigmoid_lengths = [0.5, 1.0, 2.0]
    sigmoid_amplitudes = [0.1, 0.5, 1.0]
    
    # Multi-start search: try a subset of the full grid to stay within time budget
    # Full grid would be 5*3*3*3*3*3 = 1215 combos, too many for 5-fold CV
    # Instead, use a coarser grid and then refine
    
    best_score = float('inf')
    best_hyperparams = None
    
    # Coarse grid search
    for rbf_ls in rbf_lengths:
        for poly_deg in poly_degrees:
            for sig_ls in sigmoid_lengths:
                if time.time() > deadline - 60:
                    break
                
                # Use fixed amplitudes for coarse search
                hyperparams = {
                    'rbf_length_scale': rbf_ls,
                    'rbf_amplitude': 1.0,
                    'poly_degree': poly_deg,
                    'poly_amplitude': 0.5,
                    'sigmoid_length_scale': sig_ls,
                    'sigmoid_amplitude': 0.5
                }
                
                score = cv_score(hyperparams)
                
                if score < best_score:
                    best_score = score
                    best_hyperparams = hyperparams.copy()
    
    # If coarse search didn't find anything good, try a few more
    if best_hyperparams is None:
        best_hyperparams = {
            'rbf_length_scale': 1.0,
            'rbf_amplitude': 1.0,
            'poly_degree': 2,
            'poly_amplitude': 0.5,
            'sigmoid_length_scale': 1.0,
            'sigmoid_amplitude': 0.5
        }
        best_score = cv_score(best_hyperparams)
    
    # Fine-tune amplitudes around the best found configuration
    if time.time() < deadline - 30 and best_hyperparams is not None:
        for rbf_amp in rbf_amplitudes:
            for poly_amp in poly_amplitudes:
                for sig_amp in sigmoid_amplitudes:
                    if time.time() > deadline - 30:
                        break
                    
                    hyperparams = best_hyperparams.copy()
                    hyperparams['rbf_amplitude'] = rbf_amp
                    hyperparams['poly_amplitude'] = poly_amp
                    hyperparams['sigmoid_amplitude'] = sig_amp
                    
                    score = cv_score(hyperparams)
                    
                    if score < best_score:
                        best_score = score
                        best_hyperparams = hyperparams.copy()
    
    # Final fit on all data with best hyperparameters
    try:
        K_full = composite_kernel(X_norm, X_norm, best_hyperparams)
        K_full += 1e-6 * np.eye(n_samples)
        
        alpha_full = np.linalg.solve(K_full, y)
        
        # Capture variables for closure
        final_alpha = alpha_full
        final_hyperparams = best_hyperparams.copy()
        min_vals_closure = min_vals
        range_vals_closure = range_vals
        X_train_norm = X_norm
        n_features_closure = n_features
        
        def predict(X):
            X = np.asarray(X, dtype=float)
            X_norm_test = (X - min_vals_closure) / range_vals_closure
            
            K_test = composite_kernel(X_norm_test, X_train_norm, final_hyperparams)
            
            return K_test @ final_alpha
        
        return predict
        
    except np.linalg.LinAlgError:
        # Fallback to polynomial regression
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

# EVOLVE-BLOCK-END
