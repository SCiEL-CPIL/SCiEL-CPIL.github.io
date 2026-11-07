"""
Strategy: Ensemble Symbolic Regression with Model Selection.
Instead of assuming a single polynomial form, we generate multiple feature 
bases (Polynomial, Fourier, Radial Basis Functions) and fit a linear model 
on each. We use 5-fold cross-validation on the training set to select the 
best performing basis for each equation. This allows the algorithm to 
adapt to different functional forms (periodic, smooth local, global polynomial) 
without knowing them in advance.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _features_polynomial(X, degree=4):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for idx in itertools.combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for j in idx:
                feat = feat * X[:, j]
            cols.append(feat)
    return np.column_stack(cols)

def _features_fourier(X, n_terms=10):
    """Generate Fourier features (sin/cos) for 1D inputs."""
    n, d = X.shape
    cols = [np.ones(n)]
    # Assume 1D input for Fourier as per task description (1 input)
    # If d > 1, we can still apply to first column or all, but task says 1 input.
    for term in range(1, n_terms + 1):
        cols.append(np.sin(term * X[:, 0]))
        cols.append(np.cos(term * X[:, 0]))
    return np.column_stack(cols)

def _features_rbf(X, centers, sigmas):
    """Generate RBF features."""
    n, d = X.shape
    cols = [np.ones(n)]
    for c, s in zip(centers, sigmas):
        dist = np.sum((X - c) ** 2, axis=1)
        cols.append(np.exp(-dist / (2 * s ** 2)))
    return np.column_stack(cols)

def _fit_linear(X, y, ridge_alpha=1e-6):
    """Fit linear model with slight ridge regularization for stability."""
    # Add identity for regularization
    try:
        # Using lstsq is fast and robust for small matrices
        # Ridge regression via normal equations or lstsq with augmented matrix
        # For small d, normal equations with ridge is very fast
        A = X.T @ X
        b = X.T @ y
        A += ridge_alpha * np.eye(X.shape[1])
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w

def _predict_linear(X, w):
    return X @ w

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    n, d = X_train.shape
    
    # List of (name, feature_generator)
    # We need to generate features for CV and final fit
    # To save time, we pre-generate features for all candidates if possible, 
    # but RBF needs centers. Let's generate on the fly or pre-compute static ones.
    
    candidates = []
    
    # 1. Polynomial degrees 1 to 5
    for deg in range(1, 6):
        try:
            if time.time() > deadline: break
            feats = _features_polynomial(X_train, degree=deg)
            candidates.append(('poly', deg, feats))
        except Exception:
            pass
            
    # 2. Fourier terms 5, 10, 20
    for terms in [5, 10, 20]:
        try:
            if time.time() > deadline: break
            feats = _features_fourier(X_train, n_terms=terms)
            candidates.append(('fourier', terms, feats))
        except Exception:
            pass
            
    # 3. RBF with fixed centers
    # Sample 10 centers from training data
    # Use different sigmas
    centers = X_train[np.random.choice(n, 10, replace=False)]
    for sigma in [0.5, 1.0, 2.0]:
        try:
            if time.time() > deadline: break
            feats = _features_rbf(X_train, centers, [sigma]*10)
            candidates.append(('rbf', sigma, feats))
        except Exception:
            pass

    if not candidates:
        # Fallback: constant predictor
        mean_y = np.mean(y_train)
        return lambda X: np.full(X.shape[0], mean_y)

    # Cross-Validation to select best model
    # 5-Fold CV
    n_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
    
    best_score = np.inf
    best_model = None
    best_name = None
    
    # Evaluate each candidate
    for name, param, feats_train in candidates:
        if time.time() > deadline:
            break
            
        cv_errors = []
        valid_fold = True
        
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_val = feats_train[val_idx]
            y_val = y_train[val_idx]
            X_tr = feats_train[train_idx]
            y_tr = y_train[train_idx]
            
            # Fit model on train fold
            w = _fit_linear(X_tr, y_tr)
            
            # Predict on val fold
            pred = _predict_linear(X_val, w)
            
            # Calculate MSE
            if not np.all(np.isfinite(pred)):
                valid_fold = False
                break
            mse = np.mean((pred - y_val) ** 2)
            cv_errors.append(mse)
            
        if valid_fold:
            mean_mse = np.mean(cv_errors)
            if mean_mse < best_score:
                best_score = mean_mse
                best_name = (name, param)
                # Refit on full training data for the final model
                # We reuse the feats_train generated earlier
                best_model = _fit_linear(feats_train, y_train)
                # Store the feature generator parameters to regenerate features for prediction
                # Note: We need to be able to generate features for X_test using the same params
                # We store the type and params
                best_feat_type = name
                best_feat_param = param
                # For RBF, we need the centers too
                best_centers = centers if name == 'rbf' else None

    if best_model is None:
        # Fallback to simple linear or constant if CV failed
        # Try simple linear regression on raw X
        try:
            w = _fit_linear(X_train, y_train)
            return lambda X: X @ w
        except:
            mean_y = np.mean(y_train)
            return lambda X: np.full(X.shape[0], mean_y)

    # Construct prediction function
    def predict(X):
        # Regenerate features using the best selected type and params
        if best_feat_type == 'poly':
            feats = _features_polynomial(X, degree=best_feat_param)
        elif best_feat_type == 'fourier':
            feats = _features_fourier(X, n_terms=best_feat_param)
        elif best_feat_type == 'rbf':
            feats = _features_rbf(X, best_centers, [best_feat_param]*10)
        else:
            feats = X # Should not happen
            
        pred = feats @ best_model
        return pred

    return predict
# EVOLVE-BLOCK-END
