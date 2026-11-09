"""Symbolic regression with adaptive feature search and time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Normalize inputs for better numerical behavior
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-12] = 1.0
    
    def normalize(X):
        return (X - X_mean) / X_std
    
    X_norm_train = normalize(X_train)
    
    # Feature generation functions
    def poly_features(X, degree):
        """Generate polynomial features up to given degree."""
        cols = [np.ones(len(X))]
        for deg in range(1, degree + 1):
            for combo in itertools.combinations_with_replacement(range(d), deg):
                col = np.ones(len(X))
                for idx in combo:
                    col = col * X[:, idx]
                cols.append(col)
        return np.column_stack(cols)
    
    def trig_features(X):
        """Generate trigonometric features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
        # Cross terms
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
        return np.column_stack(cols)
    
    def exp_features(X):
        """Generate exponential features (clipped for stability)."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            cols.append(np.exp(0.5 * Xj))
            cols.append(np.exp(-0.5 * Xj))
            cols.append(np.exp(0.25 * Xj))
            cols.append(np.exp(-0.25 * Xj))
        return np.column_stack(cols)
    
    def log_features(X):
        """Generate log features (only for positive values)."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            min_val = np.min(Xj)
            if min_val > 0:
                cols.append(np.log(Xj))
                cols.append(np.log(Xj ** 2))
            else:
                shift = -min_val + 1.0
                cols.append(np.log(Xj + shift))
        return np.column_stack(cols)
    
    def sqrt_features(X):
        """Generate square root features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            min_val = np.min(Xj)
            if min_val >= 0:
                cols.append(np.sqrt(Xj))
                cols.append(np.sqrt(Xj ** 3))
            else:
                shift = -min_val + 1.0
                cols.append(np.sqrt(Xj + shift))
        return np.column_stack(cols)
    
    def interaction_features(X):
        """Generate pairwise interaction features."""
        cols = [np.ones(len(X))]
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                cols.append(np.cos(X[:, i]) * np.cos(X[:, j]))
        return np.column_stack(cols)
    
    # Curated feature sets to avoid overfitting and excessive computation
    feature_generators = [
        ("poly2", lambda X: poly_features(X, 2)),
        ("poly3", lambda X: poly_features(X, 3)),
        ("trig", trig_features),
        ("poly2+trig", lambda X: np.column_stack([poly_features(X, 2)[:, 1:], trig_features(X)[:, 1:]])),
        ("poly2+exp", lambda X: np.column_stack([poly_features(X, 2)[:, 1:], exp_features(X)[:, 1:]])),
        ("poly2+log", lambda X: np.column_stack([poly_features(X, 2)[:, 1:], log_features(X)[:, 1:]])),
        ("poly2+sqrt", lambda X: np.column_stack([poly_features(X, 2)[:, 1:], sqrt_features(X)[:, 1:]])),
        ("poly2+interaction", lambda X: np.column_stack([poly_features(X, 2)[:, 1:], interaction_features(X)[:, 1:]])),
    ]
    
    # K-Fold Cross-Validation to estimate generalization error
    k_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    def cv_ridge_score(X_feat, y, alpha):
        """Compute mean CV MSE for a given feature matrix and alpha."""
        scores = []
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            X_tr = X_feat[train_idx]
            y_tr = y[train_idx]
            X_val = X_feat[val_idx]
            y_val = y[val_idx]
            
            n_features = X_feat.shape[1]
            A = X_tr.T @ X_tr + alpha * np.eye(n_features)
            b = X_tr.T @ y_tr
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
            
            y_pred = X_val @ w
            mse = np.mean((y_pred - y_val) ** 2)
            scores.append(mse)
        
        return np.mean(scores)
    
    best_model_w = None
    best_cv_score = np.inf
    best_gen_func = None
    best_alpha = None
    
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    for name, gen_func in feature_generators:
        if time.time() > deadline:
            break
        
        try:
            X_feat_train = gen_func(X_norm_train)
            
            if not np.all(np.isfinite(X_feat_train)):
                continue
            
            for alpha in alphas:
                if time.time() > deadline:
                    break
                
                cv_score = cv_ridge_score(X_feat_train, y_train, alpha)
                
                if cv_score < best_cv_score:
                    best_cv_score = cv_score
                    best_gen_func = gen_func
                    best_alpha = alpha
                    
        except Exception:
            continue
    
    # Fit final model on all training data with best parameters
    if best_gen_func is not None:
        X_feat_train = best_gen_func(X_norm_train)
        n_features = X_feat_train.shape[1]
        A = X_feat_train.T @ X_feat_train + best_alpha * np.eye(n_features)
        b = X_feat_train.T @ y_train
        
        try:
            w_final = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w_final = np.linalg.lstsq(X_feat_train, y_train, rcond=None)[0]
        
        def predict(X):
            X_norm = normalize(X)
            X_feat = best_gen_func(X_norm)
            return X_feat @ w_final
        
        return predict
    
    # Fallback: simple linear regression on normalized features
    X_lin_train = np.column_stack([np.ones(n), X_norm_train])
    w_lin, *_ = np.linalg.lstsq(X_lin_train, y_train, rcond=None)
    
    def predict_fallback(X):
        X_norm = normalize(X)
        X_lin = np.column_stack([np.ones(len(X)), X_norm])
        return X_lin @ w_lin
    
    return predict_fallback
# EVOLVE-BLOCK-END
