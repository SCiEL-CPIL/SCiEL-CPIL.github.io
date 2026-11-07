"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize inputs to [0, 1] for better numerical conditioning
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range
    
    Xn_train = normalize(X_train)
    
    # Build a rich feature set:
    # 1. Constants
    # 2. Linear terms
    # 3. Quadratic terms (including cross terms)
    # 4. Cubic terms (pure powers only, to limit dimensionality)
    # 5. Sin/cos of linear terms (periodic functions)
    # 6. Exp of linear terms (exponential growth/decay)
    # 7. Reciprocal terms (1/x, 1/x^2)
    
    def build_features(Xn):
        cols = [np.ones(len(Xn))]
        
        # Linear
        for j in range(d):
            cols.append(Xn[:, j])
        
        # Quadratic (including cross)
        for i in range(d):
            for j in range(i, d):
                cols.append(Xn[:, i] * Xn[:, j])
        
        # Cubic (pure powers only)
        for i in range(d):
            cols.append(Xn[:, i] ** 3)
        
        # Sin/Cos of linear combinations
        for j in range(d):
            cols.append(np.sin(np.pi * Xn[:, j]))
            cols.append(np.cos(np.pi * Xn[:, j]))
            cols.append(np.sin(2 * np.pi * Xn[:, j]))
            cols.append(np.cos(2 * np.pi * Xn[:, j]))
        
        # Exp terms
        for j in range(d):
            cols.append(np.exp(2 * Xn[:, j] - 1))
            cols.append(np.exp(-(2 * Xn[:, j] - 1)))
        
        # Reciprocal terms (with offset to avoid division by zero)
        for j in range(d):
            cols.append(1.0 / (Xn[:, j] + 0.1))
            cols.append(1.0 / (Xn[:, j] + 0.1) ** 2)
        
        # Cross sin/cos for 2D+ cases
        if d >= 2:
            for i in range(d):
                for j in range(i+1, d):
                    cols.append(np.sin(np.pi * (Xn[:, i] + Xn[:, j])))
                    cols.append(np.cos(np.pi * (Xn[:, i] + Xn[:, j])))
        
        return np.column_stack(cols)
    
    F_train = build_features(Xn_train)
    
    # Use ridge regression for stability
    # Try multiple lambda values and pick the best via cross-validation
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    
    # Simple 5-fold cross-validation to pick lambda
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_lambda = 1e-4
    best_cv_mse = np.inf
    
    for lam in lambdas:
        cv_mses = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            F_tr = F_train[train_idx]
            y_tr = y_train[train_idx]
            F_val = F_train[val_idx]
            y_val = y_train[val_idx]
            
            # Ridge regression
            A = F_tr.T @ F_tr + lam * np.eye(F_train.shape[1])
            b = F_tr.T @ y_tr
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(F_tr, y_tr, rcond=None)[0]
            
            pred = F_val @ w
            mse = np.mean((pred - y_val) ** 2)
            cv_mses.append(mse)
        
        mean_cv_mse = np.mean(cv_mses)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_lambda = lam
    
    # Final fit with best lambda
    A = F_train.T @ F_train + best_lambda * np.eye(F_train.shape[1])
    b = F_train.T @ y_train
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F_train, y_train, rcond=None)[0]
    
    def predict(X):
        Xn = normalize(X)
        F = build_features(Xn)
        return F @ w
    
    return predict
# EVOLVE-BLOCK-END
