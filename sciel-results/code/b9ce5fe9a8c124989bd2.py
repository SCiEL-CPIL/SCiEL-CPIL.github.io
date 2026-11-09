import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def build_features(X):
        n = X.shape[0]
        cols = [np.ones(n)]
        for j in range(d):
            x = X[:, j]
            # Polynomial terms up to degree 7
            xk = np.ones(n)
            for k in range(1, 8):
                xk = xk * x
                cols.append(xk.copy())
            # Trig terms
            for freq in [0.5, 1.0, 1.5, 2.0, 3.0, np.pi/2, np.pi]:
                cols.append(np.sin(freq * x))
                cols.append(np.cos(freq * x))
            # Exp terms
            for c in [0.5, 1.0, -0.5, -1.0]:
                cols.append(np.exp(c * x))
            # Abs and sqrt
            cols.append(np.abs(x))
            cols.append(np.sqrt(np.abs(x)))
            # Log terms (safe)
            cols.append(np.log(np.abs(x) + 1e-6))
            cols.append(np.log(np.abs(x) + 1.0))
            # Rational
            cols.append(1.0 / (x + 1e-6))
            cols.append(1.0 / (x**2 + 1e-6))
            cols.append(1.0 / (np.abs(x) + 1.0))
        return np.column_stack(cols)
    
    def ridge_fit(Xf, y, alpha):
        n, p = Xf.shape
        # Solve (Xf^T Xf + alpha*I) w = Xf^T y
        A = Xf.T @ Xf + alpha * np.eye(p)
        b = Xf.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w
    
    def predict_with_w(Xf, w):
        return Xf @ w
    
    # Build features
    Xf_train = build_features(X)
    n_feat = Xf_train.shape[1]
    
    # Cross-validate alpha
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha = 1.0
    best_cv_score = np.inf
    
    # Simple 5-fold CV
    indices = np.arange(n)
    fold_size = n // 5
    for alpha in alphas:
        cv_scores = []
        for fold in range(5):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])
            
            Xf_tr = Xf_train[train_idx]
            y_tr = y[train_idx]
            Xf_val = Xf_train[val_idx]
            y_val = y[val_idx]
            
            w = ridge_fit(Xf_tr, y_tr, alpha)
            pred = predict_with_w(Xf_val, w)
            mse = np.mean((pred - y_val)**2)
            cv_scores.append(mse)
        
        mean_cv = np.mean(cv_scores)
        if mean_cv < best_cv_score:
            best_cv_score = mean_cv
            best_alpha = alpha
    
    # Final fit
    w = ridge_fit(Xf_train, y, best_alpha)
    
    def predict(X):
        Xf = build_features(np.asarray(X, dtype=np.float64))
        return Xf @ w
    
    return predict
