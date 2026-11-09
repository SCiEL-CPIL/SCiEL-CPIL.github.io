"""Search-based symbolic regression with time-bounded candidate evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Normalize inputs for numerical stability in some features
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X_train - X_min) / X_range  # in [0,1]
    
    def norm_X(X):
        return (X - X_min) / X_range
    
    # Helper to compute normalized MSE
    def nrmse(pred):
        return np.mean((pred - y_train)**2) / y_var
    
    best_score = nrmse(np.full(n, y_mean))
    best_predictor = lambda X: np.full(X.shape[0], y_mean)
    
    # Candidate 1: Linear
    try:
        A = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        pred = A @ w
        score = nrmse(pred)
        if score < best_score:
            best_score = score
            best_predictor = lambda X, w=w: np.column_stack([np.ones(X.shape[0]), X]) @ w
    except:
        pass
    
    # Candidate 2: Quadratic polynomial
    try:
        def make_quad_features(X):
            cols = [np.ones(X.shape[0])]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        F = make_quad_features(X_train)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        pred = F @ w
        score = nrmse(pred)
        if score < best_score:
            best_score = score
            best_predictor = lambda X, w=w: make_quad_features(X) @ w
    except:
        pass
    
    # Candidate 3: Cubic polynomial (only if d is small)
    if d <= 3:
        try:
            def make_cubic_features(X):
                cols = [np.ones(X.shape[0])]
                for j in range(X.shape[1]):
                    cols.append(X[:, j])
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        cols.append(X[:, i] * X[:, j])
                for i in range(X.shape[1]):
                    for j in range(X.shape[1]):
                        for k in range(X.shape[1]):
                            if i <= j <= k:
                                cols.append(X[:, i] * X[:, j] * X[:, k])
                return np.column_stack(cols)
            
            F = make_cubic_features(X_train)
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            score = nrmse(pred)
            if score < best_score:
                best_score = score
                best_predictor = lambda X, w=w: make_cubic_features(X) @ w
        except:
            pass
    
    # Candidate 4: Sine/cosine features
    try:
        def make_trig_features(X):
            cols = [np.ones(X.shape[0])]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
                cols.append(np.sin(X[:, j]))
                cols.append(np.cos(X[:, j]))
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
                    cols.append(np.sin(X[:, i] * X[:, j]))
                    cols.append(np.cos(X[:, i] * X[:, j]))
            return np.column_stack(cols)
        
        F = make_trig_features(X_train)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        pred = F @ w
        score = nrmse(pred)
        if score < best_score:
            best_score = score
            best_predictor = lambda X, w=w: make_trig_features(X) @ w
    except:
        pass
    
    # Candidate 5: Exp/log features
    try:
        def make_explog_features(X):
            cols = [np.ones(X.shape[0])]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
                cols.append(np.exp(X[:, j]))
                cols.append(np.log(X[:, j] + 1))
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        F = make_explog_features(X_train)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        pred = F @ w
        score = nrmse(pred)
        if score < best_score:
            best_score = score
            best_predictor = lambda X, w=w: make_explog_features(X) @ w
    except:
        pass
    
    # Candidate 6: Gaussian RBF with a few centers
    try:
        # Choose a few centers from training data
        idx = np.linspace(0, n-1, min(20, n)).astype(int)
        centers = X_train[idx]
        # Try a few bandwidths
        best_rbf_score = best_score
        best_rbf_predictor = best_predictor
        for bw in [0.1, 0.3, 0.5, 1.0, 2.0]:
            def make_rbf(X, bw=bw, centers=centers):
                # Compute distances
                X_exp = X[:, np.newaxis, :]
                C_exp = centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum((X_exp - C_exp)**2, axis=2))
                return np.exp(-dists**2 / (2 * bw**2))
            
            F = make_rbf(X_train)
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            score = nrmse(pred)
            if score < best_rbf_score:
                best_rbf_score = score
                best_rbf_predictor = lambda X, w=w, bw=bw, centers=centers: make_rbf(X) @ w
        if best_rbf_score < best_score:
            best_score = best_rbf_score
            best_predictor = best_rbf_predictor
    except:
        pass
    
    # Candidate 7: Interaction + nonlinear single-variable terms
    try:
        def make_rich_features(X):
            cols = [np.ones(X.shape[0])]
            for j in range(X.shape[1]):
                x = X[:, j]
                cols.append(x)
                cols.append(x**2)
                cols.append(np.sin(x))
                cols.append(np.cos(x))
                cols.append(np.exp(x))
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
                    cols.append(np.sin(X[:, i] * X[:, j]))
            return np.column_stack(cols)
        
        F = make_rich_features(X_train)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        pred = F @ w
        score = nrmse(pred)
        if score < best_score:
            best_score = score
            best_predictor = lambda X, w=w: make_rich_features(X) @ w
    except:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best_predictor
    
    # Candidate 8: Higher-degree polynomial if d is small
    if d == 1:
        try:
            def make_high_poly(X, deg=6):
                cols = [np.ones(X.shape[0])]
                for p in range(1, deg+1):
                    cols.append(X[:, 0]**p)
                return np.column_stack(cols)
            
            F = make_high_poly(X_train)
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            pred = F @ w
            score = nrmse(pred)
            if score < best_score:
                best_score = score
                best_predictor = lambda X, w=w: make_high_poly(X) @ w
        except:
            pass
    
    return best_predictor
# EVOLVE-BLOCK-END
