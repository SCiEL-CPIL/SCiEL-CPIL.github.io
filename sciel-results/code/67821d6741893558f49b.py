"""Symbolic regression over 10 hidden equations with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
import operator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Check for degenerate cases
    if np.std(y) < 1e-12:
        return lambda X: np.zeros(len(X))
    
    # Normalize inputs to [0, 1] for numerical stability
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    y_norm = (y - y_mean) / y_std
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return max(0, deadline - time.time())
    
    # Helper: fit linear model on features, return (mse, predict_fn)
    def fit_linear(F_train, y_target, F_predict=None):
        try:
            # Add ridge regularization for stability
            w, *_ = np.linalg.lstsq(F_train, y_target, rcond=None)
            def pred(X):
                if F_predict is None:
                    F = F_train
                else:
                    F = F_predict(X)
                return F @ w
            pred_train = pred(X)
            mse = np.mean((pred_train - y_target) ** 2)
            return mse, pred
        except Exception:
            return np.inf, None
    
    # Feature builders that take normalized X and return feature matrix
    def build_polynomial(Xn, degree):
        cols = [np.ones(len(Xn))]
        for d in range(1, degree + 1):
            for combo in product(range(n_features), repeat=d):
                combo = tuple(sorted(combo))
                # Only include unique monomials
                if len(set(combo)) != len(combo) and len(combo) > 1:
                    # Allow repeated indices for powers but ensure sorted
                    pass
                feat = np.ones(len(Xn))
                for idx in combo:
                    feat = feat * Xn[:, idx]
                cols.append(feat)
        return np.column_stack(cols)
    
    def build_trig(Xn):
        cols = [np.ones(len(Xn))]
        for j in range(n_features):
            cols.append(np.sin(np.pi * Xn[:, j]))
            cols.append(np.sin(2 * np.pi * Xn[:, j]))
            cols.append(np.cos(np.pi * Xn[:, j]))
            cols.append(np.cos(2 * np.pi * Xn[:, j]))
        # Cross terms for 2 features
        if n_features >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    cols.append(np.sin(np.pi * Xn[:, i]) * np.cos(np.pi * Xn[:, j]))
                    cols.append(np.cos(np.pi * Xn[:, i]) * np.sin(np.pi * Xn[:, j]))
        return np.column_stack(cols)
    
    def build_exp_log(Xn):
        cols = [np.ones(len(Xn))]
        for j in range(n_features):
            cols.append(np.exp(Xn[:, j] - 1))  # exp(x-1) to avoid overflow
            cols.append(np.exp(-(Xn[:, j] - 1)))
            cols.append(np.log1p(Xn[:, j]))
            cols.append(np.log1p(1 - Xn[:, j]) if Xn[:, j].max() < 1.0 else np.ones(len(Xn)))
        return np.column_stack(cols)
    
    def build_rational(Xn):
        cols = [np.ones(len(Xn))]
        for j in range(n_features):
            # x/(1+x), x^2/(1+x^2) style
            denom = 1 + Xn[:, j]
            cols.append(Xn[:, j] / denom)
            denom2 = 1 + Xn[:, j] ** 2
            cols.append(Xn[:, j] / denom2)
            cols.append(Xn[:, j] ** 2 / denom2)
        return np.column_stack(cols)
    
    def build_spline(Xn, n_knots=5):
        cols = [np.ones(len(Xn))]
        for j in range(n_features):
            x = Xn[:, j]
            for k in range(1, n_knots):
                knot = k / n_knots
                cols.append(np.maximum(0, x - knot) ** 2)
                cols.append(np.maximum(0, knot - x) ** 2)
        return np.column_stack(cols)
    
    def build_interactions(Xn, degree=2):
        cols = [np.ones(len(Xn))]
        for j in range(n_features):
            cols.append(Xn[:, j])
        if degree >= 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    cols.append(Xn[:, i] * Xn[:, j])
        return np.column_stack(cols)
    
    # Try multiple models and pick the best
    best_mse = np.inf
    best_predict = None
    best_name = ""
    
    model_candidates = []
    
    # 1. Polynomial models
    for deg in [1, 2, 3]:
        if n_features <= 3:
            model_candidates.append(("poly%d" % deg, build_polynomial, (deg,)))
        else:
            if deg <= 2:
                model_candidates.append(("poly%d" % deg, build_polynomial, (deg,)))
    
    # 2. Interaction models (cheaper than full polynomial for high dim)
    model_candidates.append(("interact2", build_interactions, (2,)))
    
    # 3. Trig models
    model_candidates.append(("trig", build_trig, ()))
    
    # 4. Exp/log models
    model_candidates.append(("expl", build_exp_log, ()))
    
    # 5. Rational models
    model_candidates.append(("rational", build_rational, ()))
    
    # 6. Spline models
    model_candidates.append(("spline", build_spline, ()))
    
    for name, builder, args in model_candidates:
        if time_remaining() < 5:
            break
        try:
            F_train = builder(X_norm, *args)
            # Check for NaN/inf
            if not np.all(np.isfinite(F_train)):
                F_train = np.nan_to_num(F_train, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Standardize features
            F_mean = F_train.mean(axis=0)
            F_std = F_train.std(axis=0)
            F_std[F_std < 1e-12] = 1.0
            F_stdized = (F_train - F_mean) / F_std
            
            mse, pred_fn = fit_linear(F_stdized, y_norm)
            
            if mse < best_mse:
                best_mse = mse
                best_name = name
                
                # Create predict function that normalizes input and applies the same feature transform
                def make_predict(builder_fn, b_args, F_mean_val, F_std_val, w_val):
                    def predict(X):
                        Xn = (X - X_min) / X_range
                        F = builder_fn(Xn, *b_args)
                        if not np.all(np.isfinite(F)):
                            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
                        F = (F - F_mean_val) / F_std_val
                        return (F @ w_val) * y_std + y_mean
                    return predict
                
                # Refit to get weights
                w, *_ = np.linalg.lstsq(F_stdized, y_norm, rcond=None)
                best_predict = make_predict(builder, args, F_mean, F_std, w)
        except Exception:
            continue
    
    # If no model worked, fall back to simple linear
    if best_predict is None:
        try:
            w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n_samples), X_norm]), y, rcond=None)
            def predict_linear(X):
                Xn = (X - X_min) / X_range
                F = np.column_stack([np.ones(len(Xn)), Xn])
                return F @ w
            return predict_linear
        except Exception:
            return lambda X: np.full(len(X), y_mean)
    
    # Final validation
    try:
        y_pred = best_predict(X)
        if not np.all(np.isfinite(y_pred)):
            y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
        
        def safe_predict(X):
            try:
                pred = best_predict(X)
                if not np.all(np.isfinite(pred)):
                    pred = np.nan_to_num(pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
                return pred
            except Exception:
                return np.full(len(X), y_mean)
        
        return safe_predict
    except Exception:
        return lambda X: np.full(len(X), y_mean)

# EVOLVE-BLOCK-END
