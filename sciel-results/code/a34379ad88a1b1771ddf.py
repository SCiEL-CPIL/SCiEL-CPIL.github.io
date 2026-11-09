# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    def check_time():
        return time.time() < deadline
    
    best_predict = None
    best_val_mse = np.inf
    
    # Simple validation split
    val_frac = 0.15
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]
    
    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    def update_best(model):
        nonlocal best_predict, best_val_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_val, y_val)
        if mse < best_val_mse:
            best_val_mse = mse
            best_predict = model
    
    # --- Sum-of-Sines Kernel Ridge Regression ---
    # Strategy: Use a dense grid of frequency components for a sum-of-sines kernel.
    # K(x, x') = sum_{k=1}^{K} sin(k * x * w) * sin(k * x' * w) + cos terms
    # This is equivalent to a Fourier basis with fixed frequencies.
    
    def make_sines_kernel_model(frequencies, regularization=1e-6):
        """
        frequencies: list of base frequencies to use
        """
        # Build the feature matrix for training data
        # For each input dimension j and each frequency f:
        #   sin(f * X[:, j]), cos(f * X[:, j])
        # Also add constant term.
        
        def build_features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                for f in frequencies:
                    cols.append(np.sin(f * X[:, j]))
                    cols.append(np.cos(f * X[:, j]))
            return np.column_stack(cols)
        
        F_tr = build_features(X_tr)
        
        try:
            # Ridge regression
            A = F_tr.T @ F_tr
            b = F_tr.T @ y_tr
            A_reg = A.copy()
            A_reg[1:, 1:] += regularization
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
        
        def predict(X):
            F_test = build_features(X)
            return F_test @ w
        
        return predict
    
    # Determine appropriate frequency grid
    # Input ranges are known: [1, 3], [1, 2], [1, 5], [1, 10]
    # We'll use a grid of frequencies that covers a reasonable range.
    # For a range of size R, frequencies up to ~pi/R * max_order
    
    # Estimate the range of each input
    X_range = np.ptp(X_train, axis=0)
    avg_range = np.mean(X_range)
    if avg_range == 0:
        avg_range = 1.0
    
    # Create a dense grid of frequencies
    # Base frequency scale: 1 / avg_range
    # We'll use frequencies: 1, 2, 3, ..., max_freq
    # where max_freq is chosen based on the range
    
    # For range ~2, use frequencies up to ~10
    # For range ~4, use frequencies up to ~5
    # For range ~9, use frequencies up to ~3
    
    max_freq = int(20 / max(avg_range, 1.0))
    max_freq = max(3, min(max_freq, 15))  # Clamp to reasonable range
    
    frequencies = list(range(1, max_freq + 1))
    
    # Try different regularization values
    for reg in [1e-8, 1e-6, 1e-4, 1e-2]:
        if not check_time():
            break
        model = make_sines_kernel_model(frequencies, regularization=reg)
        update_best(model)
    
    # Also try a sparser frequency grid (every other frequency)
    if check_time():
        sparse_freqs = list(range(1, max_freq + 1, 2))
        if sparse_freqs:
            for reg in [1e-6, 1e-4]:
                if not check_time():
                    break
                model = make_sines_kernel_model(sparse_freqs, regularization=reg)
                update_best(model)
    
    # Try even sparser
    if check_time():
        very_sparse = list(range(1, max_freq + 1, 4))
        if very_sparse:
            for reg in [1e-6, 1e-4]:
                if not check_time():
                    break
                model = make_sines_kernel_model(very_sparse, regularization=reg)
                update_best(model)
    
    # Fallback: simple linear model
    if best_predict is None:
        # Linear regression
        X_tr_aug = np.column_stack([np.ones(n_train), X_tr])
        try:
            w, *_ = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)
            def predict(X):
                X_aug = np.column_stack([np.ones(X.shape[0]), X])
                return X_aug @ w
        except Exception:
            return lambda X: np.full(X.shape[0], y_mean)
        update_best(predict)
    
    # If still no model, use mean
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean)
    
    return best_predict

# EVOLVE-BLOCK-END
