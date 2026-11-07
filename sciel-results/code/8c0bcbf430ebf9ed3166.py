# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fit a flexible regression model using an ensemble of basis functions.
    Returns a predict(X) callable.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n, d = X_train.shape
    y = y_train.reshape(-1)
    
    # Normalize X for numerical stability
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    def predict_norm(X_raw):
        """Normalize new data using training stats."""
        X_raw = np.asarray(X_raw)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        return (X_raw - X_mean) / X_std

    # Generate basis functions
    def make_polynomial_basis(X, max_deg=5):
        """Polynomial basis up to max_deg."""
        cols = [np.ones(len(X))]
        if X.shape[1] == 1:
            for deg in range(1, max_deg + 1):
                cols.append(X[:, 0] ** deg)
        else:
            # For multi-d, include monomials
            # Simplified: just powers of each feature and products up to degree 2
            for i in range(X.shape[1]):
                for deg in range(1, max_deg + 1):
                    cols.append(X[:, i] ** deg)
            if max_deg >= 2:
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    def make_fourier_basis(X, freqs):
        """Fourier basis: sin(freq*x) and cos(freq*x) for each frequency."""
        cols = [np.ones(len(X))]
        for f in freqs:
            for j in range(X.shape[1]):
                cols.append(np.sin(f * X[:, j]))
                cols.append(np.cos(f * X[:, j]))
        return np.column_stack(cols)
    
    def make_rbf_basis(X, centers, bandwidth):
        """RBF basis centered at given centers with a single bandwidth."""
        # X shape: (N, D), centers shape: (K, D)
        # dist2 shape: (N, K)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        rbf_vals = np.exp(-dist2 / (2 * bandwidth ** 2))
        return np.column_stack([np.ones(len(X)), rbf_vals])

    def ridge_fit(B, y, lam=1.0):
        """Fit ridge regression: min ||Bw - y||^2 + lam ||w||^2"""
        # B shape (N, F)
        # Solve (B.T B + lam I) w = B.T y
        try:
            A = B.T @ B + lam * np.eye(B.shape[1])
            b = B.T @ y
            w = np.linalg.solve(A, b)
            return w
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular
            w, *_ = np.linalg.lstsq(B, y, rcond=None)
            return w

    def cross_val_mse(X, y, basis_func, lam, n_folds=5):
        """Simple k-fold cross-validation MSE."""
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(indices) // n_folds
        mses = []
        for k in range(n_folds):
            val_idx = indices[k * fold_size : (k + 1) * fold_size]
            train_idx = np.concatenate([indices[:k * fold_size], indices[(k + 1) * fold_size:]])
            
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            try:
                B_tr = basis_func(X_tr)
                B_val = basis_func(X_val)
                w = ridge_fit(B_tr, y_tr, lam)
                pred = B_val @ w
                mse = np.mean((pred - y_val) ** 2)
                mses.append(mse)
            except Exception:
                mses.append(np.inf)
        return np.mean(mses)

    # Candidate models
    candidates = []
    
    # 1. Polynomial degree 5
    candidates.append(("poly5", lambda X: make_polynomial_basis(X, 5), 1.0))
    
    # 2. Polynomial degree 7
    candidates.append(("poly7", lambda X: make_polynomial_basis(X, 7), 10.0))
    
    # 3. Fourier with various frequencies
    freq_sets = [
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    ]
    for i, freqs in enumerate(freq_sets):
        candidates.append((f"fourier{i}", (lambda f: lambda X: make_fourier_basis(X, f))(freqs), 1.0))
    
    # 4. RBF with centers from quantiles
    if n > 10:
        n_centers = min(15, n)
        # Use evenly spaced centers in the normalized space
        # For 1D: linspace -3 to 3
        # For multiD: grid or random subset
        if d == 1:
            centers = np.linspace(-3, 3, n_centers).reshape(-1, 1)
        else:
            # Random subset of training points for centers
            center_indices = np.random.choice(n, n_centers, replace=False)
            centers = X_norm[center_indices]
        
        bandwidths = [0.5, 1.0, 2.0]
        for bw in bandwidths:
            candidates.append((f"rbf_bw{bw}", (lambda c, b: lambda X: make_rbf_basis(X, c, b))(centers, bw), 1.0))
    
    # Evaluate candidates
    best_name = None
    best_mse = np.inf
    best_basis_func = None
    best_lam = 1.0
    
    for name, basis_func, lam in candidates:
        if time.time() > deadline - 30:
            break
        try:
            mse = cross_val_mse(X_norm, y, basis_func, lam, n_folds=4)
            if mse < best_mse:
                best_mse = mse
                best_name = name
                best_basis_func = basis_func
                best_lam = lam
        except Exception:
            continue
    
    # If no good candidate, fall back to polynomial degree 3
    if best_basis_func is None:
        best_basis_func = lambda X: make_polynomial_basis(X, 3)
        best_name = "fallback_poly3"
        best_lam = 1.0
    
    # Fit final model with ridge regression
    B_train = best_basis_func(X_norm)
    w = ridge_fit(B_train, y, best_lam)
    
    def predict(X_raw):
        X_n = predict_norm(X_raw)
        B = best_basis_func(X_n)
        preds = B @ w
        # Ensure 1D output
        return preds.flatten()
    
    return predict
# EVOLVE-BLOCK-END
