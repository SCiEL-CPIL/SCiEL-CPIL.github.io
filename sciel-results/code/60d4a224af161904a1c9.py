# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fit a flexible regression model using an ensemble of basis functions:
    - Polynomial features (up to degree 5)
    - Fourier features (sin/cos at multiple frequencies)
    - RBF features centered at a subset of training points
    
    Uses ridge regression to combine them. Selects the best model via
    cross-validation on the training set.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    n, d = X_train.shape
    y = y_train
    
    # Normalize X for numerical stability in basis functions
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    def predict_norm(X_raw):
        """Normalize new data using training stats."""
        return (X_raw - X_mean) / X_std
    
    # Generate basis functions
    def make_polynomial_basis(X, max_deg=5):
        """Polynomial basis up to max_deg."""
        cols = [np.ones(len(X))]
        # Include all monomials up to max_deg
        for deg in range(1, max_deg + 1):
            # For 1D case, just powers
            if X.shape[1] == 1:
                cols.append(X[:, 0] ** deg)
            else:
                # For multi-d, include pairwise products up to degree
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        cols.append((X[:, i] ** (deg // 2 + (deg % 2))) * (X[:, j] ** (deg // 2)))
        return np.column_stack(cols)
    
    def make_fourier_basis(X, freqs):
        """Fourier basis: sin(freq*x) and cos(freq*x) for each frequency."""
        cols = [np.ones(len(X))]
        for f in freqs:
            for j in range(X.shape[1]):
                cols.append(np.sin(f * X[:, j]))
                cols.append(np.cos(f * X[:, j]))
        return np.column_stack(cols)
    
    def make_rbf_basis(X, centers, bandwidths):
        """RBF basis centered at given centers with given bandwidths."""
        cols = [np.ones(len(X))]
        for c, bw in zip(centers, bandwidths):
            dist2 = np.sum((X - c) ** 2, axis=1)
            cols.append(np.exp(-dist2 / (2 * bw ** 2)))
        return np.column_stack(cols)
    
    # Try different models and pick best via simple cross-validation
    def cross_val_mse(X, y, basis_func, n_folds=5):
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
                # Ridge regression
                lam = 1e-3
                w, *_ = np.linalg.lstsq(B_tr + lam * np.eye(len(B_tr)), y_tr, rcond=None)
                pred = B_val @ w
                mse = np.mean((pred - y_val) ** 2)
                mses.append(mse)
            except:
                mses.append(np.inf)
        return np.mean(mses)
    
    # Candidate models
    candidates = []
    
    # 1. Polynomial degree 5
    candidates.append(("poly5", lambda X: make_polynomial_basis(X, 5)))
    
    # 2. Polynomial degree 7
    candidates.append(("poly7", lambda X: make_polynomial_basis(X, 7)))
    
    # 3. Fourier with various frequencies
    freq_sets = [
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        [0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    ]
    for i, freqs in enumerate(freq_sets):
        candidates.append((f"fourier{i}", (lambda f: lambda X: make_fourier_basis(X, f))(freqs)))
    
    # 4. RBF with centers from quantiles
    if n > 10:
        # Choose centers at quantiles
        n_centers = min(10, n)
        quantiles = np.linspace(0.05, 0.95, n_centers)
        centers = [X_norm[np.random.choice(n, int(q * n), replace=False)[0]] for q in quantiles]
        # Actually, let's use evenly spaced centers
        centers = []
        for j in range(d):
            pass
        # Simpler: use random subset
        np.random.seed(0)
        center_indices = np.random.choice(n, min(10, n), replace=False)
        centers = X_norm[center_indices]
        # Bandwidths: estimate from data
        if n > 1:
            dists = np.sqrt(np.sum((X_norm - X_norm[0]) ** 2, axis=1))
            median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
            bandwidths = [median_dist * 0.5, median_dist, median_dist * 2.0]
        else:
            bandwidths = [1.0]
        
        for bw in bandwidths:
            candidates.append((f"rbf_bw{bw:.2f}", (lambda c, b: lambda X: make_rbf_basis(X, c, [b] * len(c)))(centers, bw)))
    
    # 5. Combined: polynomial + fourier
    def combined_basis(X):
        p = make_polynomial_basis(X, 4)
        f = make_fourier_basis(X, [1.0, 2.0, 3.0, 4.0, 5.0])
        return np.column_stack([p, f])
    candidates.append(("combined", combined_basis))
    
    # Evaluate candidates
    best_name = None
    best_mse = np.inf
    best_basis_func = None
    
    for name, basis_func in candidates:
        if time.time() > deadline - 30:
            break
        try:
            mse = cross_val_mse(X_norm, y, basis_func, n_folds=4)
            if mse < best_mse:
                best_mse = mse
                best_name = name
                best_basis_func = basis_func
        except:
            continue
    
    # If no good candidate, fall back to polynomial degree 3
    if best_basis_func is None:
        best_basis_func = lambda X: make_polynomial_basis(X, 3)
        best_name = "fallback_poly3"
    
    # Fit final model with ridge regression
    B_train = best_basis_func(X_norm)
    lam = 1e-3
    w, *_ = np.linalg.lstsq(B_train + lam * np.eye(len(B_train)), y, rcond=None)
    
    def predict(X_raw):
        X_n = predict_norm(X_raw)
        B = best_basis_func(X_n)
        return B @ w
    
    return predict
# EVOLVE-BLOCK-END
