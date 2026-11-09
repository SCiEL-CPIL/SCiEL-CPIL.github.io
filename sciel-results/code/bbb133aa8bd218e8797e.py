"""Symbolic regression with hybrid trig+poly basis and kernel ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        # Linear terms
        for j in range(d):
            cols.append(Xrn[:, j])
        # Quadratic (including cross)
        for i in range(d):
            for j in range(i, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
        # Cubic pure terms
        for j in range(d):
            cols.append(Xrn[:, j] ** 3)
        # Sine/cosine with a few frequencies on each input
        for freq in [0.5, 1.0, 2.0]:
            for j in range(d):
                cols.append(np.sin(freq * Xrn[:, j]))
                cols.append(np.cos(freq * Xrn[:, j]))
        # Cross sin terms (pairwise) for low-d
        if d <= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(np.sin(1.0 * Xrn[:, i]) * np.cos(1.0 * Xrn[:, j]))
                    cols.append(np.cos(1.0 * Xrn[:, i]) * np.sin(1.0 * Xrn[:, j]))
        return np.column_stack(cols)

    def sos_kernel(X1, X2, freqs):
        """Sum-of-Sines kernel: K(x1, x2) = sum_k cos(freq_k * (x1 - x2))"""
        # X1: (n1, d), X2: (n2, d)
        # For each frequency vector, compute cos(freq . (x1 - x2))
        # This is equivalent to cos(freq.x1)*cos(freq.x2) + sin(freq.x1)*sin(freq.x2)
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for f in freqs:
            # f: (d,)
            sin_x1 = np.sin(X1 @ f)  # (n1,)
            cos_x1 = np.cos(X1 @ f)  # (n1,)
            sin_x2 = np.sin(X2 @ f)  # (n2,)
            cos_x2 = np.cos(X2 @ f)  # (n2,)
            K += np.outer(cos_x1, cos_x2) + np.outer(sin_x1, sin_x2)
        return K

    F = make_features(X)
    y_mean = y.mean()
    y_centered = y - y_mean
    
    # Generate a dense grid of frequency vectors for the SOS kernel
    # Use frequencies based on the input ranges
    freq_grid = []
    # Base frequencies per dimension
    base_freqs = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    if d == 1:
        for f in base_freqs:
            freq_grid.append(np.array([f]))
    elif d == 2:
        for f1 in base_freqs[:3]:
            for f2 in base_freqs[:3]:
                freq_grid.append(np.array([f1, f2]))
        # Also add single-dim frequencies
        for f in base_freqs:
            freq_grid.append(np.array([f, 0.0]))
            freq_grid.append(np.array([0.0, f]))
    elif d == 3:
        for f1 in base_freqs[:2]:
            for f2 in base_freqs[:2]:
                for f3 in base_freqs[:2]:
                    freq_grid.append(np.array([f1, f2, f3]))
        # Add pairwise combinations
        for i in range(d):
            for j in range(d):
                if i != j:
                    for f in base_freqs[:2]:
                        fv = np.zeros(d)
                        fv[i] = f
                        fv[j] = f
                        freq_grid.append(fv)
    
    # Remove duplicate frequency vectors
    seen = set()
    unique_freqs = []
    for f in freq_grid:
        key = tuple(np.round(f, 4))
        if key not in seen:
            seen.add(key)
            unique_freqs.append(f)
    freq_grid = unique_freqs
    
    # Compute kernel matrix
    K_train = sos_kernel(Xs, Xs, freq_grid)
    
    # Combine with polynomial features: create augmented kernel
    # K_aug = K_sos + lambda_poly * F F^T
    # We'll solve using the kernel approach with multiple regularization values
    
    best_pred_train = None
    best_train_score = np.inf
    best_alpha = None
    best_lambda = None
    
    for lam in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
        # Kernel ridge regression: alpha = (K + lam*I)^-1 y
        A = K_train + lam * np.eye(n)
        try:
            alpha = np.linalg.solve(A, y_centered)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(A, y_centered, rcond=None)[0]
        
        pred_train = K_train @ alpha + y_mean
        train_score = np.mean((pred_train - y) ** 2)
        
        if train_score < best_train_score:
            best_train_score = train_score
            best_alpha = alpha
            best_lambda = lam
    
    # Also try the original polynomial ridge regression
    best_poly_w = None
    best_poly_score = np.inf
    for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        A = F.T @ F + lam * np.eye(F.shape[1])
        b = F.T @ y_centered
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(F, y_centered, rcond=None)[0]
        pred = F @ w + y_mean
        score = np.mean((pred - y) ** 2)
        if score < best_poly_score:
            best_poly_score = score
            best_poly_w = w
    
    # Choose the better approach
    use_kernel = (best_train_score < best_poly_score) and (best_alpha is not None)
    
    if use_kernel:
        def predict(Xraw):
            Xr = np.asarray(Xraw, dtype=np.float64)
            Xrn = (Xr - mu) / sd
            K_test = sos_kernel(Xrn, Xs, freq_grid)
            return K_test @ best_alpha + y_mean
    else:
        if best_poly_w is None:
            return lambda Xraw: np.full(len(np.asarray(Xraw, dtype=np.float64)), y_mean)
        
        def predict(Xraw):
            Fr = make_features(Xraw)
            return Fr @ best_poly_w + y_mean
    
    return predict
# EVOLVE-BLOCK-END
