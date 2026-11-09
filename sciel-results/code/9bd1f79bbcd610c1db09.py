"""Symbolic regression via explicit feature construction + ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_poly_features(X, max_deg):
    """Build polynomial feature matrix up to max_deg (includes constant)."""
    n, d = X.shape
    features = [np.ones((n, 1))]
    # Linear
    features.append(X.copy())
    # Quadratic and higher
    if max_deg >= 2:
        quad = np.zeros((n, d * (d + 1) // 2))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                quad[:, idx] = X[:, i] * X[:, j]
                idx += 1
        features.append(quad)
    if max_deg >= 3:
        cub = np.zeros((n, d * (d + 1) * (d + 2) // 6))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cub[:, idx] = X[:, i] * X[:, j] * X[:, k]
                    idx += 1
        features.append(cub)
    return np.hstack(features)


def _build_fourier_features(X, n_freqs):
    """Build Fourier features: sin/cos of various frequencies."""
    n, d = X.shape
    features = [np.ones((n, 1))]
    for f in range(1, n_freqs + 1):
        for dim in range(d):
            features.append(np.sin(2 * np.pi * f * X[:, dim]).reshape(-1, 1))
            features.append(np.cos(2 * np.pi * f * X[:, dim]).reshape(-1, 1))
    return np.hstack(features)


def _build_rbf_features(X, centers, bandwidths):
    """Build RBF features centered at given centers with given bandwidths."""
    n, d = X.shape
    features = [np.ones((n, 1))]
    for center, bw in zip(centers, bandwidths):
        dist = np.sum((X - center) ** 2, axis=1)
        features.append(np.exp(-dist / (2 * bw ** 2)).reshape(-1, 1))
    return np.hstack(features)


def _build_mlp_features(X, n_hidden, activation='tanh'):
    """Build features from a single hidden layer MLP (no training, just fixed random init)."""
    n, d = X.shape
    rng = np.random.RandomState(42)
    W1 = rng.randn(d, n_hidden) * np.sqrt(2.0 / d)
    b1 = rng.randn(n_hidden) * 0.1
    Z = X @ W1 + b1
    if activation == 'tanh':
        A = np.tanh(Z)
    elif activation == 'relu':
        A = np.maximum(0, Z)
    else:
        A = Z
    return np.hstack([np.ones((n, 1)), X, A])


def _ridge_solve(A, y, lam):
    """Solve (A^T A + lam*I) w = A^T y efficiently."""
    n = A.shape[0]
    # Use normal equations with regularization
    ATA = A.T @ A
    ATy = A.T @ y
    I = np.eye(ATA.shape[0])
    return np.linalg.solve(ATA + lam * I, ATy)


def fit(X_train, y_train):
    """Fit using explicit feature construction + ridge regression with model selection."""
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize to [0,1]
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Generate candidate feature sets
    feature_sets = {}
    
    # 1. Polynomial features
    for deg in [1, 2, 3]:
        if time.time() > deadline - 20:
            break
        F = _build_poly_features(X_norm, deg)
        feature_sets[f'poly_{deg}'] = F
    
    # 2. Fourier features
    for n_freqs in [2, 3, 4]:
        if time.time() > deadline - 20:
            break
        F = _build_fourier_features(X_norm, n_freqs)
        feature_sets[f'fourier_{n_freqs}'] = F
    
    # 3. RBF features with various center/bandwidth configs
    rng = np.random.RandomState(123)
    n_rbf = 20
    centers = rng.rand(n_rbf, d)
    for n_bw in [1, 3]:
        if time.time() > deadline - 20:
            break
        if n_bw == 1:
            bandwidths = [0.3] * n_rbf
        else:
            bandwidths = [0.2, 0.5, 1.0] * (n_rbf // 3 + 1)
            bandwidths = bandwidths[:n_rbf]
        F = _build_rbf_features(X_norm, centers, bandwidths)
        feature_sets[f'rbf_bw{n_bw}'] = F
    
    # 4. MLP-like features
    for n_hidden in [16, 32, 64]:
        if time.time() > deadline - 20:
            break
        F = _build_mlp_features(X_norm, n_hidden, 'tanh')
        feature_sets[f'mlp_tanh_{n_hidden}'] = F
        F2 = _build_mlp_features(X_norm, n_hidden, 'relu')
        feature_sets[f'mlp_relu_{n_hidden}'] = F2
    
    # 5. Combined: poly + fourier
    if time.time() < deadline - 20:
        F_poly2 = _build_poly_features(X_norm, 2)
        F_four2 = _build_fourier_features(X_norm, 2)
        feature_sets['combined_poly4_four2'] = np.hstack([F_poly2, F_four2])
    
    # Model selection via simple 2-fold CV
    best_score = np.inf
    best_key = None
    best_coef = None
    best_lam = None
    
    lam_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    half = n // 2
    
    for key, F in feature_sets.items():
        if time.time() > deadline - 10:
            break
        
        F_train1 = F[idx[:half]]
        y_train1 = y[idx[:half]]
        F_val1 = F[idx[half:]]
        y_val1 = y[idx[half:]]
        
        F_train2 = F[idx[half:]]
        y_train2 = y[idx[half:]]
        F_val2 = F[idx[:half]]
        y_val2 = y[idx[:half]]
        
        for lam in lam_values:
            try:
                # Fold 1
                w1 = _ridge_solve(F_train1, y_train1, lam)
                pred1 = F_val1 @ w1
                mse1 = np.mean((pred1 - y_val1) ** 2)
                
                # Fold 2
                w2 = _ridge_solve(F_train2, y_train2, lam)
                pred2 = F_val2 @ w2
                mse2 = np.mean((pred2 - y_val2) ** 2)
                
                score = (mse1 + mse2) / 2.0
                
                if score < best_score:
                    best_score = score
                    best_key = key
                    best_lam = lam
                    # Solve on full training set
                    best_coef = _ridge_solve(F, y, lam)
                    
            except Exception:
                continue
    
    # Fallback if nothing worked
    if best_key is None:
        # Simple linear ridge
        F = np.hstack([np.ones((n, 1)), X_norm])
        best_coef = _ridge_solve(F, y, 1e-3)
        best_key = 'fallback_linear'
        best_lam = 1e-3
        feature_sets[best_key] = F
    
    # Store what we need for prediction
    F_train_full = feature_sets[best_key]
    X_min_stored = X_min
    X_range_stored = X_range
    best_coef_stored = best_coef
    best_key_stored = best_key
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_min_stored) / X_range_stored
        
        # Rebuild features using the same configuration
        if best_key_stored.startswith('poly_'):
            deg = int(best_key_stored.split('_')[1])
            F_new = _build_poly_features(X_new_norm, deg)
        elif best_key_stored.startswith('fourier_'):
            nf = int(best_key_stored.split('_')[1])
            F_new = _build_fourier_features(X_new_norm, nf)
        elif best_key_stored.startswith('rbf_'):
            # Rebuild RBF features
            rng = np.random.RandomState(123)
            n_rbf = 20
            centers = rng.rand(n_rbf, d)
            if 'bw1' in best_key_stored:
                bandwidths = [0.3] * n_rbf
            else:
                bandwidths = [0.2, 0.5, 1.0] * (n_rbf // 3 + 1)
                bandwidths = bandwidths[:n_rbf]
            F_new = _build_rbf_features(X_new_norm, centers, bandwidths)
        elif best_key_stored.startswith('mlp_tanh_'):
            nh = int(best_key_stored.split('_')[2])
            F_new = _build_mlp_features(X_new_norm, nh, 'tanh')
        elif best_key_stored.startswith('mlp_relu_'):
            nh = int(best_key_stored.split('_')[2])
            F_new = _build_mlp_features(X_new_norm, nh, 'relu')
        elif best_key_stored == 'combined_poly4_four2':
            F_p = _build_poly_features(X_new_norm, 2)
            F_f = _build_fourier_features(X_new_norm, 2)
            F_new = np.hstack([F_p, F_f])
        else:  # fallback linear
            F_new = np.hstack([np.ones((X_new_norm.shape[0], 1)), X_new_norm])
        
        return F_new @ best_coef_stored
    
    return predict
# EVOLVE-BLOCK-END
