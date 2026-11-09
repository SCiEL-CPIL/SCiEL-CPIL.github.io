"""Symbolic regression via sparse RBF network with k-means centers and exact least squares."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _kmeans(X, n_clusters, max_iters=100, n_init=10, seed=42):
    """Simple k-means implementation."""
    n_samples, n_features = X.shape
    n_clusters = min(n_clusters, n_samples)
    
    rng = np.random.RandomState(seed)
    best_inertia = float('inf')
    best_centroids = None
    
    for init in range(n_init):
        # Initialize centroids by random selection
        idx = rng.choice(n_samples, n_clusters, replace=False)
        centroids = X[idx].copy()
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        
        # Compute inertia
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        inertia = np.sum(dists[np.arange(n_samples), np.argmin(dists, axis=1)]**2)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
    
    return best_centroids

def _compute_rbf_features(X, centers, gamma):
    """Compute RBF feature matrix: phi(x) = exp(-gamma * ||x - c_j||^2) for each center c_j."""
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    
    # Compute squared distances
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    C_sq = np.sum(centers**2, axis=1, keepdims=True)
    sq_dists = X_sq + C_sq.T - 2 * (X @ centers.T)
    sq_dists = np.maximum(sq_dists, 0)
    
    return np.exp(-gamma * sq_dists)

def _build_poly_features(X, max_degree=3):
    """Build polynomial features up to max_degree."""
    n_samples, n_features = X.shape
    feature_list = [np.ones((n_samples, 1))]
    
    for deg in range(1, max_degree + 1):
        # Generate all monomials of degree deg
        from itertools import combinations_with_replacement
        for combo in combinations_with_replacement(range(n_features), deg):
            feat = np.ones(n_samples)
            for idx in combo:
                feat *= X[:, idx]
            feature_list.append(feat[:, np.newaxis])
    
    return np.hstack(feature_list)

def fit(X_train, y_train):
    """
    Fits a regression model using a sparse RBF network with k-means centers.
    Combines RBF features with polynomial features for diverse functional forms.
    Hyperparameters are tuned via validation split.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n_samples, n_features = X.shape
    
    # 1. Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-12] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # 2. Split into train and validation sets for model selection
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    val_size = int(0.2 * n_samples)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr = X_norm[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_norm[val_idx]
    y_val = y[val_idx]
    
    n_tr = len(y_tr)
    n_val = len(y_val)
    
    best_val_mse = float('inf')
    best_config = None
    
    # Hyperparameter grid
    n_centers_list = [10, 20, 30, 50]
    gamma_list = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    poly_degrees = [0, 2, 3]  # 0 means no poly features
    alphas = [1e-6, 1e-5, 1e-4, 1e-3]
    
    def evaluate_config(n_centers, gamma, poly_degree, alpha):
        if time.time() > deadline - 15:
            return float('inf'), None
        
        try:
            # Build RBF features
            centers_tr = _kmeans(X_tr, n_centers, n_init=5, seed=42)
            Phi_tr_rbf = _compute_rbf_features(X_tr, centers_tr, gamma)
            Phi_val_rbf = _compute_rbf_features(X_val, centers_tr, gamma)
            
            # Build polynomial features if needed
            if poly_degree > 0:
                Phi_tr_poly = _build_poly_features(X_tr, poly_degree)
                Phi_val_poly = _build_poly_features(X_val, poly_degree)
                Phi_tr = np.hstack([Phi_tr_rbf, Phi_tr_poly])
                Phi_val = np.hstack([Phi_val_rbf, Phi_val_poly])
            else:
                Phi_tr = Phi_tr_rbf
                Phi_val = Phi_val_rbf
            
            # Solve ridge regression
            n_features_total = Phi_tr.shape[1]
            A = Phi_tr.T @ Phi_tr + alpha * np.eye(n_features_total)
            b = Phi_tr.T @ y_tr
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A, b, rcond=None)
            
            # Evaluate on validation
            y_val_pred = Phi_val @ w
            mse_val = np.mean((y_val_pred - y_val)**2)
            
            return mse_val, (n_centers, gamma, poly_degree, alpha)
            
        except Exception:
            return float('inf'), None
    
    # Search grid
    for n_centers in n_centers_list:
        for gamma in gamma_list:
            for poly_degree in poly_degrees:
                for alpha in alphas:
                    if time.time() > deadline - 10:
                        break
                    mse, cfg = evaluate_config(n_centers, gamma, poly_degree, alpha)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = cfg
    
    # Fallback if no config found
    if best_config is None:
        best_config = (20, 5.0, 2, 1e-4)
    
    n_centers, gamma, poly_degree, best_alpha = best_config
    
    # Refit on ALL data with the best config
    centers_full = _kmeans(X_norm, n_centers, n_init=10, seed=42)
    Phi_full_rbf = _compute_rbf_features(X_norm, centers_full, gamma)
    
    if poly_degree > 0:
        Phi_full_poly = _build_poly_features(X_norm, poly_degree)
        Phi_full = np.hstack([Phi_full_rbf, Phi_full_poly])
    else:
        Phi_full = Phi_full_rbf
    
    n_features_total = Phi_full.shape[1]
    A_full = Phi_full.T @ Phi_full + best_alpha * np.eye(n_features_total)
    b_full = Phi_full.T @ y
    
    try:
        w_full = np.linalg.solve(A_full, b_full)
    except np.linalg.LinAlgError:
        w_full, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
    
    # Capture variables for closure
    final_w = w_full
    final_centers = centers_full
    final_gamma = gamma
    final_poly_degree = poly_degree
    final_min_vals = min_vals
    final_range_vals = range_vals

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_norm_test = (X - final_min_vals) / final_range_vals
        
        # Build RBF features
        Phi_test_rbf = _compute_rbf_features(X_norm_test, final_centers, final_gamma)
        
        # Build polynomial features if needed
        if final_poly_degree > 0:
            Phi_test_poly = _build_poly_features(X_norm_test, final_poly_degree)
            Phi_test = np.hstack([Phi_test_rbf, Phi_test_poly])
        else:
            Phi_test = Phi_test_rbf
        
        return Phi_test @ final_w

    return predict
# EVOLVE-BLOCK-END
