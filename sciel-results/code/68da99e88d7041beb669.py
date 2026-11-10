import numpy as np
from scipy.linalg import solve
import time

def fit(X_train, y_train):
    """
    Fits a Sparse Radial Basis Function (RBF) network.
    Uses K-means to select centers and linear least squares to solve weights.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s buffer for safety
    
    # 1. Preprocessing: Standardize inputs
    X_mean = X_train.mean(axis=0)
    X_std_dev = X_train.std(axis=0)
    X_std_dev[X_std_dev == 0] = 1.0
    
    X_train_std = (X_train - X_mean) / X_std_dev
    y_mean = y_train.mean()
    y_std_dev = y_train.std()
    y_std_dev = y_std_dev if y_std_dev != 0 else 1.0
    y_train_std = (y_train - y_mean) / y_std_dev
    
    N, D = X_train_std.shape
    
    # 2. Hyperparameter Search
    # We search over the number of centers (K) and the RBF width (sigma)
    # K determines the complexity. Sigma determines the smoothness.
    
    # Candidate K values (number of RBF centers)
    # Since N=200, K up to 50 is reasonable and fast.
    K_values = [5, 10, 20, 30, 50]
    
    # Candidate sigmas (RBF width)
    # K(x, c) = exp(-||x-c||^2 / (2 * sigma^2))
    # Small sigma -> sharp peaks, high variance
    # Large sigma -> smooth, low variance
    sigmas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    # Split for validation
    indices = np.arange(N)
    val_size = int(0.2 * N)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]
    
    X_tr = X_train_std[train_idx]
    y_tr = y_train_std[train_idx]
    X_val = X_train_std[val_idx]
    y_val = y_train_std[val_idx]
    
    best_val_mse = np.inf
    best_K = 10
    best_sigma = 1.0
    best_centers = None
    best_weights = None
    
    # Helper function to run K-means
    # Simple implementation to avoid dependency on sklearn if not available, 
    # but sklearn is often available. However, the prompt says NumPy and SciPy.
    # I will implement a simple K-means using NumPy to be safe and fast for small N.
    def simple_kmeans(X, K, max_iters=50, tol=1e-4):
        n_samples = X.shape[0]
        # Initialize centers randomly
        rng = np.random.default_rng(42)
        init_indices = rng.choice(n_samples, K, replace=False)
        centers = X[init_indices].copy()
        
        for _ in range(max_iters):
            # Assign points to nearest center
            # Distances: (N, K)
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs**2, axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centers
            new_centers = np.zeros_like(centers)
            for k in range(K):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    # If a cluster is empty, reinitialize to a random point
                    new_centers[k] = X[rng.choice(n_samples)]
            
            # Check convergence
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < tol:
                break
        
        return centers

    # Pre-compute nothing, just loop
    for K in K_values:
        # Time check
        if time.time() > deadline - 10:
            break
            
        # Get centers from training data
        centers = simple_kmeans(X_tr, K)
        
        for sigma in sigmas:
            # Compute design matrix Phi for training set
            # Phi[i, j] = exp(-||X_tr[i] - centers[j]||^2 / (2 * sigma^2))
            
            # Distances between X_tr and centers
            # X_tr: (N_tr, D), centers: (K, D)
            # dists: (N_tr, K)
            diffs = X_tr[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diffs**2, axis=2)
            
            Phi_tr = np.exp(-dists_sq / (2 * sigma**2))
            
            # Solve linear least squares: Phi_tr @ w = y_tr
            # Add small regularization for stability
            reg = 1e-6
            try:
                # Using lstsq is robust
                weights, residuals, rank, s = np.linalg.lstsq(Phi_tr + reg * np.eye(K), y_tr, rcond=None)
            except:
                weights = np.zeros(K)
            
            # Evaluate on validation set
            diffs_val = X_val[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq_val = np.sum(diffs_val**2, axis=2)
            Phi_val = np.exp(-dists_sq_val / (2 * sigma**2))
            
            y_val_pred = Phi_val @ weights
            val_mse = np.mean((y_val - y_val_pred)**2)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_K = K
                best_sigma = sigma
                best_centers = centers
                best_weights = weights
        
        # Time check inside K loop
        if time.time() > deadline - 10:
            break
    
    # 3. Refit on full training set with best hyperparameters
    K = best_K
    sigma = best_sigma
    
    # Recalculate centers on full training set? 
    # Usually, it's better to use the centers found during search or recalculate on full data.
    # Recalculating on full data ensures the model uses all information for center placement.
    centers_final = simple_kmeans(X_train_std, K)
    
    # Compute design matrix for full training set
    diffs_all = X_train_std[:, np.newaxis, :] - centers_final[np.newaxis, :, :]
    dists_sq_all = np.sum(diffs_all**2, axis=2)
    Phi_all = np.exp(-dists_sq_all / (2 * sigma**2))
    
    # Solve for weights
    reg = 1e-6
    try:
        w_final, residuals, rank, s = np.linalg.lstsq(Phi_all + reg * np.eye(K), y_train_std, rcond=None)
    except:
        w_final = np.zeros(K)
    
    # 4. Define predict function
    def predict(X):
        # Standardize X
        # Use X_std_dev from outer scope (renamed to avoid collision)
        X_test_std = (X - X_mean) / X_std_dev
        
        # Compute kernel between test and centers
        diffs_test = X_test_std[:, np.newaxis, :] - centers_final[np.newaxis, :, :]
        dists_sq_test = np.sum(diffs_test**2, axis=2)
        Phi_test = np.exp(-dists_sq_test / (2 * sigma**2))
        
        y_pred_std = Phi_test @ w_final
        
        # Inverse standardize
        y_pred = y_pred_std * y_std_dev + y_mean
        
        return y_pred
    
    return predict
