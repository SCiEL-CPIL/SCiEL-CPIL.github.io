"""
Symbolic regression via adaptive piecewise linear interpolation with multi-resolution search.
Since all hidden equations have 1 input, we exploit this to use 1D interpolation.
We try multiple bin counts and pick the one with best validation performance.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # safety margin
    
    X = X_train.reshape(-1, 1)
    y = y_train
    n = len(y)
    
    # Sort by input for 1D interpolation
    order = np.argsort(X[:, 0])
    X_sorted = X[order]
    y_sorted = y[order]
    
    # Split into train/val for model selection
    # Use 80/20 split
    n_val = max(20, n // 5)
    n_tr = n - n_val
    
    # Shuffle for split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    
    X_tr = X_sorted[tr_idx]
    y_tr = y_sorted[tr_idx]
    X_val = X_sorted[val_idx]
    y_val = y_sorted[val_idx]
    
    # Try multiple bin counts for piecewise linear interpolation
    best_score = np.inf
    best_bins = 50
    best_predict = None
    
    # Also try a few other approaches as fallbacks
    candidate_bins = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]
    
    # Time check
    if time.time() > deadline - 60:
        # Just use a reasonable default
        best_bins = 50
    else:
        for bins in candidate_bins:
            if time.time() > deadline - 30:
                break
            
            # Build piecewise linear interpolator
            # Create bin edges
            x_min = X_sorted[0, 0]
            x_max = X_sorted[-1, 0]
            edges = np.linspace(x_min, x_max, bins + 1)
            
            # For each bin, compute mean y value
            bin_indices = np.digitize(X_sorted[:, 0], edges) - 1
            bin_indices = np.clip(bin_indices, 0, bins - 1)
            
            bin_means = np.zeros(bins)
            bin_counts = np.zeros(bins)
            for b in range(bins):
                mask = bin_indices == b
                if np.any(mask):
                    bin_means[b] = np.mean(y_sorted[mask])
                    bin_counts[b] = np.sum(mask)
                else:
                    bin_counts[b] = 0
            
            # For empty bins, interpolate from neighbors
            empty = bin_counts == 0
            if np.any(empty):
                non_empty = ~empty
                if np.sum(non_empty) >= 2:
                    empty_idx = np.where(empty)[0]
                    non_empty_idx = np.where(non_empty)[0]
                    for ei in empty_idx:
                        # Find nearest non-empty neighbors
                        left = non_empty_idx[non_empty_idx < ei]
                        right = non_empty_idx[non_empty_idx > ei]
                        if len(left) > 0 and len(right) > 0:
                            li = left[-1]
                            ri = right[0]
                            frac = (ei - li) / (ri - li)
                            bin_means[ei] = bin_means[li] * (1 - frac) + bin_means[ri] * frac
                        elif len(left) > 0:
                            bin_means[ei] = bin_means[left[-1]]
                        elif len(right) > 0:
                            bin_means[ei] = bin_means[right[0]]
            
            # Build interpolation function
            bin_centers = (edges[:-1] + edges[1:]) / 2
            
            def predict_interp(X_pred, bin_centers=bin_centers, bin_means=bin_means, 
                              x_min=x_min, x_max=x_max):
                X_pred = np.asarray(X_pred).reshape(-1, 1)
                xp = X_pred[:, 0]
                # Clip to range
                xp_clipped = np.clip(xp, x_min, x_max)
                # Linear interpolation
                return np.interp(xp_clipped, bin_centers, bin_means)
            
            # Evaluate on validation set
            y_val_pred = predict_interp(X_val)
            val_mse = np.mean((y_val_pred - y_val) ** 2)
            
            if val_mse < best_score:
                best_score = val_mse
                best_bins = bins
                best_predict = predict_interp
    
    if best_predict is None:
        # Fallback: simple linear regression
        X_f = np.column_stack([np.ones(n), X[:, 0]])
        w, *_ = np.linalg.lstsq(X_f, y, rcond=None)
        def predict_lin(X_pred):
            X_pred = np.asarray(X_pred).reshape(-1, 1)
            return X_pred @ w[1] + w[0]
        return predict_lin
    
    return best_predict
# EVOLVE-BLOCK-END
