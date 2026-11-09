import numpy as np
from scipy.interpolate import CubicSpline, Rbf
import time

def fit(X_train, y_train):
    X = X_train
    y = y_train
    N, D = X.shape
    
    if D == 1:
        x = X.flatten()
        # Sort data
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Handle duplicates
        unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        
        if len(unique_x) < N:
            # Average y for duplicates
            y_unique = np.zeros_like(unique_x)
            for i in range(len(unique_x)):
                mask = inverse_indices == i
                y_unique[i] = np.mean(y_sorted[mask])
            x_fit = unique_x
            y_fit = y_unique
        else:
            x_fit = x_sorted
            y_fit = y_sorted
            
        try:
            # Cubic Spline with not-a-knot boundary conditions
            spline = CubicSpline(x_fit, y_fit, bc_type='not-a-knot')
            
            def predict(X_test):
                x_test = X_test.flatten()
                return spline(x_test)
                
            return predict
            
        except Exception:
            pass

    # Fallback: RBF with normalized inputs
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Estimate epsilon based on mean distance
    # For normalized data, mean distance is roughly sqrt(D)
    # Let's try a few epsilons or just a standard one
    # RBF is sensitive to epsilon. 
    # For 1D normalized to ~[-1.5, 1.5], mean dist ~ 1. Epsilon 0.5-1.0 is good.
    # For higher D, epsilon needs to be smaller.
    # Let's use a heuristic: epsilon = 0.1 * sqrt(D) + 0.1 ?
    # Or just use the default 'gaussian' with epsilon estimated from data.
    
    # A safer fallback is to use scipy.stats.gaussian_kde or just nearest neighbor if RBF fails.
    # But Rbf is usually robust.
    
    try:
        # Try to estimate epsilon
        # Mean pairwise distance
        # This is O(N^2) which is fine for N=200
        dists = np.sqrt(np.sum((X_norm[:, None, :] - X_norm[None, :, :])**2, axis=2))
        mean_dist = np.mean(dists[dists > 0])
        epsilon = max(0.1, mean_dist / 2.0)
        
        rbf = Rbf(X_norm, y, function='gaussian', epsilon=epsilon)
        
        def predict(X_test):
            X_test_norm = (X_test - X_mean) / X_std
            return rbf(X_test_norm)
            
        return predict
        
    except Exception:
        # Final fallback: Mean
        y_mean = np.mean(y)
        def predict(X_test):
            return np.full(X_test.shape[0], y_mean)
        return predict
