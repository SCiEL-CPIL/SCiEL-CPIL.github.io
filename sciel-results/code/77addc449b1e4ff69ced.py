import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.spatial import cKDTree

def fit(X_train, y_train):
    X = X_train
    y = y_train
    
    # Check if 1D
    if X.shape[1] == 1:
        x = X.flatten()
        # Handle duplicates: sort and average
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Remove duplicates by averaging y for same x
        unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) < len(x_sorted):
            y_sum = np.zeros_like(unique_x, dtype=np.float64)
            np.add.at(y_sum, inverse_indices, y_sorted)
            y_unique = y_sum / counts
        else:
            unique_x = x_sorted
            y_unique = y_sorted
            
        # Fit Cubic Spline
        # CubicSpline requires strictly increasing x
        try:
            spline = CubicSpline(unique_x, y_unique)
        except Exception:
            # Fallback to UnivariateSpline with minimal smoothing if CubicSpline fails
            try:
                spline = UnivariateSpline(unique_x, y_unique, s=0)
            except Exception:
                # Fallback to KNN
                tree = cKDTree(X)
                def predict(X_test):
                    dists, idxs = tree.query(X_test, k=5)
                    # Weighted average
                    weights = 1.0 / (dists + 1e-8)
                    preds = np.sum(y_train[idxs] * weights[:, None], axis=1) / np.sum(weights, axis=1)
                    return preds
                return predict

        def predict(X_test):
            x_test = X_test.flatten()
            return spline(x_test)
        
        return predict
    else:
        # Fallback for multi-dimensional (not expected based on contract, but safe)
        # Use KNN with small K
        tree = cKDTree(X)
        k = 5
        def predict(X_test):
            dists, idxs = tree.query(X_test, k=k)
            if dists.shape[1] == 1:
                dists = dists.reshape(-1, 1)
                idxs = idxs.reshape(-1, 1)
            weights = 1.0 / (dists + 1e-8)
            preds = np.sum(y_train[idxs] * weights[:, None], axis=1) / np.sum(weights, axis=1)
            return preds
        return predict
