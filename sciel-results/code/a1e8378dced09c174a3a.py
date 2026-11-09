import numpy as np
from scipy.interpolate import UnivariateSpline, RBFInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    d = X.shape[1]
    
    # Handle potential edge cases or ensure finite values
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        # Fallback to simple mean if data is corrupted
        mean_y = np.mean(y)
        return lambda X_test: np.full(X_test.shape[0], mean_y)

    if d == 1:
        # 1D Spline Interpolation
        # x must be sorted
        X_1d = X[:, 0]
        order = np.argsort(X_1d)
        x_sorted = X_1d[order]
        y_sorted = y[order]
        
        # UnivariateSpline with s=0 for noiseless interpolation
        # If there are duplicate x values, spline might fail or behave oddly.
        # Let's handle duplicates by averaging y for same x.
        unique_x, unique_indices, count = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) < len(x_sorted):
            y_sum = np.zeros_like(unique_x)
            np.add.at(y_sum, unique_indices, y_sorted)
            y_avg = y_sum / count
            x_final = unique_x
            y_final = y_avg
        else:
            x_final = x_sorted
            y_final = y_sorted
            
        try:
            spline = UnivariateSpline(x_final, y_final, s=0, k=3)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                return spline(X_test[:, 0])
            return predict
        except Exception:
            # Fallback to linear interp or polyfit if spline fails
            try:
                coeffs = np.polyfit(x_final, y_final, 3)
                def predict(X_test):
                    X_test = np.asarray(X_test, dtype=float)
                    if X_test.ndim == 1:
                        X_test = X_test.reshape(-1, 1)
                    return np.polyval(coeffs, X_test[:, 0])
                return predict
            except Exception:
                mean_y = np.mean(y)
                return lambda X_test: np.full(X_test.shape[0], mean_y)
                
    else:
        # Multivariate RBF Interpolation
        # Thin Plate Spline is good for smooth functions
        try:
            interpolator = RBFInterpolator(X, y, kernel='thin_plate_spline', epsilon=1.0)
            # Note: epsilon is a scaling factor for the kernel. 
            # For TPS, the kernel is r^2 * log(r). 
            # The 'epsilon' parameter in RBFInterpolator scales the distances.
            # We might need to tune epsilon. 
            # A good heuristic is to use the median distance.
            
            # Let's compute a scale factor
            # Calculate pairwise distances is expensive for N=200? No, 200x200 is small.
            # But RBFInterpolator does this internally.
            
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(1, -1)
                return interpolator(X_test)
            return predict
        except Exception:
            # Fallback to KRR with RBF kernel
            def rbf_kernel(X1, X2, l=1.0):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-sq_dists / (2 * l**2))
            
            # Simple KRR
            # Estimate length scale
            # Use median distance
            # Sample a subset to estimate scale
            sample_idx = np.random.choice(X.shape[0], min(50, X.shape[0]), replace=False)
            X_sample = X[sample_idx]
            dists = np.sqrt(np.sum((X_sample[:, None, :] - X_sample[None, :, :])**2, axis=-1))
            l = np.median(dists[dists > 0])
            if l == 0: l = 1.0
            
            K = rbf_kernel(X, X, l)
            alpha = np.linalg.solve(K + 1e-6 * np.eye(X.shape[0]), y)
            
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(1, -1)
                K_test = rbf_kernel(X_test, X, l)
                return K_test @ alpha
            return predict
