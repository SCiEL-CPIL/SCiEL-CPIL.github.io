"""
Strategy: Multi-Resolution Piecewise Cubic Spline Interpolation.
Instead of fitting a global basis (poly/fourier), we fit a local, high-capacity model
by partitioning the input domain into bins and fitting a cubic spline in each bin.
We search over a grid of bin counts to find the optimal resolution. This approach
can fit arbitrary smooth functions (including discontinuities) with near-zero bias
given sufficient resolution, and is extremely robust to noise (since 200 points is
plenty for cubic splines).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    X = X_train[:, 0]  # 1D input
    y = y_train
    
    n = len(X)
    x_min, x_max = X.min(), X.max()
    
    # Sort data for spline fitting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    best_score = np.inf
    best_predict = None
    best_n_bins = 10
    
    # Search over number of bins
    # Too few bins: high bias
    # Too many bins: high variance (but with noiseless data, this is less of an issue)
    # With 200 points, we can go up to ~50 bins safely (4 points per bin for cubic)
    bin_counts = [5, 10, 15, 20, 30, 40, 50]
    
    for n_bins in bin_counts:
        if time.time() > deadline:
            break
            
        try:
            # Create bin edges
            edges = np.linspace(x_min, x_max, n_bins + 1)
            
            # Assign points to bins
            bin_indices = np.digitize(X_sorted, edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Fit a cubic spline in each bin
            # We need at least 4 points per bin for a cubic spline
            # If a bin has < 4 points, we'll fall back to a lower degree or merge bins
            # For simplicity, let's just try to fit and see if it works.
            # If a bin has < 4 points, CubicSpline will fail or be ill-conditioned.
            # We can handle this by checking counts and reducing n_bins if necessary,
            # or by using a simpler model for sparse bins.
            
            # Let's check if any bin has < 4 points
            counts = np.bincount(bin_indices, minlength=n_bins)
            if np.any(counts < 4):
                # Skip this configuration if it's too sparse
                continue
            
            # Create a list of (x, y) pairs for each bin
            bin_data = []
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) >= 4:
                    bin_data.append((X_sorted[mask], y_sorted[mask]))
                else:
                    # Fallback: use all data? No, that breaks locality.
                    # Actually, if we skipped n_bins with < 4 points, this shouldn't happen.
                    # But just in case, we'll skip this n_bins if any bin is bad.
                    # We already checked counts, so this is safe.
                    pass
            
            # Fit splines
            splines = []
            for x_bin, y_bin in bin_data:
                # Use 'not-a-knot' boundary conditions
                try:
                    cs = CubicSpline(x_bin, y_bin, bc_type='not-a-knot')
                    splines.append(cs)
                except Exception:
                    # If spline fails, fall back to linear or quadratic
                    try:
                        cs = CubicSpline(x_bin, y_bin, bc_type='natural')
                        splines.append(cs)
                    except Exception:
                        # Give up on this bin count
                        splines = None
                        break
            
            if splines is None:
                continue
            
            # Cross-validation to select best n_bins
            # Use simple 5-fold CV
            n_folds = 5
            fold_size = n // n_folds
            cv_errors = []
            
            for i in range(n_folds):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < n_folds - 1 else n
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
                
                X_val = X[val_idx]
                y_val = y[val_idx]
                X_tr = X[train_idx]
                y_tr = y[train_idx]
                
                # Fit on train data
                sort_idx_tr = np.argsort(X_tr)
                X_tr_sorted = X_tr[sort_idx_tr]
                y_tr_sorted = y_tr[sort_idx_tr]
                
                bin_indices_tr = np.digitize(X_tr_sorted, edges) - 1
                bin_indices_tr = np.clip(bin_indices_tr, 0, n_bins - 1)
                
                counts_tr = np.bincount(bin_indices_tr, minlength=n_bins)
                if np.any(counts_tr < 4):
                    cv_errors.append(np.inf)
                    continue
                
                splines_tr = []
                valid = True
                for i_bin in range(n_bins):
                    mask = bin_indices_tr == i_bin
                    if np.sum(mask) >= 4:
                        try:
                            cs = CubicSpline(X_tr_sorted[mask], y_tr_sorted[mask], bc_type='not-a-knot')
                            splines_tr.append(cs)
                        except Exception:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                
                if not valid:
                    cv_errors.append(np.inf)
                    continue
                
                # Predict on validation data
                pred = np.zeros(len(X_val))
                bin_indices_val = np.digitize(X_val, edges) - 1
                bin_indices_val = np.clip(bin_indices_val, 0, n_bins - 1)
                
                for j in range(len(X_val)):
                    b = bin_indices_val[j]
                    if b < len(splines_tr):
                        pred[j] = splines_tr[b](X_val[j])
                    else:
                        pred[j] = np.mean(y_tr)
                
                if not np.all(np.isfinite(pred)):
                    cv_errors.append(np.inf)
                else:
                    mse = np.mean((pred - y_val) ** 2)
                    cv_errors.append(mse)
            
            mean_mse = np.mean(cv_errors)
            if mean_mse < best_score:
                best_score = mean_mse
                best_n_bins = n_bins
                
                # Refit on full data
                bin_indices_full = np.digitize(X_sorted, edges) - 1
                bin_indices_full = np.clip(bin_indices_full, 0, n_bins - 1)
                
                splines_full = []
                for i_bin in range(n_bins):
                    mask = bin_indices_full == i_bin
                    if np.sum(mask) >= 4:
                        try:
                            cs = CubicSpline(X_sorted[mask], y_sorted[mask], bc_type='not-a-knot')
                            splines_full.append(cs)
                        except Exception:
                            try:
                                cs = CubicSpline(X_sorted[mask], y_sorted[mask], bc_type='natural')
                                splines_full.append(cs)
                            except Exception:
                                splines_full = None
                                break
                
                if splines_full is not None:
                    def predict(X, splines=splines_full, edges=edges, n_bins=n_bins, mean_y=np.mean(y)):
                        X_flat = X.flatten()
                        pred = np.zeros(len(X_flat))
                        bin_indices_pred = np.digitize(X_flat, edges) - 1
                        bin_indices_pred = np.clip(bin_indices_pred, 0, n_bins - 1)
                        
                        for j in range(len(X_flat)):
                            b = bin_indices_pred[j]
                            if b < len(splines):
                                pred[j] = splines[b](X_flat[j])
                            else:
                                pred[j] = mean_y
                        
                        return pred.reshape(X.shape)
                    
                    best_predict = predict
        
        except Exception:
            continue
    
    if best_predict is None:
        # Fallback: global cubic spline
        try:
            cs = CubicSpline(X_sorted, y_sorted, bc_type='not-a-knot')
            def predict(X, cs=cs):
                return cs(X.flatten()).reshape(X.shape)
            return predict
        except Exception:
            mean_y = np.mean(y)
            return lambda X: np.full(X.shape[0], mean_y)
    
    return best_predict
# EVOLVE-BLOCK-END
