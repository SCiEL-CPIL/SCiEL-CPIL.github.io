import numpy as np
import time
from itertools import product

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Handle constant x
    if np.std(x) < 1e-12:
        return np.array([np.mean(y)] + [0.0]*5)
    
    # 1. Generate a fixed grid of points across the range of x
    # We want enough points to resolve the function but not too many to be slow
    x_min, x_max = np.min(x), np.max(x)
    # Add a small margin
    margin = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - margin, x_max + margin, 200)
    
    # 2. LOESS Smoothing Strategy
    # We will evaluate LOESS with different bandwidths and local degrees on the grid.
    # Then we will fit a global degree-5 polynomial to the smoothed values.
    # We use Leave-One-Out (LOO) Cross-Validation on the original training data to select the best LOESS parameters.
    
    # Precompute distances? No, for LOO on n points and grid evaluation, we need flexibility.
    # However, n is likely moderate (1000-10000). 
    # LOO MSE for LOESS can be computed efficiently using hat matrix or direct calculation.
    # Given time constraints, we will sample a subset of training points for CV if n is large, 
    # or use efficient LOO formula.
    
    # Let's try a few bandwidths (fraction of range) and local degrees (1, 2)
    # Bandwidths: 0.1, 0.2, 0.3, 0.5, 0.7 of the range
    # Local Degrees: 1, 2
    
    bandwidths = [0.1, 0.2, 0.3, 0.5, 0.7]
    local_degrees = [1, 2]
    
    best_cv_mse = np.inf
    best_bandwidth = 0.3
    best_local_degree = 1
    
    # Helper function to compute LOO MSE for a given LOESS configuration
    def compute_loo_mse(x, y, bandwidth_frac, local_deg):
        if time.time() > deadline - 10:
            return np.inf
            
        h = bandwidth_frac * (np.max(x) - np.min(x))
        if h <= 0:
            return np.inf
            
        n_pts = len(x)
        # To speed up CV, if n is very large, subsample
        if n_pts > 2000:
            idx = np.random.choice(n_pts, 2000, replace=False)
            x_cv = x[idx]
            y_cv = y[idx]
            n_cv = len(x_cv)
        else:
            x_cv = x
            y_cv = y
            n_cv = n_pts
            
        loo_errors = np.zeros(n_cv)
        
        # We can optimize this, but for n=2000 and few configurations, direct is okay?
        # 2000 * 2000 = 4M operations per config. 10 configs = 40M. Might be tight in Python.
        # Let's use a vectorized approach for weights if possible, or just loop with Numpy ops.
        
        for i in range(n_cv):
            if time.time() > deadline - 5:
                return np.inf
            # Distances
            dists = np.abs(x_cv - x_cv[i])
            # Tricube weights (standard LOESS)
            u = dists / h
            # Mask points within bandwidth
            mask = u <= 1.0
            if np.sum(mask) < local_deg + 1:
                # Not enough points, use all or skip? Standard LOESS uses all if < p+1
                mask = np.ones(n_cv, dtype=bool)
                u = dists / (h * 1.5) # Fudge factor to include more
                mask = u <= 1.0
                if np.sum(mask) < local_deg + 1:
                    loo_errors[i] = 0.0 # Can't compute, assume 0 error? Or large?
                    continue
            
            w = (1.0 - u[mask]**6)**3
            w[x_cv == x_cv[i]] = 1.0 # Ensure self weight is 1 if included? No, LOO excludes self.
            
            # Exclude the point i
            w_excl = np.ones(np.sum(mask))
            # Find index of i in the masked array
            # This is tricky. Let's construct the masked arrays and remove i.
            x_masked = x_cv[mask]
            y_masked = y_cv[mask]
            
            # Find index of x_cv[i] in x_masked
            # Since x might have duplicates, this is ambiguous. 
            # Better approach: compute weights for all, set weight of i to 0.
            
            # Let's restart this helper with a cleaner vectorized LOO logic
            # Actually, standard LOO for LOESS is computationally expensive in pure Python loops.
            # Alternative: Use the fact that we just need to select a "good" smoothness.
            # We can approximate LOO by holding out a random 20% of the data (K-fold style) 
            # which is much faster to implement correctly in vectorized NumPy.
            
            pass # Placeholder, see below for actual implementation

    # Let's implement a simpler 5-Fold CV to be safe on time and correctness
    def compute_cv_mse(x, y, bandwidth_frac, local_deg, n_folds=5):
        if time.time() > deadline - 10:
            return np.inf
        h = bandwidth_frac * (np.max(x) - np.min(x))
        if h <= 0:
            return np.inf
            
        n_pts = len(x)
        # Shuffle indices
        indices = np.arange(n_pts)
        np.random.shuffle(indices)
        fold_size = n_pts // n_folds
        
        mse_sum = 0.0
        for k in range(n_folds):
            if time.time() > deadline - 5:
                return np.inf
            val_start = k * fold_size
            val_end = (k + 1) * fold_size if k < n_folds - 1 else n_pts
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]
            
            # Fit LOESS on train, predict on val
            # For each val point, find local neighborhood in train
            # This is O(n_val * n_train). 
            # n_val ~ n/5. n_train ~ 4n/5.
            # Total per fold ~ 4/25 n^2. Total ~ 4/5 n^2.
            # For n=2000, n^2=4M. 4/5 * 4M = 3.2M ops per fold. 5 folds = 16M.
            # 10 configs = 160M ops. Might be too slow in Python.
            
            # Optimization: Vectorize the prediction step for the validation set.
            # Distances matrix: (n_val, n_train)
            # This can be memory heavy if n is large.
            # Let's subsample training data for CV if n > 1000.
            
            if n_pts > 1000:
                # Subsample training data for speed in CV selection
                subsample_size = 1000
                if len(train_idx) > subsample_size:
                    sub_idx = np.random.choice(len(train_idx), subsample_size, replace=False)
                    x_train_sub = x_train[sub_idx]
                    y_train_sub = y_train[sub_idx]
                else:
                    x_train_sub = x_train
                    y_train_sub = y_train
            else:
                x_train_sub = x_train
                y_train_sub = y_train
                
            n_tr = len(x_train_sub)
            n_va = len(x_val)
            
            # Compute distances
            # To avoid memory blowup, process in chunks if n_va is large?
            # n_va is ~400. n_tr is ~800. 400*800 = 320k floats. Fine.
            dists = np.abs(x_val[:, None] - x_train_sub[None, :])
            u = dists / h
            # Tricube weights
            w = np.where(u <= 1.0, (1.0 - u**6)**3, 0.0)
            
            # For each validation point, fit local poly
            preds = np.zeros(n_va)
            for i in range(n_va):
                if time.time() > deadline - 2:
                    return np.inf
                wi = w[i]
                # Find non-zero weights
                mask = wi > 0
                if np.sum(mask) < local_deg + 1:
                    # Use all points with distance < h*2 or just nearest neighbors
                    # Find nearest neighbors
                    dists_i = dists[i]
                    k_nearest = min(local_deg + 1, n_tr)
                    nn_idx = np.argsort(dists_i)[:k_nearest]
                    x_loc = x_train_sub[nn_idx]
                    y_loc = y_train_sub[nn_idx]
                    # Compute weights based on max dist in this set
                    d_max = np.max(dists_i[nn_idx])
                    if d_max == 0:
                        preds[i] = np.mean(y_loc)
                        continue
                    u_loc = dists_i[nn_idx] / d_max
                    w_loc = (1.0 - u_loc**6)**3
                else:
                    x_loc = x_train_sub[mask]
                    y_loc = y_train_sub[mask]
                    w_loc = wi[mask]
                
                # Weighted Least Squares
                X_loc = np.vander(x_loc - x_val[i], N=local_deg+1, increasing=True)
                W_mat = np.diag(w_loc)
                # Solve (X^T W X) beta = X^T W y
                XtW = X_loc.T * w_loc[:, None]
                try:
                    beta = np.linalg.lstsq(XtW, w_loc * y_loc, rcond=None)[0]
                    preds[i] = beta[0] # Value at x_val[i] is beta[0] since centered at 0
                except:
                    preds[i] = np.average(y_loc, weights=w_loc)
            
            mse_sum += np.mean((preds - y_val)**2)
        
        return mse_sum / n_folds

    # Run CV for candidate configurations
    configs = []
    for bw in bandwidths:
        for ld in local_degrees:
            configs.append((bw, ld))
            
    # Shuffle configs to ensure we try diverse ones if time runs out
    np.random.shuffle(configs)
    
    for bw, ld in configs:
        if time.time() > deadline - 20:
            break
        cv_mse = compute_cv_mse(x, y, bw, ld)
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_bandwidth = bw
            best_local_degree = ld
            
    # 3. Generate Smoothed Data using the Best LOESS Parameters
    # We evaluate the LOESS function on the x_grid
    h = best_bandwidth * (np.max(x) - np.min(x))
    y_smooth = np.zeros(len(x_grid))
    
    for i, xg in enumerate(x_grid):
        if time.time() > deadline - 10:
            # Fallback to simple mean if time runs out
            y_smooth[i:] = np.mean(y)
            break
            
        dists = np.abs(x - xg)
        u = dists / h
        mask = u <= 1.0
        if np.sum(mask) < best_local_degree + 1:
            # Use nearest neighbors
            k_nn = min(best_local_degree + 1, n)
            nn_idx = np.argsort(dists)[:k_nn]
            x_loc = x[nn_idx]
            y_loc = y[nn_idx]
            d_max = np.max(dists[nn_idx])
            if d_max == 0:
                y_smooth[i] = np.mean(y_loc)
                continue
            u_loc = dists[nn_idx] / d_max
            w_loc = (1.0 - u_loc**6)**3
        else:
            x_loc = x[mask]
            y_loc = y[mask]
            w_loc = (1.0 - u[mask]**6)**3
            
        X_loc = np.vander(x_loc - xg, N=best_local_degree+1, increasing=True)
        XtW = X_loc.T * w_loc[:, None]
        try:
            beta = np.linalg.lstsq(XtW, w_loc * y_loc, rcond=None)[0]
            y_smooth[i] = beta[0]
        except:
            y_smooth[i] = np.average(y_loc, weights=w_loc)

    # 4. Fit Global Degree-5 Polynomial to the Smoothed Data
    # We want to fit a polynomial in the ORIGINAL basis (x**0 ... x**5)
    # To ensure stability, we might still want to center/scale for the fit, 
    # but the output must be in original basis.
    
    # Fit on the grid
    X_grid = np.vander(x_grid, N=6, increasing=True)
    
    # Weight the fit? The smoothed points are roughly equally spaced.
    # We can just do least squares.
    try:
        coeff, residuals, rank, s = np.linalg.lstsq(X_grid, y_smooth, rcond=None)
        if rank < 6:
             # If rank deficient, try with centered basis?
             # Or just use the result if it's finite
             if not np.all(np.isfinite(coeff)):
                 # Fallback to global fit on original data
                 X_orig = np.vander(x, N=6, increasing=True)
                 coeff, _, _, _ = np.linalg.lstsq(X_orig, y, rcond=None)
    except:
        # Fallback to global fit on original data
        X_orig = np.vander(x, N=6, increasing=True)
        try:
            coeff, _, _, _ = np.linalg.lstsq(X_orig, y, rcond=None)
        except:
            coeff = np.zeros(6)
            coeff[0] = np.mean(y)

    # Ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        
    return coeff
