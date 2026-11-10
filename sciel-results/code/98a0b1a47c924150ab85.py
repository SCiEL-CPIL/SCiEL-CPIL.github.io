"""Non-parametric local linear regression closure approximated by a global polynomial."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _local_linear_predict(x_train, y_train, x_test, bandwidth):
    """
    Compute Nadaraya-Watson local linear regression predictions.
    Uses Gaussian kernel K(u) = exp(-0.5 * u^2).
    """
    n_train = len(x_train)
    n_test = len(x_test)
    y_pred = np.zeros(n_test)
    
    # We process in chunks if necessary, but for this problem size (likely < 100k points),
    # vectorized operations over test points against training points might be memory heavy 
    # if done all at once (n_test x n_train). 
    # However, local linear regression usually benefits from restricting to neighbors.
    # Given the smoothness, we can use a simple approach:
    # For each test point, find neighbors within ~ 3*bandwidth.
    # But finding neighbors efficiently requires a sorted array or KD-tree.
    # Since we only have 1D data, sorting is efficient.
    
    # Sort training data
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    for i in range(n_test):
        xt = x_test[i]
        
        # Estimate number of neighbors needed. 
        # In 1D Gaussian kernel, effective support is ~ 3*sigma.
        # We want to include enough points to have a stable local linear fit.
        # Let's find indices in x_sorted close to xt.
        
        # Use searchsorted to find the window
        # Heuristic: window size depends on bandwidth and data density.
        # Let's take a fixed number of nearest neighbors for stability, or a distance cutoff.
        # Distance cutoff is more principled for bandwidth selection.
        # Cutoff: 3 * bandwidth.
        
        lo_idx = np.searchsorted(x_sorted, xt - 3 * bandwidth, side='right')
        hi_idx = np.searchsorted(x_sorted, xt + 3 * bandwidth, side='left')
        
        # Ensure we have at least 2 points for local linear regression (slope + intercept)
        # If too few, expand window or fall back to global mean?
        # Better: enforce a minimum number of neighbors, say 10.
        min_neighbors = 10
        if hi_idx - lo_idx < min_neighbors:
            # Expand window
            mid = (lo_idx + hi_idx) // 2
            half_width = (min_neighbors - (hi_idx - lo_idx)) // 2 + 1
            lo_idx = max(0, mid - half_width)
            hi_idx = min(n_train, mid + half_width)
            
        if hi_idx <= lo_idx:
            # No neighbors found (should not happen if data covers range)
            y_pred[i] = np.mean(y_train)
            continue
            
        x_local = x_sorted[lo_idx:hi_idx]
        y_local = y_sorted[lo_idx:hi_idx]
        
        # Weights
        u = (x_local - xt) / bandwidth
        w = np.exp(-0.5 * u * u)
        
        # Local linear regression:
        # Minimize sum w_i * (y_i - (a + b*(x_i - xt)))^2
        # Let z_i = x_i - xt
        # Matrix form: [1, z] * [a, b]^T = y
        # Weighted least squares: (V^T W V)^-1 V^T W y
        z = x_local - xt
        V = np.column_stack([np.ones_like(z), z])
        
        # Weighted normal equations
        # A = V^T (W * V)
        # b = V^T (W * y)
        # To avoid explicit W matrix (n x n), multiply V and y by sqrt(w)
        sqrt_w = np.sqrt(w)
        V_w = V * sqrt_w[:, np.newaxis]
        y_w = y_local * sqrt_w
        
        try:
            # Solve 2x2 system
            # V_w^T V_w is 2x2
            A = V_w.T @ V_w
            b_vec = V_w.T @ y_w
            
            # Add tiny ridge for stability if singular
            A += 1e-12 * np.eye(2)
            coeffs = np.linalg.solve(A, b_vec)
            
            # Prediction at xt (where z=0) is just the intercept 'a'
            y_pred[i] = coeffs[0]
        except np.linalg.LinAlgError:
            # Fallback to weighted mean
            y_pred[i] = np.sum(w * y_local) / np.sum(w)
            
    return y_pred


def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 10:
        return np.array([float(np.mean(y)) if n else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    y_var = float(np.var(y))
    if y_var < 1e-15:
        y_var = 1e-15

    # 1. Select bandwidth via Cross-Validation
    # We need to test a range of bandwidths.
    # Standard deviation of x is a good scale.
    x_std = float(np.std(x))
    if x_std < 1e-12:
        x_std = 1.0

    # Bandwidths relative to std
    # Local linear regression is more robust to bandwidth choice than NW, but still sensitive.
    # Typical range: 0.1 * std to 3.0 * std
    bw_candidates = x_std * np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    
    # If data is small, CV is expensive. If large, subsample for CV.
    n_cv = min(5000, n)
    rng = np.random.RandomState(42)
    cv_idx = rng.choice(n, n_cv, replace=False)
    x_cv = x[cv_idx]
    y_cv = y[cv_idx]
    
    best_bw = bw_candidates[0]
    best_cv_mse = np.inf
    
    # K-fold CV on the subsample for bandwidth selection
    # To save time, use 3-fold on the subsample
    K = 3
    fold_size = n_cv // K
    indices = np.arange(n_cv)
    
    for bw in bw_candidates:
        if time.time() > deadline - 10.0: # Leave time for final fit
            break
        mse_sum = 0.0
        cnt = 0
        for k in range(K):
            lo = k * fold_size
            hi = lo + fold_size
            if k == K - 1:
                hi = n_cv
            
            val_idx = indices[lo:hi]
            train_idx = np.concatenate([indices[:lo], indices[hi:]])
            
            x_train_cv = x_cv[train_idx]
            y_train_cv = y_cv[train_idx]
            x_val_cv = x_cv[val_idx]
            y_val_cv = y_cv[val_idx]
            
            # Predict
            y_pred = _local_linear_predict(x_train_cv, y_train_cv, x_val_cv, bw)
            
            mse = np.mean((y_pred - y_val_cv) ** 2)
            mse_sum += mse
            cnt += 1
            
        cv_mse = mse_sum / max(cnt, 1)
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_bw = bw

    # 2. Generate "true" closure values using the best bandwidth on ALL training data
    # This creates a non-parametric estimate of the function f(x).
    # We use the training data to predict the training data (1-fold, effectively in-sample but smoothed).
    # To avoid overfitting the polynomial to the noise in the training y, we might want to 
    # use a leave-one-out or just accept the smoothing effect. 
    # Local linear regression has a built-in bias that reduces overfitting.
    # Let's compute the smoothed y values.
    
    # Optimization: _local_linear_predict is O(N^2) worst case if no pruning, but with sorting and searchsorted it's O(N log N + N * K_neighbors).
    # However, the Python loop in _local_linear_predict is slow for large N.
    # If N is large (e.g., 100k), this might timeout.
    # Let's check N.
    if n > 20000:
        # Subsample for the "oracle" generation to save time, then fit poly on subsample?
        # No, we need the poly to fit the global trend. Subsampling for the oracle is fine if random.
        oracle_idx = rng.choice(n, 10000, replace=False)
        x_oracle = x[oracle_idx]
        y_oracle = y[oracle_idx]
    else:
        x_oracle = x
        y_oracle = y

    # Generate smoothed targets
    # Note: predicting training points with training data is "in-sample".
    # For a smoother, this is often acceptable as the kernel width introduces bias.
    # A better approach: Split data, fit on half, predict other half, combine? 
    # Or just use the smoothed version. Given the task is to fit a polynomial to the "unresolved forcing",
    # and the forcing is a smooth function of x, the smoothed y is a good proxy for the true function.
    
    y_smoothed = _local_linear_predict(x_oracle, y_oracle, x_oracle, best_bw)
    
    # 3. Fit a degree-5 polynomial to the smoothed data
    # Use centered and scaled x for numerical stability
    mu = float(np.mean(x_oracle))
    sigma = float(np.std(x_oracle))
    if sigma < 1e-12:
        sigma = 1.0
    
    x_scaled = (x_oracle - mu) / sigma
    
    # Design matrix for degree 5
    V = np.column_stack([x_scaled ** k for k in range(6)])
    
    # Ridge regression for stability
    alpha = 1e-6
    A = V.T @ V + alpha * np.eye(6)
    b = V.T @ y_smoothed
    
    try:
        c_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c_scaled = np.linalg.lstsq(V, y_smoothed, rcond=None)[0]
    
    # 4. Convert back to original basis
    # p(x) = sum c_i * ((x-mu)/sigma)^i
    # We need coeffs C_k such that sum C_k x^k = p(x)
    C = np.zeros(6)
    for i in range(6):
        ci = c_scaled[i]
        if ci == 0.0:
            continue
        inv_si = (1.0 / sigma) ** i
        for k in range(i + 1):
            # binom(i, k)
            if k == 0 or k == i:
                Ck = 1.0
            else:
                mm = min(k, i - k)
                num = 1
                den = 1
                for t in range(1, mm + 1):
                    num *= (i - mm + t)
                    den *= t
                Ck = num / den
            C[k] += ci * Ck * ((-mu) ** (i - k)) * inv_si
            
    if not np.all(np.isfinite(C)):
        # Fallback to mean
        C = np.array([float(np.mean(y)), 0, 0, 0, 0, 0], dtype=np.float64)

    return C
# EVOLVE-BLOCK-END
