# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    n, d = X_train.shape
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_mean = np.mean(y_train)
        return lambda X: np.full(len(X), y_mean)

    # Strict deadline: 300s total for 10 equations -> 30s each.
    # Use 25s to be safe.
    deadline = start_time + 25.0

    def time_ok():
        return time.time() < deadline

    # --- Strategy 1: Kernel Ridge Regression with Sum-of-Sines kernel ---
    # This is the structural change requested.
    # We construct a feature matrix K where columns are sin/cos of various frequencies.
    # For each input dimension, we use a set of frequencies.
    # Since inputs are in [1, 3], [1, 2], [1, 5], [1, 10], frequencies should be adapted or just broad.
    # Let's use a fixed set of frequencies that works reasonably well for these ranges.
    # Ranges are small (width 1 to 9).
    # Frequencies: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30
    
    # To handle different scales, we can normalize X to [0, 1] or use raw X?
    # The problem statement says uniform input ranges.
    # If we use raw X, sin(20 * 10) is very high freq.
    # If we normalize, we need to know the range. We can estimate from data.
    
    # Let's try normalizing X to [0, 1] range based on min/max of training data.
    # This makes the frequency choice more universal.
    
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-6] = 1.0  # Avoid division by zero
    
    def normalize(X):
        return (X - X_min) / X_range

    def denormalize(X):
        return X * X_range + X_min

    # Frequencies for normalized inputs [0, 1].
    # We want to capture shapes that might oscillate a few times in the range.
    # Let's use a rich set of frequencies.
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0])
    
    # Build feature matrix for normalized X
    def gen_kernel_features(X_norm):
        cols = [np.ones(len(X_norm))]
        # Linear terms in normalized space
        for j in range(d):
            cols.append(X_norm[:, j])
            
        # Sine and Cosine terms
        for j in range(d):
            x = X_norm[:, j]
            for f in freqs:
                cols.append(np.sin(2 * np.pi * f * x))
                cols.append(np.cos(2 * np.pi * f * x))
        
        # Interactions?
        # Sum-of-sines kernel usually implies a product structure or just additive?
        # A standard kernel k(x, x') = sum_j sum_f c_{jf} (sin(2pi f x_j) sin(2pi f x'_j) + cos(...)) 
        # This is an additive kernel. It doesn't capture interactions like x1*x2.
        # For symbolic regression, interactions are common.
        # Let's add some low-order polynomial interactions in normalized space.
        
        # Degree 2 interactions
        for i in range(d):
            for j in range(i, d):
                cols.append(X_norm[:, i] * X_norm[:, j])
        
        # Maybe some cross terms with sines?
        # This can explode in size. d is max 3.
        # Number of features:
        # 1 + d (linear) + d * 2 * len(freqs) (trig) + d(d+1)/2 (poly2)
        # d=3: 1 + 3 + 3*2*15 + 6 = 4 + 90 + 6 = 100 features.
        # This is very manageable for 200 points.
        
        return np.column_stack(cols)

    X_train_norm = normalize(X_train)
    F_train = gen_kernel_features(X_train_norm)
    
    # Check for numerical stability
    if not np.all(np.isfinite(F_train)):
        # Fallback to simple linear if kernel features fail
        F_train = np.column_stack([np.ones(n), X_train_norm])
    
    # Ridge Regression
    # We need to pick alpha (regularization).
    # Let's try a few alphas and pick the best on training set (since we don't have validation split explicitly, and data is noiseless, training MSE is a good proxy for generalization if model isn't overfitting too much).
    # With 100 features and 200 points, overfitting is possible.
    # Ridge helps.
    
    best_mse = np.inf
    best_w = None
    best_alpha = 1e-5
    
    # Try a range of alphas
    for alpha in [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]:
        if not time_ok():
            break
        try:
            # Solve (F^T F + alpha I) w = F^T y
            # Using np.linalg.lstsq is also fine, but explicit ridge is faster for small N
            # Actually, np.linalg.lstsq with rcond is robust.
            # Let's use the normal equations with regularization for speed and control.
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A + alpha * np.eye(A.shape[0])
            
            # Check condition number or just solve
            try:
                w = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A_reg, b, rcond=None)[0]
            
            pred_train = F_train @ w
            mse = np.mean((pred_train - y_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w
                best_alpha = alpha
                
        except Exception:
            continue

    # If kernel method failed or produced bad result, fallback to other strategies
    if best_w is None or best_mse > 0.1 * y_var: # If MSE is more than 10% of variance, it's bad
         # Try simple polynomial features on raw X (as in previous code)
         def gen_poly_features(X):
             cols = [np.ones(len(X))]
             for j in range(d):
                 cols.append(X[:, j])
             for i in range(d):
                 for j in range(i, d):
                     cols.append(X[:, i] * X[:, j])
             if d == 1:
                 cols.append(X[:, 0]**3)
                 cols.append(X[:, 0]**4)
             return np.column_stack(cols)
         
         F_poly = gen_poly_features(X_train)
         if np.all(np.isfinite(F_poly)):
             try:
                 w_poly, *_ = np.linalg.lstsq(F_poly, y_train, rcond=None)
                 pred_poly = F_poly @ w_poly
                 mse_poly = np.mean((pred_poly - y_train) ** 2)
                 
                 if mse_poly < best_mse:
                     best_mse = mse_poly
                     best_w = w_poly
                     # We need to switch the predictor to use poly features
                     # Let's create a flag
                     use_poly = True
                     # Store F_poly generator
                     gen_func = gen_poly_features
                     X_ref = X_train # raw X
             except:
                 use_poly = False
         else:
             use_poly = False
    else:
        use_poly = False
        gen_func = gen_kernel_features
        X_ref = X_train_norm

    # Finalize predictor
    if best_w is None:
        # Ultimate fallback: mean
        mean_y = np.mean(y_train)
        def safe_predict(X):
            return np.full(len(X), mean_y)
        return safe_predict

    if use_poly:
        def predict(X):
            try:
                F = gen_func(X)
                if not np.all(np.isfinite(F)):
                    return np.full(len(X), np.mean(y_train))
                res = F @ best_w
                return np.nan_to_num(res, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
            except:
                return np.full(len(X), np.mean(y_train))
    else:
        def predict(X):
            try:
                X_norm = normalize(X)
                F = gen_func(X_norm)
                if not np.all(np.isfinite(F)):
                    return np.full(len(X), np.mean(y_train))
                res = F @ best_w
                return np.nan_to_num(res, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
            except:
                return np.full(len(X), np.mean(y_train))

    return predict
# EVOLVE-BLOCK-END
