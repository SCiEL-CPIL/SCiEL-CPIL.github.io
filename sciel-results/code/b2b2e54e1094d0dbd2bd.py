"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE_MARGIN = 15.0  # seconds before the 330s wall-clock limit to stop
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Since d is always 1 based on the task description, we treat x as 1D
    x = X.flatten()
    
    def time_left():
        return (start_time + 330.0 - DEADLINE_MARGIN) - time.time()
    
    # Helper to safely compute features
    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)
    
    # 1. Generate a diverse library of candidate basis functions
    # We want to cover polynomials, trigonometric, exponential, rational, and activation-like functions.
    
    def generate_basis(x_vals):
        x = np.asarray(x_vals, dtype=np.float64)
        n = len(x)
        cols = [np.ones(n)]
        
        # Polynomials
        for p in range(1, 15):
            cols.append(x ** p)
            
        # Trigonometric
        for k in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
            cols.append(np.sin(k * x) * x)
            cols.append(np.cos(k * x) * x)
            
        # Exponential
        for c in [0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]:
            cols.append(np.exp(c * x))
            cols.append(np.exp(c * x) * x)
            
        # Rational / Lorentzian
        cols.append(1.0 / (1.0 + x**2))
        cols.append(x / (1.0 + x**2))
        cols.append(1.0 / (1.0 + np.abs(x)))
        cols.append(np.tanh(x))
        cols.append(np.tanh(2 * x))
        cols.append(np.tanh(0.5 * x))
        cols.append(1.0 / (1.0 + np.exp(-x))) # sigmoid
        cols.append(1.0 / (1.0 + np.exp(-2*x)))
        
        # Logarithmic
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.log(np.abs(x) + 1e-12))
        cols.append(np.log1p(x) if np.min(x) > -1 else np.log1p(np.abs(x)))
        
        # Sqrt
        cols.append(np.sqrt(np.abs(x)))
        cols.append(np.sign(x) * np.sqrt(np.abs(x)))
        
        # Reciprocal
        cols.append(safe_div(1.0, x + 1e-12))
        cols.append(safe_div(x, x**2 + 1.0))
        
        # Combinations
        cols.append(np.sin(x) * np.exp(-0.1 * np.abs(x)))
        cols.append(np.cos(x) * np.exp(-0.1 * np.abs(x)))
        
        B = np.column_stack(cols)
        # Clean up NaN/Inf
        B = np.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
        return B

    B = generate_basis(x)
    
    # 2. Preprocessing for stability
    y_mean = np.mean(y)
    y_var = np.var(y)
    
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(len(X), c)
        
    # Standardize features for better conditioning in Ridge/OLS
    B_mean = np.mean(B, axis=0)
    B_std = np.std(B, axis=0)
    B_std[B_std < 1e-12] = 1.0
    B_stdized = (B - B_mean) / B_std
    
    # Add intercept
    X_design = np.column_stack([np.ones(n), B_stdized])
    
    # 3. Model Selection Strategy
    # We will use a sparse selection approach or Ridge with varying alphas.
    # Given 200 points and ~100 features, Lasso or ElasticNet would be ideal, but sklearn is not guaranteed.
    # We can implement a simple coordinate descent for Lasso or just use Ridge with a small alpha and then prune.
    # However, Ridge tends to keep all features. 
    # Let's try a few approaches:
    # A) Ridge with small alpha (e.g., 1e-4)
    # B) OLS on a subset of top correlated features
    # C) Forward selection (greedy) - might be slow but robust for 200 pts.
    
    # Let's try a hybrid:
    # 1. Compute correlation of each feature with y
    # 2. Select top K features
    # 3. Fit OLS on those K features + intercept
    
    y_centered = y - y_mean
    y_norm = np.linalg.norm(y_centered)
    
    # Correlations
    B_centered = B_stdized  # already centered and scaled
    # Note: B_stdized includes the original B columns centered. 
    # The first column of X_design is 1s, so B_stdized is just the non-constant features centered.
    
    # Calculate correlation for each feature in B_stdized
    # corr = (x - mean) dot (y - mean) / (norm(x) * norm(y))
    # Since B_stdized is standardized (mean 0, var 1), norm(x) is sqrt(n)
    # So corr = dot(B_stdized[:, j], y_centered) / (sqrt(n) * y_norm)
    
    sst = np.sum(y_centered**2)
    
    # Compute R^2 for individual features
    r2_scores = np.zeros(B_stdized.shape[1])
    for j in range(B_stdized.shape[1]):
        col = B_stdized[:, j]
        if np.linalg.norm(col) < 1e-12:
            continue
        # Projection
        # w = dot(col, y_centered) / dot(col, col)
        # pred = w * col
        # sse = sst - 2*w*dot(col, y_centered) + w^2*dot(col, col)
        # Actually simpler: R^2 = corr^2
        corr = np.dot(col, y_centered) / (np.linalg.norm(col) * y_norm)
        r2_scores[j] = corr**2
        
    # Sort features by R^2
    sorted_indices = np.argsort(r2_scores)[::-1]
    
    # Try subsets of top K features
    best_r2 = -np.inf
    best_K = 1
    best_w = None
    
    # We can try K from 1 to 50
    max_K = 50
    # But we must be careful with time.
    # Fitting OLS for K=50 on 200x51 matrix is very fast.
    
    for K in range(1, max_K + 1):
        if time_left() < 10.0:
            break
            
        # Select top K features
        top_idx = sorted_indices[:K]
        
        # Construct design matrix: Intercept + Top K features
        # We use the original B columns for these indices, or B_stdized?
        # Using B_stdized is fine if we store the mean/std for prediction.
        # Let's use B_stdized for numerical stability during fit.
        X_sub = np.column_stack([np.ones(n), B_stdized[:, top_idx]])
        
        try:
            # Solve least squares
            w, residuals, rank, s = np.linalg.lstsq(X_sub, y, rcond=None)
            y_pred = X_sub @ w
            sse = np.sum((y - y_pred)**2)
            r2 = 1 - sse / sst
            
            # We want to avoid overfitting. 
            # A simple penalty: if adding a feature doesn't improve R^2 significantly, stop?
            # Or just pick the K that gives best R^2? 
            # With 200 points, R^2 on training set is not a great estimator of test error for high K.
            # However, if the function is in the span of these features, R^2 will be ~1.0.
            # If it's not, R^2 will plateau.
            
            # Let's use a heuristic: Stop when R^2 improvement is < 1e-4
            if best_r2 > -np.inf:
                if r2 - best_r2 < 1e-5:
                    # We might have overfit or plateaued. 
                    # Keep the previous best if it was significantly better?
                    # Actually, let's just track the best R^2 seen so far, but be wary of overfitting.
                    # For symbolic regression, if the true function is in the basis, R^2 will be 1.
                    # If not, we want the simplest model that fits well.
                    # Let's continue to see if a larger K gives a much better fit.
                    pass
            
            if r2 > best_r2:
                best_r2 = r2
                best_K = K
                best_w = w
                best_top_idx = top_idx
                
        except np.linalg.LinAlgError:
            continue
            # If singular, skip this K
            
    # If we found a good fit (R^2 > 0.99), use it.
    if best_r2 > 0.99:
        # We need to reconstruct the prediction function using the selected features.
        # We stored best_w (coefficients for [1, B_stdized[:, top_idx]])
        # and best_top_idx (indices into B_stdized)
        # We also need B_mean and B_std to un-standardize? 
        # No, we can just compute B_stdized on the fly for prediction.
        
        # Save necessary state
        final_top_idx = best_top_idx
        final_w = best_w
        final_B_mean = B_mean
        final_B_std = B_std
        final_B_shape = B.shape[1] # total features generated
        
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            x_pred = X_pred.flatten()
            
            # Regenerate full basis to get the correct columns
            # This is inefficient but safe. 
            # Optimization: Only generate the specific columns needed?
            # Since generate_basis is deterministic, we can just generate all and pick.
            B_pred = generate_basis(x_pred)
            
            # Standardize
            B_pred_std = (B_pred - final_B_mean) / final_B_std
            
            # Select top features
            X_design_pred = np.column_stack([np.ones(len(x_pred)), B_pred_std[:, final_top_idx]])
            
            return X_design_pred @ final_w

        return predict

    # If no single subset gave a great R^2, try Ridge Regression on ALL features
    # This acts as a smoother and might generalize better if the function is complex but not in the span.
    
    best_ridge_r2 = -np.inf
    best_ridge_w = None
    best_alpha = 1e-4
    
    # Try a few alphas
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for alpha in alphas:
        if time_left() < 5.0:
            break
            
        # Ridge: (X^T X + alpha I) w = X^T y
        # Don't regularize intercept
        A = X_design.T @ X_design
        b = X_design.T @ y
        
        A_reg = A.copy()
        A_reg[1:, 1:] += alpha * np.eye(X_design.shape[1] - 1)
        
        try:
            w_ridge = np.linalg.solve(A_reg, b)
            y_pred = X_design @ w_ridge
            sse = np.sum((y - y_pred)**2)
            r2 = 1 - sse / sst
            
            if r2 > best_ridge_r2:
                best_ridge_r2 = r2
                best_ridge_w = w_ridge
                best_alpha = alpha
        except:
            continue

    # Compare the best subset OLS with the best Ridge
    if best_ridge_r2 > best_r2:
        # Use Ridge
        final_B_mean = B_mean
        final_B_std = B_std
        final_w_ridge = best_ridge_w
        
        def predict_ridge(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            x_pred = X_pred.flatten()
            B_pred = generate_basis(x_pred)
            B_pred_std = (B_pred - final_B_mean) / final_B_std
            X_design_pred = np.column_stack([np.ones(len(x_pred)), B_pred_std])
            return X_design_pred @ final_w_ridge

        return predict_ridge
    else:
        # Use the best subset OLS found earlier
        # If best_r2 was -inf (no fit found), we should fall back to something simple
        if best_r2 == -np.inf:
             # Fallback: Linear fit
             X_lin = np.column_stack([np.ones(n), x])
             w_lin, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
             def predict_lin(X_pred):
                 X_pred = np.asarray(X_pred, dtype=np.float64)
                 x_pred = X_pred.flatten()
                 return np.column_stack([np.ones(len(x_pred)), x_pred]) @ w_lin
             return predict_lin
        else:
             # Use the subset OLS logic defined above
             final_top_idx = best_top_idx
             final_w = best_w
             final_B_mean = B_mean
             final_B_std = B_std
             
             def predict_subset(X_pred):
                 X_pred = np.asarray(X_pred, dtype=np.float64)
                 x_pred = X_pred.flatten()
                 B_pred = generate_basis(x_pred)
                 B_pred_std = (B_pred - final_B_mean) / final_B_std
                 X_design_pred = np.column_stack([np.ones(len(x_pred)), B_pred_std[:, final_top_idx]])
                 return X_design_pred @ final_w
                 
             return predict_subset

# EVOLVE-BLOCK-END
