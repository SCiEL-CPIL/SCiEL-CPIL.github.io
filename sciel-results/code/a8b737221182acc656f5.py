"""Symbolic regression using a rich basis function expansion and forward feature selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def fit(X_train, y_train):
    start_time = time.time()
    timeout = 320.0  # Leave 10s margin for the 330s limit
    
    # 1. Generate a rich set of candidate basis functions
    # We include:
    # - Constants
    # - Linear terms x_i
    # - Non-linear transforms of x_i: sin, cos, exp, log, tanh, sqrt, x^2, x^3
    # - Interactions: x_i * x_j, x_i * sin(x_j), etc. (limited to keep dimension manageable)
    
    d = X_train.shape[1]
    n_samples = X_train.shape[0]
    
    # Helper to check if a function is safe (no NaNs/Infs) on the training data
    def is_safe(func):
        try:
            vals = func(X_train)
            if not np.all(np.isfinite(vals)):
                return False
            # If variance is zero, it's a constant, which is fine but redundant if we have ones
            return True
        except Exception:
            return False

    # Define base transformations for each dimension
    # We apply these to each input dimension x_j
    transforms = {
        'x': lambda x: x,
        'sin': lambda x: np.sin(x),
        'cos': lambda x: np.cos(x),
        'exp': lambda x: np.exp(x),
        'log': lambda x: np.log(x),
        'tanh': lambda x: np.tanh(x),
        'x2': lambda x: x ** 2,
        'x3': lambda x: x ** 3,
        'sqrt': lambda x: np.sqrt(x),
        'abs': lambda x: np.abs(x),
    }
    
    # To handle log(sqrt) safely, we clip inputs
    def safe_log(x):
        return np.log(np.clip(x, 1e-8, None))
    def safe_sqrt(x):
        return np.sqrt(np.clip(x, 0, None))
        
    transforms['log'] = safe_log
    transforms['sqrt'] = safe_sqrt

    # Generate candidate features
    # Start with constant 1
    candidates = [np.ones(n_samples)]
    candidate_names = ['const']
    
    # Add univariate transforms
    for j in range(d):
        for t_name, t_func in transforms.items():
            try:
                vals = t_func(X_train[:, j])
                if np.all(np.isfinite(vals)):
                    candidates.append(vals)
                    candidate_names.append(f"{t_name}(x{j})")
            except Exception:
                pass

    # Add simple pairwise interactions
    # x_i * x_j
    for i in range(d):
        for j in range(i+1, d):
            vals = X_train[:, i] * X_train[:, j]
            if np.all(np.isfinite(vals)):
                candidates.append(vals)
                candidate_names.append(f"x{i}*x{j}")
                
    # Add mixed transforms like sin(x_i)*x_j or x_i*log(x_j)
    # Limit combinations to avoid explosion
    simple_transforms = ['x', 'sin', 'cos', 'exp', 'log']
    for i in range(d):
        for j in range(d):
            if i == j: continue
            for t1 in simple_transforms:
                for t2 in simple_transforms:
                    try:
                        v1 = transforms[t1](X_train[:, i])
                        v2 = transforms[t2](X_train[:, j])
                        vals = v1 * v2
                        if np.all(np.isfinite(vals)):
                            # Avoid duplicates if possible, but for now just add
                            candidates.append(vals)
                            candidate_names.append(f"{t1}(x{i})*{t2}(x{j})")
                    except Exception:
                        pass

    # Combine into a matrix
    # Filter out any columns that are all zeros or constant (redundant)
    X_basis = np.column_stack(candidates)
    
    # Remove constant columns (except the first one) and zero-variance columns to reduce noise
    # We keep the constant column (index 0)
    # For others, check variance
    keep_mask = np.zeros(X_basis.shape[1], dtype=bool)
    keep_mask[0] = True
    for i in range(1, X_basis.shape[1]):
        if np.std(X_basis[:, i]) > 1e-8:
            keep_mask[i] = True
    X_basis = X_basis[:, keep_mask]
    candidate_names = [n for n, k in zip(candidate_names, keep_mask) if k]
    
    # If basis is too large, we might need to subsample or limit, but with d<=3, it's likely < 200 features
    # Let's say we have M features
    
    # 2. Forward Feature Selection
    # We want to find the subset of features that minimizes the normalized MSE on training data.
    # Since data is noiseless, the true model will have ~0 MSE.
    # We use Ridge Regression for stability.
    
    n_features = X_basis.shape[1]
    y = y_train
    
    # Pre-compute Gram matrix? No, we do incremental least squares or re-solve small systems.
    # Since n_features is likely < 100, solving full system is fast.
    
    best_mse = np.inf
    best_indices = [0]  # Always include constant
    best_w = np.zeros(1)
    
    # Greedy forward selection
    # We will add features one by one if they significantly reduce MSE
    
    current_indices = [0]
    current_mse = np.inf
    
    # Initial fit with just constant
    def fit_ridge(indices, alpha=1e-6):
        X_sub = X_basis[:, indices]
        # Add small ridge for stability
        try:
            w, residuals, rank, s = np.linalg.lstsq(X_sub, y, rcond=None)
            y_pred = X_sub @ w
            mse = np.mean((y - y_pred) ** 2)
            return mse, w
        except Exception:
            return np.inf, None

    mse, w = fit_ridge(current_indices)
    current_mse = mse
    if mse < best_mse:
        best_mse = mse
        best_indices = current_indices.copy()
        best_w = w
        
    # Iterate adding features
    # To speed up, we can pre-compute correlations or just loop.
    # With ~50-100 features, 100 iterations * 100 lstsq calls is fine within 300s.
    
    remaining_indices = [i for i in range(n_features) if i != 0]
    
    while True:
        # Check time
        if time.time() - start_time > timeout:
            break
            
        best_new_mse = current_mse
        best_new_idx = None
        
        # Try adding each remaining feature
        for idx in remaining_indices:
            trial_indices = current_indices + [idx]
            mse_trial, _ = fit_ridge(trial_indices)
            
            if mse_trial < best_new_mse - 1e-10:  # Significant improvement
                best_new_mse = mse_trial
                best_new_idx = idx
        
        if best_new_idx is None:
            break
            
        # Add the best feature
        current_indices.append(best_new_idx)
        remaining_indices.remove(best_new_idx)
        current_mse = best_new_mse
        
        # Update best
        if current_mse < best_mse:
            best_mse = current_mse
            best_indices = current_indices.copy()
            _, best_w = fit_ridge(best_indices)
            
        # Early stopping if MSE is effectively zero
        if current_mse < 1e-12:
            break

    # 3. Construct the predict function
    # We use the best found model
    # If best_indices is empty or invalid, fallback to constant mean
    if not best_indices:
        mean_y = np.mean(y)
        return lambda X: np.full(X.shape[0], mean_y)
        
    # Extract the relevant columns from X_basis for the best model
    # But wait, X_basis is built from X_train. We need to build the SAME basis from X_test.
    # This requires re-generating the features for X_test using the same logic.
    
    # To do this efficiently, we store the "recipe" of the best features.
    # However, my candidate generation above is dynamic. 
    # A simpler approach: Store the indices into the *generated* basis for train, 
    # and regenerate the basis for test in the exact same order.
    
    # Let's refactor to store the generation logic or just re-run generation for X.
    # Since the generation is deterministic given X, we can just re-run it for X_test.
    
    def build_basis(X):
        n = X.shape[0]
        d = X.shape[1]
        cands = [np.ones(n)]
        # Univariate
        for j in range(d):
            for t_name, t_func in transforms.items():
                try:
                    vals = t_func(X[:, j])
                    if np.all(np.isfinite(vals)):
                        cands.append(vals)
                except Exception:
                    pass
        # Pairwise
        for i in range(d):
            for j in range(i+1, d):
                vals = X[:, i] * X[:, j]
                if np.all(np.isfinite(vals)):
                    cands.append(vals)
        # Mixed
        for i in range(d):
            for j in range(d):
                if i == j: continue
                for t1 in simple_transforms:
                    for t2 in simple_transforms:
                        try:
                            v1 = transforms[t1](X[:, i])
                            v2 = transforms[t2](X[:, j])
                            vals = v1 * v2
                            if np.all(np.isfinite(vals)):
                                cands.append(vals)
                        except Exception:
                            pass
        X_b = np.column_stack(cands)
        # Apply same filtering: keep const, keep non-constant
        # Note: The filtering depends on variance in X_train. 
        # For X_test, we should ideally use the same set of columns.
        # To be safe, we assume the structure of columns is identical.
        # The "keep_mask" was based on variance in X_train. 
        # If a column was constant in X_train, it's likely constant in X_test (uniform dist).
        # We will just return the full matrix and select columns by index from the *unfiltered* list?
        # No, the indices `best_indices` refer to the *filtered* `X_basis`.
        # This is a mismatch risk.
        
        # Better approach: Don't filter during index selection. 
        # Select indices from the RAW candidate list.
        # Let's restart the logic to use raw indices.
        
        return X_b, cands # Return raw and names for debugging if needed

    # Re-do: The previous `best_indices` are indices into `X_basis` (filtered).
    # To make this robust, I will re-implement the selection using the RAW basis and 
    # then map back? Or just ensure the filtering is deterministic and identical for train/test.
    # Variance filtering is data-dependent.
    
    # SAFE APPROACH: 
    # 1. Generate Raw Basis for Train.
    # 2. Perform selection on Raw Basis.
    # 3. Predict by generating Raw Basis for Test and picking the same column indices.
    
    # Let's discard the previous `best_indices` and redo selection on RAW basis.
    
    # Re-generate Raw Basis for Train
    def generate_raw_basis(X):
        n = X.shape[0]
        d = X.shape[1]
        cands = [np.ones(n)]
        for j in range(d):
            for t_name, t_func in transforms.items():
                try:
                    vals = t_func(X[:, j])
                    if np.all(np.isfinite(vals)):
                        cands.append(vals)
                except Exception:
                    pass
        for i in range(d):
            for j in range(i+1, d):
                vals = X[:, i] * X[:, j]
                if np.all(np.isfinite(vals)):
                    cands.append(vals)
        for i in range(d):
            for j in range(d):
                if i == j: continue
                for t1 in simple_transforms:
                    for t2 in simple_transforms:
                        try:
                            v1 = transforms[t1](X[:, i])
                            v2 = transforms[t2](X[:, j])
                            vals = v1 * v2
                            if np.all(np.isfinite(vals)):
                                cands.append(vals)
                        except Exception:
                            pass
        return np.column_stack(cands)

    X_raw_train = generate_raw_basis(X_train)
    n_raw = X_raw_train.shape[1]
    
    # Selection on Raw Basis
    best_mse_raw = np.inf
    best_indices_raw = [0]
    best_w_raw = np.zeros(1)
    
    current_indices_raw = [0]
    current_mse_raw = np.inf
    
    def fit_ridge_raw(indices, alpha=1e-6):
        X_sub = X_raw_train[:, indices]
        try:
            w, residuals, rank, s = np.linalg.lstsq(X_sub, y, rcond=None)
            y_pred = X_sub @ w
            mse = np.mean((y - y_pred) ** 2)
            return mse, w
        except Exception:
            return np.inf, None

    mse, w = fit_ridge_raw(current_indices_raw)
    current_mse_raw = mse
    if mse < best_mse_raw:
        best_mse_raw = mse
        best_indices_raw = current_indices_raw.copy()
        best_w_raw = w
        
    remaining_raw = [i for i in range(n_raw) if i != 0]
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        best_new_mse = current_mse_raw
        best_new_idx = None
        
        for idx in remaining_raw:
            trial_indices = current_indices_raw + [idx]
            mse_trial, _ = fit_ridge_raw(trial_indices)
            
            if mse_trial < best_new_mse - 1e-10:
                best_new_mse = mse_trial
                best_new_idx = idx
        
        if best_new_idx is None:
            break
            
        current_indices_raw.append(best_new_idx)
        remaining_raw.remove(best_new_idx)
        current_mse_raw = best_new_mse
        
        if current_mse_raw < best_mse_raw:
            best_mse_raw = current_mse_raw
            best_indices_raw = current_indices_raw.copy()
            _, best_w_raw = fit_ridge_raw(best_indices_raw)
            
        if current_mse_raw < 1e-12:
            break

    # Final Predict Function
    # It must generate the raw basis for X and multiply by best_w_raw
    selected_indices = best_indices_raw
    
    def predict(X):
        X_raw_test = generate_raw_basis(X)
        X_sub = X_raw_test[:, selected_indices]
        return X_sub @ best_w_raw

    return predict

# EVOLVE-BLOCK-END
