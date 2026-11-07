import numpy as np
import time
import sys

def fit(X_train, y_train):
    """
    Fits a regression model using Forward Feature Selection on a rich basis.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 330s total limit
    
    X = X_train
    y = y_train
    n_samples, n_features = X.shape
    
    # 1. Generate Candidate Feature Library
    # We assume inputs are positive based on the ranges provided in the prompt.
    # We will generate a diverse set of features.
    
    basis = [np.ones(n_samples)]  # Bias term
    
    # Polynomial features up to degree 3
    for d in range(1, 4):
        # Single variable powers
        for i in range(n_features):
            basis.append(X[:, i] ** d)
        
        # Interactions for degree 2 and 3
        if d == 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    if i != j:
                        basis.append(X[:, i] * X[:, j])
        elif d == 3:
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        # Only add if distinct or specific combinations to limit count
                        if len({i, j, k}) == 3:
                             basis.append(X[:, i] * X[:, j] * X[:, k])
    
    # Non-linear features
    for i in range(n_features):
        xi = X[:, i]
        # Trig
        basis.append(np.sin(xi))
        basis.append(np.cos(xi))
        # Log (safe as inputs >= 1)
        basis.append(np.log(xi))
        # Inverse
        basis.append(1.0 / xi)
        # Square root
        basis.append(np.sqrt(xi))
        
        # Gaussian RBFs centered at min, mid, max of the feature
        min_val = np.min(xi)
        max_val = np.max(xi)
        mid_val = (min_val + max_val) / 2.0
        spread = max(1e-6, (max_val - min_val) / 3.0)
        
        centers = [min_val, mid_val, max_val]
        for c in centers:
            basis.append(np.exp(-((xi - c) ** 2) / (2 * spread ** 2)))
    
    # Convert to matrix
    # Check for NaNs or Infs
    for b in basis:
        if not np.all(np.isfinite(b)):
             # Replace non-finite with 0 or remove feature
             b[np.isnan(b)] = 0
             b[np.isinf(b)] = 0
             
    Feature_Matrix = np.column_stack(basis)
    n_candidates = Feature_Matrix.shape[1]
    
    # 2. Forward Selection
    selected_indices = [0]  # Start with bias
    current_mse = np.inf
    max_features = 8  # Limit complexity
    min_improvement = 1e-6
    
    # Pre-compute y_centered for MSE calculation efficiency? 
    # MSE = mean((y - X_sel @ w)^2)
    # We will use Ridge Regression for stability.
    
    def get_ridge_coefficients(X_sel, y, alpha=1.0):
        # X_sel includes bias
        # Solve (X^T X + alpha * I) w = X^T y
        # Note: alpha should not be applied to the bias term usually, 
        # but for simplicity and stability with small data, we apply it to all 
        # or just non-bias. Let's apply to all for simplicity, 
        # bias is usually well-determined.
        XtX = X_sel.T @ X_sel
        Xty = X_sel.T @ y
        # Add regularization
        reg = np.eye(XtX.shape[0]) * alpha
        # Don't regularize bias (index 0)
        reg[0, 0] = 0
        try:
            w = np.linalg.solve(XtX + reg, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_sel, y, rcond=None)[0]
        return w
    
    def compute_mse(X_sel, y, w):
        pred = X_sel @ w
        return np.mean((y - pred) ** 2)
    
    # Initial fit with just bias
    w_current = get_ridge_coefficients(Feature_Matrix[:, selected_indices], y, alpha=1.0)
    current_mse = compute_mse(Feature_Matrix[:, selected_indices], y, w_current)
    
    for step in range(max_features):
        if time.time() > deadline - 10:
            break
            
        best_mse = current_mse
        best_idx = -1
        best_w = w_current
        
        # Evaluate adding each candidate feature
        for i in range(n_candidates):
            if i in selected_indices:
                continue
                
            # Time check inside loop
            if time.time() > deadline - 5:
                break
                
            new_indices = selected_indices + [i]
            X_new = Feature_Matrix[:, new_indices]
            
            # Quick check: correlation with residual? 
            # Or just fit. With n_candidates < 100 and n_samples=200, 
            # solving 9x9 systems is fast.
            
            w_new = get_ridge_coefficients(X_new, y, alpha=1.0)
            mse_new = compute_mse(X_new, y, w_new)
            
            if mse_new < best_mse - min_improvement:
                best_mse = mse_new
                best_idx = i
                best_w = w_new
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            w_current = best_w
            current_mse = best_mse
        else:
            break
            
    # 3. Final Fit
    # Use a slightly stronger regularization for the final model to prevent overfitting
    X_final = Feature_Matrix[:, selected_indices]
    w_final = get_ridge_coefficients(X_final, y, alpha=5.0)
    
    # Store selected indices and weights
    sel_idx = np.array(selected_indices)
    
    def predict(X_test):
        # Generate the same basis for test data
        # This is the expensive part if done naively, but n_test is small (200).
        # We can optimize by only generating the selected features.
        
        # Re-implement basis generation for only selected indices?
        # It's easier to regenerate the full basis and slice, 
        # but that's O(n_candidates * n_test).
        # n_candidates ~ 50-100. n_test = 200. 100*200 = 20,000 ops. Negligible.
        
        n_test = X_test.shape[0]
        basis_test = [np.ones(n_test)]
        
        # We need to map the index in the original basis list to the feature generation logic.
        # This requires the basis generation logic to be a function or we store the generator.
        # Since we generated `basis` as a list of arrays, we can just store the *type* of feature 
        # or regenerate the whole thing.
        
        # Regenerating the whole thing is safer to avoid index mismatch bugs.
        
        X_t = X_test
        b_list = [np.ones(n_test)]
        
        for d in range(1, 4):
            for i in range(X_t.shape[1]):
                b_list.append(X_t[:, i] ** d)
            if d == 2:
                for i in range(X_t.shape[1]):
                    for j in range(i, X_t.shape[1]):
                        if i != j:
                            b_list.append(X_t[:, i] * X_t[:, j])
            elif d == 3:
                for i in range(X_t.shape[1]):
                    for j in range(i, X_t.shape[1]):
                        for k in range(j, X_t.shape[1]):
                            if len({i, j, k}) == 3:
                                b_list.append(X_t[:, i] * X_t[:, j] * X_t[:, k])
                                
        for i in range(X_t.shape[1]):
            xi = X_t[:, i]
            b_list.append(np.sin(xi))
            b_list.append(np.cos(xi))
            b_list.append(np.log(xi))
            b_list.append(1.0 / xi)
            b_list.append(np.sqrt(xi))
            
            min_val = np.min(X_train[:, i])
            max_val = np.max(X_train[:, i])
            mid_val = (min_val + max_val) / 2.0
            spread = max(1e-6, (max_val - min_val) / 3.0)
            centers = [min_val, mid_val, max_val]
            for c in centers:
                b_list.append(np.exp(-((xi - c) ** 2) / (2 * spread ** 2)))
                
        Feature_Matrix_test = np.column_stack(b_list)
        
        # Ensure the length matches the training basis length
        if Feature_Matrix_test.shape[1] != Feature_Matrix.shape[1]:
            # Fallback: just use linear regression on raw features if basis mismatch
            # This shouldn't happen if logic is identical
            pass
            
        X_sel_test = Feature_Matrix_test[:, sel_idx]
        return X_sel_test @ w_final

    return predict
