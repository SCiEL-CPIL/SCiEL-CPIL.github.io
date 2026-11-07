"""Rewritten program using Forward Selection on a rich symbolic feature basis."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def fit(X_train, y_train):
    """
    Fits a regression model using Forward Selection on a rich set of symbolic features.
    """
    X = X_train
    y = y_train
    n_samples = X.shape[0]
    d = X.shape[1]
    
    # The problem states 1 input for all equations. 
    # If d > 1, we handle it generally, but the contract says 1 input.
    # We will assume the primary input is X[:, 0] for feature generation.
    # If d > 1, we might need to adjust, but the prompt specifies "1 input(s)".
    
    x = X[:, 0] if d > 0 else np.zeros(n_samples)
    
    # Define a rich set of candidate features
    # We use a small epsilon to avoid division by zero and log(0)
    eps = 1e-8
    
    features_dict = {}
    
    # Polynomials
    for k in range(1, 8):
        features_dict[f'poly_{k}'] = x ** k
        
    # Trigonometric
    for k in range(1, 5):
        features_dict[f'sin_{k}x'] = np.sin(k * x)
        features_dict[f'cos_{k}x'] = np.cos(k * x)
        
    # Exponential (clipped to prevent overflow)
    x_clip = np.clip(x, -10, 10)
    features_dict['exp_x'] = np.exp(x_clip)
    features_dict['exp_neg_x'] = np.exp(-x_clip)
    features_dict['exp_x2'] = np.exp(np.clip(x**2, -10, 10))
    
    # Logarithmic
    features_dict['log_abs_x'] = np.log(np.abs(x) + eps)
    features_dict['log_x2'] = np.log(x**2 + eps)
    
    # Rational
    features_dict['inv_x'] = 1.0 / (x + np.where(x >= 0, eps, -eps))
    features_dict['inv_x2'] = 1.0 / (x**2 + eps)
    
    # Mixed
    features_dict['x_sin_x'] = x * np.sin(x)
    features_dict['x_cos_x'] = x * np.cos(x)
    features_dict['x_exp_x'] = x * np.exp(np.clip(x, -10, 10))
    features_dict['x_log_abs_x'] = x * np.log(np.abs(x) + eps)
    
    # Tanh
    features_dict['tanh_x'] = np.tanh(x)
    
    # Absolute value
    features_dict['abs_x'] = np.abs(x)
    
    # Square root
    features_dict['sqrt_abs_x'] = np.sqrt(np.abs(x))

    # Prepare candidate matrix
    candidate_names = list(features_dict.keys())
    candidate_matrix = np.column_stack([features_dict[name] for name in candidate_names])
    
    # Handle any NaNs or Infs from numerical issues
    candidate_matrix = np.nan_to_num(candidate_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Start with intercept
    # We will use a list of column indices selected
    selected_indices = []
    
    # Function to fit and return SSE
    def fit_and_sse(indices):
        if len(indices) == 0:
            # Just intercept
            mean_y = np.mean(y)
            pred = np.full_like(y, mean_y)
        else:
            # Design matrix: Intercept + selected features
            X_design = np.column_stack([np.ones(n_samples)] + [candidate_matrix[:, i] for i in indices])
            try:
                # Use lstsq for OLS
                w, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                pred = X_design @ w
            except np.linalg.LinAlgError:
                return float('inf'), None
        sse = np.sum((y - pred) ** 2)
        return sse, pred

    # Initial fit with just intercept
    current_sse, _ = fit_and_sse([])
    
    max_features = 5 # Limit complexity to avoid overfitting, noiseless data helps but we don't want to fit noise if any exists
    min_improvement = 1e-6 * current_sse
    
    for _ in range(max_features):
        best_idx = -1
        best_sse = current_sse
        best_pred = None
        
        for i in range(len(candidate_names)):
            if i in selected_indices:
                continue
            new_indices = selected_indices + [i]
            sse, pred = fit_and_sse(new_indices)
            if sse < best_sse:
                best_sse = sse
                best_idx = i
                best_pred = pred
        
        # If no feature improves SSE significantly, stop
        if best_idx == -1 or (current_sse - best_sse) < min_improvement:
            break
            
        # Add the best feature
        selected_indices.append(best_idx)
        current_sse = best_sse
        # Update min improvement threshold relative to new SSE
        min_improvement = 1e-6 * current_sse

    # Final fit with the selected features to get the weights for the prediction function
    # We store the selected indices and the final weights
    final_indices = selected_indices
    
    # Construct the final design matrix to get weights
    if len(final_indices) == 0:
        w_final = np.array([np.mean(y)])
    else:
        X_design = np.column_stack([np.ones(n_samples)] + [candidate_matrix[:, i] for i in final_indices])
        w_final, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        
    # Store the selected feature names/indices for the predict function
    selected_feature_funcs = [features_dict[candidate_names[i]] for i in final_indices]
    
    def predict(X_test):
        X_t = X_test
        n_test = X_t.shape[0]
        x_t = X_t[:, 0] if X_t.shape[1] > 0 else np.zeros(n_test)
        
        # Recreate the features for test data
        # This is inefficient if done from scratch, but safe. 
        # Better: pre-compute the feature generation logic.
        
        # Re-generate features for test set
        test_features = []
        for idx in final_indices:
            name = candidate_names[idx]
            # Re-calculate feature based on name logic to ensure consistency
            # This is a bit hacky, better to store the function or lambda
            # Let's refactor to store lambdas or a function that generates features given x
            pass
            
        # Actually, let's just store the lambda functions or a generator.
        # Let's rewrite the feature generation to be callable.
        
        # This block is a placeholder for the actual implementation below
        return np.zeros(n_test)

    # RE-IMPLEMENTATION for cleaner predict
    # We need to capture the feature generation logic in a way that can be applied to new X.
    
    # Let's define a function that takes x and returns the feature matrix
    def generate_features(x_vec):
        feats = []
        # Polynomials
        for k in range(1, 8):
            feats.append(x_vec ** k)
        # Trigonometric
        for k in range(1, 5):
            feats.append(np.sin(k * x_vec))
            feats.append(np.cos(k * x_vec))
        # Exponential
        x_c = np.clip(x_vec, -10, 10)
        feats.append(np.exp(x_c))
        feats.append(np.exp(-x_c))
        feats.append(np.exp(np.clip(x_vec**2, -10, 10)))
        # Log
        feats.append(np.log(np.abs(x_vec) + eps))
        feats.append(np.log(x_vec**2 + eps))
        # Rational
        feats.append(1.0 / (x_vec + np.where(x_vec >= 0, eps, -eps)))
        feats.append(1.0 / (x_vec**2 + eps))
        # Mixed
        feats.append(x_vec * np.sin(x_vec))
        feats.append(x_vec * np.cos(x_vec))
        feats.append(x_vec * np.exp(np.clip(x_vec, -10, 10)))
        feats.append(x_vec * np.log(np.abs(x_vec) + eps))
        # Tanh
        feats.append(np.tanh(x_vec))
        # Abs
        feats.append(np.abs(x_vec))
        # Sqrt
        feats.append(np.sqrt(np.abs(x_vec)))
        
        return np.column_stack(feats)
    
    # We need to map the indices from the training feature matrix to this generator.
    # The order in `candidate_matrix` above must match `generate_features`.
    # Let's verify the order:
    # 1-7: poly 1-7
    # 8-9: sin x, cos x
    # 10-11: sin 2x, cos 2x
    # 12-13: sin 3x, cos 3x
    # 14-15: sin 4x, cos 4x
    # 16: exp x
    # 17: exp -x
    # 18: exp x2
    # 19: log abs x
    # 20: log x2
    # 21: inv x
    # 22: inv x2
    # 23: x sin x
    # 24: x cos x
    # 25: x exp x
    # 26: x log abs x
    # 27: tanh x
    # 28: abs x
    # 29: sqrt abs x
    
    # The `candidate_names` list order in the previous block:
    # poly_1..7 (7)
    # sin_1x, cos_1x, sin_2x, cos_2x, sin_3x, cos_3x, sin_4x, cos_4x (8)
    # exp_x, exp_neg_x, exp_x2 (3)
    # log_abs_x, log_x2 (2)
    # inv_x, inv_x2 (2)
    # x_sin_x, x_cos_x, x_exp_x, x_log_abs_x (4)
    # tanh_x (1)
    # abs_x (1)
    # sqrt_abs_x (1)
    # Total: 7+8+3+2+2+4+1+1+1 = 29 features.
    
    # The `generate_features` function above produces them in the same order.
    # So `final_indices` are valid indices into the output of `generate_features`.
    
    # Final Weights calculation (already done in w_final, but let's ensure it's consistent)
    # w_final corresponds to [Intercept, Feature_1, ..., Feature_k]
    
    def predict(X_test):
        x_t = X_test[:, 0] if X_test.shape[1] > 0 else np.zeros(X_test.shape[0])
        # Generate features
        F = generate_features(x_t)
        # Select the ones we trained on
        if len(final_indices) > 0:
            F_sel = F[:, final_indices]
        else:
            F_sel = np.zeros((X_test.shape[0], 0))
        
        # Construct design matrix with intercept
        X_design = np.column_stack([np.ones(X_test.shape[0])] + [F_sel]) if F_sel.size > 0 else np.ones((X_test.shape[0], 1))
        
        # Predict
        pred = X_design @ w_final
        return pred

    return predict

# EVOLVE-BLOCK-END
