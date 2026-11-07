"""
Strategy: Adaptive Spectral Feature Selection via Greedy Forward Search.

Unlike the seed program (fixed degree-2 polynomial) or inspirations (kernel/GP tuning),
this strategy assumes the hidden equations are composed of a small number of
"basis primitives" (e.g., x, x^2, sin(x), exp(x)). 

We generate a rich library of candidate features and use a fast, iterative 
greedy forward selection algorithm to pick the top K features that minimize 
training MSE. This acts as a lightweight symbolic regression engine that 
adapts to the specific functional form of each hidden equation without 
expensive hyperparameter optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_library(x):
    """
    Generate a diverse library of candidate features for 1D input x.
    Includes polynomials, trigonometric, exponential, and hyperbolic functions.
    """
    feats = []
    # Polynomials up to degree 8
    for d in range(1, 9):
        feats.append(x ** d)
    
    # Trigonometric (scaled to match input range roughly [-3, 3])
    # Input ranges are [-3,3], [-2,2], [-2,2], [0.1,6]
    # Use a few frequencies
    for f in [0.5, 1.0, 1.5, 2.0, 3.0]:
        feats.append(np.sin(f * x))
        feats.append(np.cos(f * x))
    
    # Exponentials (damped to avoid overflow/underflow issues in fit)
    # We use small coefficients to keep values bounded
    for c in [0.2, 0.5, 1.0]:
        feats.append(np.exp(c * x))
        feats.append(np.exp(-c * x))
        
    # Hyperbolic
    feats.append(np.tanh(x))
    feats.append(np.tanh(2*x))
    
    # Absolute value and sign
    feats.append(np.abs(x))
    
    # Reciprocals (safe: add small epsilon or mask zeros)
    # Since ranges don't include 0 for the last one, but others do.
    # We'll skip 1/x to avoid infinities if x=0 is present, or handle via masking later.
    # Actually, let's add x * exp(-x^2) type gaussians
    for c in [1.0, 2.0]:
        feats.append(x * np.exp(-c * x**2))
        feats.append(np.exp(-c * x**2))
        
    return np.column_stack(feats)

def _greedy_forward_select(X, y, max_features=5, time_limit=20.0):
    """
    Iteratively add the feature that reduces MSE the most.
    X: (n, m) feature matrix
    y: (n,) target vector
    """
    start_time = time.time()
    n, m = X.shape
    
    # Normalize y for numerical stability in selection? 
    # MSE is scale invariant for selection if we use relative error, 
    # but standard MSE works fine for ranking if features are similar scale.
    # Let's standardize X columns to 0 mean, 1 var to make feature comparison fair.
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    selected_indices = []
    current_mse = np.var(y)  # Baseline: predicting mean
    
    for _ in range(max_features):
        if time.time() - start_time > time_limit:
            break
            
        best_idx = -1
        best_mse = current_mse
        
        # Try adding each unselected feature
        for j in range(m):
            if j in selected_indices:
                continue
            
            # Candidate set: current selected + new feature j
            cand_indices = selected_indices + [j]
            X_cand = X_norm[:, cand_indices]
            
            # Solve least squares
            # Add small ridge for stability
            try:
                # Use np.linalg.lstsq for speed with small K
                w, *_ = np.linalg.lstsq(X_cand, y, rcond=None)
                y_pred = X_cand @ w
                mse = np.mean((y - y_pred) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_idx = j
            except Exception:
                continue
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        current_mse = best_mse
        
        # Early stopping if improvement is negligible
        if current_mse < 1e-10 * np.var(y):
            break

    return selected_indices, current_mse

def fit(X_train, y_train):
    start_time = time.time()
    
    # Handle 1D case specifically as per task description
    # X_train shape (200, 1)
    x = X_train[:, 0]
    y = y_train
    
    # 1. Generate Library
    library = _generate_library(x)
    
    # 2. Greedy Selection
    selected_indices, train_mse = _greedy_forward_select(library, y, max_features=5, time_limit=15.0)
    
    # 3. Fit final model using the selected features
    if len(selected_indices) == 0:
        # Fallback to mean
        mean_y = np.mean(y)
        return lambda X: np.full(X.shape[0], mean_y)
        
    X_sel = library[:, selected_indices]
    
    # Use Ridge Regression for the final fit to prevent overfitting on the 200 points
    # with potentially correlated features
    # Alpha search
    best_w = None
    best_val_mse = np.inf
    
    # We don't have a validation set, so we use Ridge with a reasonable alpha
    # or just OLS if features are well-conditioned.
    # Given we selected 5 features from 200 points, OLS is usually safe, 
    # but Ridge is safer against collinearity (e.g., x and x^2).
    
    # Let's try a small grid of alphas
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    for alpha in alphas:
        try:
            # Ridge: (X^T X + alpha I) w = X^T y
            A = X_sel.T @ X_sel + alpha * np.eye(len(selected_indices))
            b = X_sel.T @ y
            w = np.linalg.solve(A, b)
            y_pred = X_sel @ w
            mse = np.mean((y - y_pred) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_w = w
        except Exception:
            continue
            
    if best_w is None:
        # Fallback to OLS
        w, *_ = np.linalg.lstsq(X_sel, y, rcond=None)
        best_w = w

    def predict(X):
        # Reconstruct library for test set
        x_test = X[:, 0]
        lib_test = _generate_library(x_test)
        X_sel_test = lib_test[:, selected_indices]
        return X_sel_test @ best_w

    return predict
# EVOLVE-BLOCK-END
