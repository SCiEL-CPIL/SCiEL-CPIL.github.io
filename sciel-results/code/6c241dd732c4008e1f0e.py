import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    x = np.asarray(training_x, dtype=float).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=float).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Exploit physical antisymmetry: subgrid forcing is odd in x.
    # Restrict to odd powers: x^1, x^3, x^5.
    # This reduces effective dimensionality and improves generalization.
    
    # Construct design matrix for odd powers only: [x, x^3, x^5]
    V = np.column_stack([x, x**3, x**5])
    
    # Normalize columns to improve numerical conditioning
    col_norms = np.linalg.norm(V, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    V_norm = V / col_norms
    
    # Use ridge regression with cross-validation to select regularization parameter.
    # This helps prevent overfitting and improves performance on the unseen F=10 regime.
    
    # Candidate regularization parameters (log-spaced, centered around typical scale)
    lambdas = np.logspace(-8, 1, 20)
    
    # k-fold cross-validation
    k = 5
    if n < k:
        k = max(2, n)
    
    # Shuffle indices for CV with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_size = n // k
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
    if n % k != 0:
        folds[-1] = np.concatenate([folds[-1], indices[k*fold_size:]])
    
    best_lambda = lambdas[0]
    best_cv_score = np.inf
    best_c_norm = None
    
    for lam in lambdas:
        if time.time() > deadline - 15.0:
            break
        
        cv_errors = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            V_train = V_norm[train_idx]
            y_train = y[train_idx]
            V_val = V_norm[val_idx]
            y_val = y[val_idx]
            
            # Ridge regression on normalized design matrix
            try:
                A = V_train.T @ V_train + lam * np.eye(3)
                b = V_train.T @ y_train
                c_norm = np.linalg.solve(A, b)
                
                # Convert back to original scale
                c = c_norm / col_norms
                
                pred_val = V_val @ (c_norm / col_norms)
                mse = np.mean((pred_val - y_val)**2)
                cv_errors.append(mse)
            except np.linalg.LinAlgError:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_score:
            best_cv_score = mean_cv_error
            best_lambda = lam
    
    # Final fit on all data with best lambda
    A = V_norm.T @ V_norm + best_lambda * np.eye(3)
    b = V_norm.T @ y
    try:
        c_norm = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c_norm = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Convert back to original scale
    c_odd = c_norm / col_norms
    
    # Ensure finite
    c_odd = np.nan_to_num(c_odd, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Construct the full 6-coefficient vector in ascending powers
    # c = [c_0, c_1, c_2, c_3, c_4, c_5]
    # Odd powers only: c_1, c_3, c_5 are non-zero
    c = np.zeros(6)
    c[1] = c_odd[0]  # x^1
    c[3] = c_odd[1]  # x^3
    c[5] = c_odd[2]  # x^5
    
    return c
