import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fits a polynomial of degree up to 5 to the training data using cross-validation
    to select the optimal degree, preventing overfitting on the unseen F=10 regime.
    """
    start_time = time.time()
    deadline = 150.0  # Safety margin before 120s limit? No, limit is 120s, evaluator kills at 150s. 
    # Actually, "evaluator KILLS the program after 150s wall-clock". 
    # "Each evaluation must finish within 120 s."
    # So we should aim for < 100s to be safe.
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    if len(x) < 6:
        # Not enough points for degree 5, fall back to linear or constant
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 6 - len(coeff)))
        except Exception:
            return np.zeros(6)

    # K-fold Cross-Validation for degree selection
    # We test degrees 1, 2, 3, 4, 5
    n_splits = 5
    indices = np.arange(len(x))
    np.random.seed(42) # For reproducibility in CV split
    np.random.shuffle(indices)
    
    fold_size = len(x) // n_splits
    if fold_size == 0:
        # If data is very small, just fit degree 3 or 5 directly? 
        # Assume training data is sufficiently large for L96 trajectories.
        pass

    best_mse = np.inf
    best_degree = 1
    
    # Pre-compute folds
    folds = [indices[i::n_splits] for i in range(n_splits)]
    
    for degree in range(1, 6):
        fold_mses = []
        for i in range(n_splits):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
            
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]
            
            try:
                # Fit polynomial
                coeff = np.polynomial.polynomial.polyfit(x_train, y_train, degree)
                # Evaluate
                y_pred = np.polynomial.polynomial.polyval(x_val, coeff)
                mse = np.mean((y_val - y_pred) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_degree = degree
            
        # Time check
        if time.time() > deadline - 10.0:
            break

    # Final fit on all data with best_degree
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, best_degree)
    except Exception:
        # Fallback to linear if fit fails
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
    # Ensure we return exactly 6 coefficients
    if len(coeff) < 6:
        coeff = np.pad(coeff, (0, 6 - len(coeff)))
    elif len(coeff) > 6:
        coeff = coeff[:6]
        
    # Verify finiteness
    if not np.all(np.isfinite(coeff)):
        # Replace non-finite values with 0
        coeff = np.where(np.isfinite(coeff), coeff, 0.0)

    return coeff
