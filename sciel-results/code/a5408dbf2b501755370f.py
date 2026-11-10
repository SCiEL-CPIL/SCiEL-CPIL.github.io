"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Ensure 1D arrays
    x = x.flatten()
    y = y.flatten()
    
    n = len(x)
    if n < 10:
        # Fallback for very small data: return zeros or simple mean
        return np.zeros(6)

    # Degree of the polynomial
    degree = 5
    
    # Create Vandermonde matrix for degree 5
    # V[:, i] = x**i
    V = np.vander(x, N=degree + 1, increasing=True)
    
    # Split into train and validation sets for hyperparameter selection
    # Use a fixed random seed for reproducibility in the split
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    split_idx = int(0.8 * n)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    V_train = V[train_idx]
    y_train = y[train_idx]
    V_val = V[val_idx]
    y_val = y[val_idx]
    
    # Grid of lambda values for Ridge Regression
    # We want to cover a wide range of regularization strengths
    lambdas = np.logspace(-8, 4, 20)
    
    best_lambda = 0.0
    best_val_mse = np.inf
    best_coeff = np.zeros(degree + 1)
    
    # Precompute V_train.T @ V_train and V_train.T @ y_train for efficiency
    # However, since lambda changes, the term (V^T V + lambda I) changes.
    # We can precompute V^T V and V^T y.
    VtV = V_train.T @ V_train
    Vty = V_train.T @ y_train
    
    for lam in lambdas:
        # Check time limit
        if time.time() - start_time > 140.0:
            break
            
        # Ridge solution: c = (V^T V + lambda I)^-1 V^T y
        # Add lambda to the diagonal
        A = VtV + lam * np.eye(degree + 1)
        try:
            c = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular (unlikely with regularization)
            c = np.linalg.lstsq(V_train, y_train, rcond=None)[0]
            
        # Evaluate on validation set
        y_pred_val = V_val @ c
        mse_val = np.mean((y_pred_val - y_val) ** 2)
        
        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_lambda = lam
            best_coeff = c
            
    # Refit on the entire training set with the best lambda
    if best_lambda > 0:
        VtV_full = V.T @ V
        Vty_full = V.T @ y
        A_full = VtV_full + best_lambda * np.eye(degree + 1)
        try:
            final_coeff = np.linalg.solve(A_full, Vty_full)
        except np.linalg.LinAlgError:
            final_coeff = best_coeff
    else:
        final_coeff = best_coeff

    # Ensure coefficients are finite
    if not np.all(np.isfinite(final_coeff)):
        # Fallback to OLS if something went wrong
        try:
            final_coeff = np.linalg.lstsq(V, y, rcond=None)[0]
        except:
            final_coeff = np.zeros(degree + 1)

    return final_coeff
# EVOLVE-BLOCK-END
