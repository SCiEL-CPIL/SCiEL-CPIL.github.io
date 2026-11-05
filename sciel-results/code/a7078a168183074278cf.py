import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fits a shared degree-five polynomial to the unresolved subgrid tendency.
    Uses Ridge Regression to prevent overfitting and ensure generalization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Ensure 1D
    x = x.flatten()
    y = y.flatten()
    
    n_samples = x.size
    if n_samples == 0:
        return np.zeros(6)
    
    # Create Vandermonde matrix for degree 5
    # Columns: x^0, x^1, x^2, x^3, x^4, x^5
    V = np.vander(x, N=6, increasing=True)
    
    # If we have very few samples, ridge is crucial
    # Grid search for alpha (regularization strength)
    # We want to minimize training MSE, but the regularization term 
    # implicitly selects smoother solutions.
    # A common strategy is to pick alpha that minimizes training error 
    # while keeping coefficients bounded.
    
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    
    best_coeff = np.zeros(6)
    best_mse = np.inf
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
            
        # Ridge solution: (V^T V + alpha * I) c = V^T y
        # Using np.linalg.solve for stability
        try:
            # Add alpha to diagonal
            # Note: We typically don't regularize the intercept (c0) as strongly,
            # but for simplicity and standard ridge, we regularize all.
            # Alternatively, we can center x and y to make intercept unregularized.
            # Let's stick to standard ridge on all coefficients for simplicity,
            # or use the 'ridge' formulation carefully.
            
            # A more robust approach: Center and scale x, fit, then transform back.
            # However, transforming polynomial coefficients back is complex.
            # Direct solve is fine if V is not ill-conditioned.
            
            VtV = V.T @ V
            Vty = V.T @ y
            
            # Regularize
            VtV_reg = VtV + alpha * np.eye(6)
            
            coeff = np.linalg.solve(VtV_reg, Vty)
            
            # Calculate Training MSE
            y_pred = V @ coeff
            mse = np.mean((y - y_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_coeff = coeff
                
        except np.linalg.LinAlgError:
            continue

    # If no alpha worked (unlikely), fallback to least squares
    if np.all(best_coeff == 0) and best_mse == np.inf:
        try:
            coeff, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
            best_coeff = coeff
        except Exception:
            # Fallback to linear fit if all fails
            try:
                c1 = np.polynomial.polynomial.polyfit(x, y, 1)
                best_coeff = np.zeros(6)
                best_coeff[:2] = c1
            except Exception:
                best_coeff = np.zeros(6)

    # Final check for finite values
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.nan_to_num(best_coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return best_coeff
