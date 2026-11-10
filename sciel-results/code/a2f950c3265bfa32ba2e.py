"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial to the unresolved subgrid tendency.
    Uses a multi-strategy approach:
    1. Full degree-5 polynomial fit (baseline)
    2. Physics-informed sparse basis with L1 regularization (Lasso)
    3. Combined: use the better of the two based on a simple validation heuristic
    
    Returns 6 coefficients in ascending powers of x.
    """
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove NaN/inf if any
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        # Fallback: return zeros
        return np.zeros(6)
    
    # Strategy 1: Full degree-5 polynomial fit
    try:
        coeff_full = np.polynomial.polynomial.polyfit(x, y, 5)
        coeff_full = np.asarray(coeff_full, dtype=np.float64)
        if not np.all(np.isfinite(coeff_full)):
            coeff_full = np.zeros(6)
    except Exception:
        coeff_full = np.zeros(6)
    
    # Strategy 2: Sparse basis with L1 regularization (Lasso-style)
    # Build a basis that includes physics-informed terms
    # The unresolved tendency in Lorenz-96 fast scale involves:
    # - Linear term in X
    # - Cubic-like nonlinear terms
    # - Mean-field interactions
    # We'll use a set of candidate basis functions and select via L1
    
    if time.time() > deadline - 20:
        # Not enough time, return the full polynomial fit
        return coeff_full
    
    # Candidate basis functions (all should be representable as polynomial in x)
    # For a single x, we can only use powers of x. But we can try different
    # polynomial degrees and use regularization to select.
    
    # Let's try Lasso on polynomial features of degree 5
    # Build design matrix
    X_design = np.column_stack([x**i for i in range(6)])
    
    # Normalize y for better conditioning
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Normalize columns of design matrix
    X_mean = np.mean(X_design, axis=0)
    X_std = np.std(X_design, axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_norm = (X_design - X_mean) / X_std
    
    # Try several lambda values for L1 regularization
    # Use a simple coordinate descent or scipy's lasso if available
    best_l1_coeff = None
    best_l1_score = np.inf
    
    lambda_values = np.logspace(-6, 3, 20)
    
    for lam in lambda_values:
        if time.time() > deadline - 15:
            break
        
        # Solve Lasso: minimize 0.5/n * ||y - Xw||^2 + lam * ||w||_1
        # Use scipy.optimize.minimize with L1 penalty
        try:
            n = len(y_norm)
            
            def objective(w):
                pred = X_norm @ w
                mse = 0.5 / n * np.sum((y_norm - pred) ** 2)
                l1 = lam * np.sum(np.abs(w))
                return mse + l1
            
            # Initial guess: OLS solution
            try:
                w0, _, _, _ = np.linalg.lstsq(X_norm, y_norm, rcond=None)
            except Exception:
                w0 = np.zeros(6)
            
            result = minimize(objective, w0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12})
            
            if result.success:
                w_l1 = result.x
                # Evaluate on training data (as proxy for validation)
                pred = X_norm @ w_l1
                score = np.mean((y_norm - pred) ** 2)
                
                # Penalize complexity
                non_zero = np.sum(np.abs(w_l1) > 1e-6)
                score += 0.01 * non_zero
                
                if score < best_l1_score:
                    best_l1_score = score
                    best_l1_coeff = w_l1.copy()
        except Exception:
            continue
    
    # Convert L1 coefficients back to original scale
    if best_l1_coeff is not None:
        # Undo normalization
        # pred = X_norm @ w_l1 = (X - X_mean)/X_std @ w_l1
        # We want: y = X @ c
        # X_norm @ w_l1 = (X - X_mean)/X_std @ w_l1
        # = X @ (w_l1/X_std) - X_mean @ (w_l1/X_std)
        # So c = w_l1/X_std, and we need to account for the offset
        # Actually, let's just do a direct polynomial fit with the selected support
        
        # Identify support (non-zero coefficients)
        support = np.abs(best_l1_coeff) > 1e-6
        if np.sum(support) >= 1:
            # Fit only on supported terms
            X_sub = X_design[:, support]
            try:
                c_sub, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
                c_full_l1 = np.zeros(6)
                c_full_l1[support] = c_sub
                if np.all(np.isfinite(c_full_l1)):
                    # Compare training MSE
                    pred_full = np.polynomial.polynomial.polyval(x, coeff_full)
                    mse_full = np.mean((y - pred_full) ** 2)
                    
                    pred_l1 = X_design @ c_full_l1
                    mse_l1 = np.mean((y - pred_l1) ** 2)
                    
                    # Use the one with lower training MSE as a heuristic
                    if mse_l1 < mse_full * 1.1:
                        return c_full_l1
            except Exception:
                pass
    
    # Return the full polynomial fit as default
    return coeff_full
# EVOLVE-BLOCK-END
