"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Convert to numpy arrays for efficiency
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove NaN/Inf values if any
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Fallback to simple linear fit if not enough data
        coeff = np.zeros(6)
        if len(x) >= 2:
            coeff[:2] = np.polyfit(x, y, 1)[::-1]
        return coeff
    
    # Normalize x to improve numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for degree-5 polynomial in normalized coordinates
    # V[:, j] = x_norm^j for j=0..5
    V = np.vander(x_norm, 6, increasing=True)
    
    # Ridge regression with multiple lambda values to find best fit
    # We'll use cross-validation on training data to select lambda
    best_lambda = 1e-6
    best_coeff_norm = None
    best_mse = np.inf
    
    # Try a range of regularization strengths
    lambdas = np.logspace(-10, 2, 20)
    
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
            
        # Ridge regression: (V^T V + lambda*I) c = V^T y
        # But we want to regularize only non-constant terms
        I = np.eye(6)
        I[0, 0] = 0  # Don't regularize the constant term
        
        A = V.T @ V + lam * I
        b = V.T @ y
        
        try:
            c_norm = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_norm = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Evaluate MSE on training data
        pred = V @ c_norm
        mse = np.mean((pred - y) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
            best_coeff_norm = c_norm
    
    # Convert from normalized to original x coordinates
    # P(x) = sum(c_j * ((x-x_mean)/x_std)^j)
    # Expand using binomial theorem
    coeff = np.zeros(6)
    if best_coeff_norm is not None:
        for j in range(6):
            cj = best_coeff_norm[j]
            # ((x - x_mean)/x_std)^j = (1/x_std^j) * sum_{k=0}^{j} binom(j,k) * (-x_mean)^{j-k} * x^k
            inv_std_power = 1.0 / (x_std ** j)
            for k in range(j + 1):
                binom_coeff = 1.0
                for m in range(k):
                    binom_coeff *= (j - m) / (m + 1)
                coeff[k] += cj * inv_std_power * binom_coeff * ((-x_mean) ** (j - k))
    
    # Sanity check: ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear fit
        coeff = np.zeros(6)
        if len(x) >= 2:
            coeff[:2] = np.polyfit(x, y, 1)[::-1]
    
    return coeff
# EVOLVE-BLOCK-END
