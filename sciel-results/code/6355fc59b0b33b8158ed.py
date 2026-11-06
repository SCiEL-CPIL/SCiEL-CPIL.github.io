import numpy as np

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # 1. Scale x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
        
    x_scaled = (x - mu) / sigma
    
    # 2. Construct Vandermonde matrix for scaled x
    degrees = np.arange(6)
    V = np.power.outer(x_scaled, degrees)
    
    # 3. Regularized Least Squares
    # A = V.T @ V
    # b = V.T @ t
    A = V.T @ V
    b = V.T @ t
    
    # Regularization
    # We want to penalize large coefficients in the scaled basis to prevent overfitting
    # This also helps with the ill-conditioning of the polynomial basis
    mean_diag = np.mean(np.diag(A))
    if mean_diag == 0:
        return np.zeros(6)
        
    # Choose lambda. 1e-4 relative to the scale of A seems reasonable for generalization.
    # If the task was pure interpolation, lambda=0. But "unseen regime" implies generalization.
    lambda_reg = 1e-4 * mean_diag
    
    I = np.eye(6)
    A_reg = A + lambda_reg * I
    
    try:
        a = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        a, _, _, _ = np.linalg.lstsq(A_reg, b, rcond=None)
        
    # 4. Transform coefficients from scaled basis to original basis
    # t = sum_{k=0}^5 a_k * (x')^k
    # x' = (x - mu) / sigma
    # (x')^k = (1/sigma^k) * (x - mu)^k
    # (x - mu)^k = sum_{j=0}^k binom(k, j) * x^j * (-mu)^(k-j)
    # c_j = sum_{k=j}^5 a_k * (1/sigma^k) * binom(k, j) * (-mu)^(k-j)
    
    c = np.zeros(6)
    for j in range(6):
        c_j_val = 0.0
        for k in range(j, 6):
            # binom(k, j)
            # Calculate binomial coefficient
            # k choose j = k! / (j! (k-j)!)
            # Since k <= 5, we can use a simple function or precomputed values
            # Using math.comb is available in Python 3.8+
            import math
            binom_val = math.comb(k, j)
            term = a[k] * (1.0 / (sigma ** k)) * binom_val * ((-mu) ** (k - j))
            c_j_val += term
        c[j] = c_j_val
        
    if not np.all(np.isfinite(c)):
        return np.zeros(6)
        
    return c
