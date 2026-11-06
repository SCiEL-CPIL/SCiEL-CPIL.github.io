import numpy as np

def construct(training_x, training_subgrid_tendency):
    """
    Fits a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Args:
        training_x: Array of slow-state values.
        training_subgrid_tendency: Array of corresponding subgrid tendencies.
        
    Returns:
        Array of 6 polynomial coefficients [c0, c1, c2, c3, c4, c5] 
        such that tendency ~ c0 + c1*x + ... + c5*x^5.
    """
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter out non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Not enough data for a degree-5 fit, return zeros or best effort
        # If very little data, linear fit might be safer, but contract expects 6 coeffs.
        # Zeros is a neutral guess.
        return np.zeros(6)
    
    # Standardize x for numerical stability during fitting
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
        
    x_scaled = (x - mu) / sigma
    
    # Construct Vandermonde matrix for scaled x (degree 0 to 5)
    # V[:, i] = x_scaled^i
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Solve least squares: V * beta = y
    # Using lstsq for stability
    try:
        beta, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to simple linear fit if lstsq fails (unlikely)
        beta = np.zeros(6)
        if len(x) >= 2:
            coeff_lin = np.polyfit(x, y, 1)
            # Convert polyfit (descending) to our ascending scaled basis? 
            # Actually, just return zeros or a very simple fit.
            # Let's just return zeros to be safe if something goes wrong.
            return np.zeros(6)
            
    # Convert coefficients from scaled basis (x_scaled) to raw basis (x)
    # x_scaled = (x - mu) / sigma  =>  x = sigma * x_scaled + mu
    # We have P(x_scaled) = sum_{i=0}^5 beta_i * x_scaled^i
    # We want Q(x) = sum_{j=0}^5 alpha_j * x^j such that Q(x) = P((x-mu)/sigma)
    
    # Expansion: x_scaled^i = ( (x - mu) / sigma )^i = sigma^{-i} * (x - mu)^i
    # (x - mu)^i = sum_{j=0}^i binom(i, j) x^j * (-mu)^(i-j)
    # So term beta_i * x_scaled^i contributes to alpha_j (for j <= i):
    # beta_i * sigma^{-i} * binom(i, j) * (-mu)^(i-j)
    
    alpha = np.zeros(6)
    for i in range(6):
        if beta[i] == 0:
            continue
        coeff_base = beta[i] / (sigma ** i)
        for j in range(i + 1):
            # binom(i, j) * (-mu)^(i-j)
            binom_val = float(np.math.comb(i, j)) if hasattr(np, 'math') else 1.0
            # numpy doesn't have math.comb in older versions, use scipy or manual
            # Let's implement a simple binomial coefficient or use scipy if available.
            # Standard library only? "Python standard library, NumPy and SciPy are available."
            # We can use scipy.special.comb or just calculate it.
            try:
                import scipy.special as sp
                binom_val = sp.comb(i, j, exact=False)
            except ImportError:
                # Fallback calculation
                binom_val = 1.0
                for k in range(1, j+1):
                    binom_val = binom_val * (i - k + 1) / k
            
            alpha[j] += coeff_base * binom_val * ((-mu) ** (i - j))
            
    return alpha
