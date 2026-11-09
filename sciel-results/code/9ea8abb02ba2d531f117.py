import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 110.0  # 10s buffer before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
        
    # Handle case where x has very small variance
    x_min = np.min(x)
    x_max = np.max(x)
    x_center = (x_max + x_min) / 2.0
    x_half_range = (x_max - x_min) / 2.0
    
    if x_half_range < 1e-9:
        x_half_range = 1.0
        
    # Scale x to u in [-1, 1]
    u = (x - x_center) / x_half_range
    
    # Build Vandermonde matrix for u
    # V[i, j] = u[i]^j
    V = np.vander(u, N=6, increasing=True)
    
    # Ridge Regression
    # lambda = 1e-6 * trace(V^T V) / 6
    # Or just a small fixed value relative to data magnitude
    # Let's use a small relative regularization
    VtV = V.T @ V
    Vty = V.T @ y
    
    # Regularization parameter
    # If data is large, VtV elements are large.
    # lambda = 1e-6 * mean(diag(VtV))
    lam = 1e-6 * np.mean(np.diag(VtV))
    if lam == 0:
        lam = 1e-6
        
    A = VtV + lam * np.eye(6)
    
    try:
        a = np.linalg.solve(A, Vty)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (should not happen with ridge)
        a = np.linalg.lstsq(V, y, rcond=None)[0]
        
    # Check time
    if time.time() - start_time > deadline:
         # This shouldn't happen, but return zeros if it does? 
         # No, the solve is fast.
         pass
         
    # Transform coefficients from u-space to x-space
    # u = alpha * x + beta
    alpha = 1.0 / x_half_range
    beta = -x_center / x_half_range
    
    c = np.zeros(6)
    # c_j = sum_{k=j}^5 a_k * binom(k, j) * alpha^j * beta^(k-j)
    
    # Precompute powers of alpha and beta
    # This is a small loop, 6x6
    for j in range(6):
        c_j = 0.0
        for k in range(j, 6):
            # binom(k, j)
            # k is small, compute directly
            # binom(k, j) = k! / (j! * (k-j)!)
            # We can use math.comb if available, or simple loop
            # Python 3.8+ has math.comb
            try:
                from math import comb
                binom_val = comb(k, j)
            except ImportError:
                # Fallback
                def _comb(n, r):
                    if r < 0 or r > n:
                        return 0
                    r = min(r, n - r)
                    num = 1
                    den = 1
                    for i in range(r):
                        num *= (n - i)
                        den *= (i + 1)
                    return num // den
                binom_val = _comb(k, j)
                
            term = a[k] * binom_val * (alpha ** j) * (beta ** (k - j))
            c_j += term
        c[j] = c_j
        
    return c
