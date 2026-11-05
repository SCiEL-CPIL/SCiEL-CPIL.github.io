import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses Ridge Regression on centered inputs for stability and generalization.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten just in case
    x = x.flatten()
    y = y.flatten()
    
    N = len(x)
    if N == 0:
        return np.zeros(6)
    
    # 1. Centering (no scaling to avoid coefficient blow-up during transformation)
    mu_x = np.mean(x)
    
    z = x - mu_x
    
    # 2. Build Vandermonde matrix for degree 5
    # V[:, j] = z^j
    V = np.vander(z, N=6, increasing=True)
    
    # 3. Ridge Regression
    # Minimize ||V a - y||^2 + lambda ||a||^2
    # (V^T V + lambda I) a = V^T y
    
    VtV = V.T @ V
    Vty = V.T @ y
    
    # We want a robust lambda. 
    # The diagonal elements of VtV are sums of powers of z.
    # Let's try a small grid search for lambda to pick a good one.
    # Since we have time, we can test a few candidates.
    
    best_lambda = 0.0
    best_err = np.inf
    best_a = None
    
    # Candidate lambdas: 0 (OLS), and small values relative to diag
    diag_mean = np.mean(np.diag(VtV))
    # If diag_mean is 0, data is constant, handle separately
    if diag_mean < 1e-12:
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
        
    lambdas = [0.0, 1e-10 * diag_mean, 1e-8 * diag_mean, 1e-6 * diag_mean, 1e-4 * diag_mean]
    
    for lam in lambdas:
        if time.time() - start_time > DEADLINE:
            break
        A = VtV + lam * np.eye(6)
        try:
            a = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            continue
            
        # Calculate training error to guide lambda selection (proxy for generalization)
        # Note: Training error is not the final metric, but severe overfitting (high variance)
        # usually manifests as high sensitivity. However, without validation set, we rely on
        # the assumption that mild regularization is better than none or too much.
        # Actually, for polynomial fits, OLS (lambda=0) is often best if N >> 6.
        # Let's just pick a very small lambda for numerical stability.
        
        # Let's stick to a single robust choice to save time and complexity.
        # Lambda = 1e-8 * diag_mean is very mild.
        if lam == 1e-8 * diag_mean:
            best_a = a
            break
            
    if best_a is None:
        # Fallback to OLS if loop failed or didn't reach the preferred lambda
        try:
            best_a = np.linalg.solve(VtV, Vty)
        except:
            best_a = np.linalg.lstsq(V, y, rcond=None)[0]

    # 4. Transform coefficients from centered domain z = x - mu to original domain x
    # P(z) = sum_{k=0}^5 a_k z^k
    # z^k = (x - mu)^k = sum_{j=0}^k binom(k, j) x^j (-mu)^{k-j}
    # c_j = sum_{k=j}^5 a_k * binom(k, j) * (-mu)^{k-j}
    
    c = np.zeros(6)
    # Precompute powers of -mu
    neg_mu_powers = np.array([(-mu_x) ** i for i in range(6)])
    
    # Binomial coefficients for k=0..5
    # k=0: [1]
    # k=1: [1, 1]
    # k=2: [1, 2, 1]
    # k=3: [1, 3, 3, 1]
    # k=4: [1, 4, 6, 4, 1]
    # k=5: [1, 5, 10, 10, 5, 1]
    binoms = [
        [1],
        [1, 1],
        [1, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 6, 4, 1],
        [1, 5, 10, 10, 5, 1]
    ]
    
    for k in range(6):
        a_k = best_a[k]
        if a_k == 0:
            continue
        for j in range(k + 1):
            c[j] += a_k * binoms[k][j] * neg_mu_powers[k - j]
            
    # Ensure finite
    if not np.all(np.isfinite(c)):
        # Fallback to simple OLS on original x if something went wrong
        V_orig = np.vander(x, N=6, increasing=True)
        try:
             c, _, _, _ = np.linalg.lstsq(V_orig, y, rcond=None)
        except:
             c = np.zeros(6)
             c[0] = np.mean(y)
             
    return c
