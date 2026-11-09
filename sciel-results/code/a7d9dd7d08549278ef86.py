import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes a 10x10 probability distribution P[y, x] to maximize the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    This implementation uses a parameterized 2D Gaussian distribution and
    numerical optimization to find parameters that improve upon the simple
    column-vector baseline.
    """
    n = 10
    x_coords = np.arange(n)
    y_coords = np.arange(n)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    # P is defined as P[y, x], so we need to align coordinates
    # X is shape (10, 10) with X[i, j] = j (x)
    # Y is shape (10, 10) with Y[i, j] = i (y)
    
    def calc_entropy(probs):
        probs = probs[probs > 1e-12]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def get_pooled_probs(P, func):
        # P is (10, 10). P[y, x]
        # func takes x, y and returns value
        # We need to pool probabilities based on the value of func(x, y)
        values = func(X, Y)
        unique_vals, inverse = np.unique(values, return_inverse=True)
        pooled = np.zeros(len(unique_vals))
        np.add.at(pooled, inverse, P.flatten())
        return pooled
    
    def score(params):
        mu_x, mu_y, sigma_x, sigma_y, rho = params
        # Ensure sigma > 0
        sigma_x = max(sigma_x, 0.1)
        sigma_y = max(sigma_y, 0.1)
        rho = np.clip(rho, -0.99, 0.99)
        
        # Standardize
        zx = (X - mu_x) / sigma_x
        zy = (Y - mu_y) / sigma_y
        
        # Gaussian density
        exp_term = -(zx**2 - 2*rho*zx*zy + zy**2) / (2 * (1 - rho**2))
        P = np.exp(exp_term)
        
        # Normalize
        P_sum = np.sum(P)
        if P_sum < 1e-12:
            return 0.0
        P /= P_sum
        
        # Calculate Entropies
        # D = H(X - Y)
        pooled_D = get_pooled_probs(P, lambda x, y: x - y)
        H_D = calc_entropy(pooled_D)
        
        # M = max(H(X), H(X+Y), H(X+2Y))
        pooled_X = get_pooled_probs(P, lambda x, y: x)
        H_X = calc_entropy(pooled_X)
        
        pooled_XY = get_pooled_probs(P, lambda x, y: x + y)
        H_XY = calc_entropy(pooled_XY)
        
        pooled_X2Y = get_pooled_probs(P, lambda x, y: x + 2*y)
        H_X2Y = calc_entropy(pooled_X2Y)
        
        M = max(H_X, H_XY, H_X2Y)
        
        if M < 1e-9:
            return 0.0
            
        return H_D / M
    
    def neg_score(params):
        return -score(params)

    # Initial guess: Centered, moderate spread, some correlation
    # Try to start near the known good solution (column) or center
    # Column solution is effectively sigma_x -> 0, mu_x = 0, sigma_y ~ 3
    # Let's try a few initial points and pick the best
    
    initial_guesses = [
        [4.5, 4.5, 1.0, 1.0, 0.0],  # Centered
        [0.0, 4.5, 0.1, 2.0, 0.0],  # Near column
        [4.5, 4.5, 2.0, 1.0, 0.5],  # Correlated
        [4.5, 4.5, 1.0, 2.0, -0.5], # Anti-correlated
    ]
    
    best_p = None
    best_score = 0.0
    
    for guess in initial_guesses:
        try:
            res = minimize(neg_score, guess, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            s = -res.fun
            if s > best_score:
                best_score = s
                # Reconstruct P for best params
                mu_x, mu_y, sigma_x, sigma_y, rho = res.x
                sigma_x = max(sigma_x, 0.1)
                sigma_y = max(sigma_y, 0.1)
                rho = np.clip(rho, -0.99, 0.99)
                zx = (X - mu_x) / sigma_x
                zy = (Y - mu_y) / sigma_y
                exp_term = -(zx**2 - 2*rho*zx*zy + zy**2) / (2 * (1 - rho**2))
                P_candidate = np.exp(exp_term)
                P_candidate /= np.sum(P_candidate)
                best_p = P_candidate
        except Exception:
            continue
            
    if best_p is None:
        # Fallback to current known good solution
        P = np.zeros((n, n), dtype=np.float64)
        P[:, 0] = 1.0 / n
        return P
        
    return best_p
