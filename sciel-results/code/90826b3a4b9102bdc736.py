# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes a 2D Gaussian-like surface parameters to maximize the arithmetic Kakeya
    entropy ratio on a 10x10 grid.
    """
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute projection values
    proj_diff = X - Y       # x - y
    proj_x = X              # x
    proj_xy = X + Y         # x + y
    proj_x2y = X + 2 * Y    # x + 2y

    def compute_entropies(P):
        """
        Computes entropies for the given distribution P.
        """
        # Ensure P is normalized and non-negative
        P = np.maximum(P, 0)
        s = P.sum()
        if s < 1e-9:
            return 0, 0, 0, 0
        P = P / s
        
        def shannon_entropy(values, probs):
            # Flatten
            v = values.flatten()
            p = probs.flatten()
            # Pool by value
            unique_vals = np.unique(v)
            ent = 0.0
            for val in unique_vals:
                mask = (v == val)
                p_val = p[mask].sum()
                if p_val > 1e-15:
                    ent -= p_val * np.log2(p_val)
            return ent

        H_diff = shannon_entropy(proj_diff, P)
        H_x = shannon_entropy(proj_x, P)
        H_xy = shannon_entropy(proj_xy, P)
        H_x2y = shannon_entropy(proj_x2y, P)
        
        return H_diff, H_x, H_xy, H_x2y

    def objective(params):
        A, B, C, D, E = params
        # Quadratic form: A*x^2 + B*y^2 + C*x*y + D*x + E*y
        # We want P proportional to exp(-Q). 
        # To avoid overflow/underflow, we shift Q by its max value on the grid.
        Q = A * X**2 + B * Y**2 + C * X * Y + D * X + E * Y
        Q_max = Q.max()
        # exp(-(Q - Q_max)) = exp(Q_max - Q) = exp(-Q) * exp(Q_max)
        # We can just use exp(-(Q - Q_max)) as the unnormalized weights.
        # This ensures the max weight is 1, preventing overflow.
        weights = np.exp(-(Q - Q_max))
        
        # Check if weights are valid
        if not np.all(np.isfinite(weights)):
            return 1e9 # Penalty for bad params
        
        H_diff, H_x, H_xy, H_x2y = compute_entropies(weights)
        
        M = max(H_x, H_xy, H_x2y)
        
        if M < 1e-9:
            return 1e9 # Invalid M
            
        ratio = H_diff / M
        return -ratio # Minimize negative ratio

    # Initial guess: Slightly off-center Gaussian
    # A, B > 0 for decay. C for correlation.
    # Let's start with a symmetric Gaussian centered at 4.5
    # Q = a(x-4.5)^2 + b(y-4.5)^2 + c(x-4.5)(y-4.5)
    # Expand: a(x^2 - 9x + 20.25) + b(y^2 - 9y + 20.25) + c(xy - 4.5x - 4.5y + 20.25)
    # Coeffs:
    # x^2: a
    # y^2: b
    # xy: c
    # x: -9a - 4.5c
    # y: -9b - 4.5c
    
    a0 = 0.5
    b0 = 0.5
    c0 = 0.2
    
    A0 = a0
    B0 = b0
    C0 = c0
    D0 = -9*a0 - 4.5*c0
    E0 = -9*b0 - 4.5*c0
    
    x0 = np.array([A0, B0, C0, D0, E0])
    
    # Try multiple starts to avoid local minima
    best_result = None
    best_score = -np.inf
    
    starts = [
        x0,
        [0.1, 0.1, -0.5, 4.0, 4.0], # Different correlation
        [0.01, 0.01, 0.0, 0.0, 0.0], # Near uniform
        [1.0, 1.0, -1.0, 9.0, 9.0], # Strong negative corr
    ]
    
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = -res.fun
                    best_result = res
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to uniform
        return np.ones((10, 10), dtype=np.float64) / 100.0

    A, B, C, D, E = best_result.x
    Q = A * X**2 + B * Y**2 + C * X * Y + D * X + E * Y
    Q_max = Q.max()
    P = np.exp(-(Q - Q_max))
    
    # Normalize
    P[P < 0] = 0
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10), dtype=np.float64) / 100.0
    P = P / total
    
    return P

# EVOLVE-BLOCK-END
