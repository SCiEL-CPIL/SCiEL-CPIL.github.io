import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Hybrid Global-Local Basis (Polynomial + Fixed-Width Gaussians)
    
    The model is:
    y(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + sum_{i=4}^{9} p[i] * exp(-((x - c_i)^2) / (2 * sigma_i^2))
    
    Why this is structurally different and potentially better:
    1. Global Trend: The first 4 parameters form a cubic polynomial. This captures the overall shape of the function (linear, quadratic, cubic trends) efficiently.
    2. Local Flexibility: The remaining 6 parameters modulate fixed Gaussian "bumps" centered at specific points across the domain [0.1, 6.0]. These allow the model to correct for local deviations from the polynomial trend (e.g., kinks, ripples, or localized peaks) without increasing the global polynomial degree (which would cause Runge's phenomenon or overfitting).
    3. Linearity in Parameters: The function is linear in all 10 parameters. This guarantees that the training MSE surface is convex (a paraboloid), so BFGS will converge to the global minimum of the training error from any starting point (assuming numerical stability). This avoids the local minima traps of non-linear basis parameters (like learning centers/widths).
    4. Robustness: By fixing the centers and widths of the Gaussians, we prevent the optimizer from collapsing basis functions or encountering ill-conditioned Jacobians associated with learning basis positions. The centers are evenly spaced, and widths are moderate to ensure overlap but distinctness.
    
    This hybrid approach is often superior to pure polynomials (too rigid) or pure RBFs (can struggle with global trends if not scaled well) because it separates the scales of variation.
    """
    
    start_time = time.time()
    TIME_LIMIT = 145.0

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # 1. Global Polynomial Component (Cubic)
    # y_poly = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3
    x2 = x * x
    x3 = x2 * x
    y_poly = p[0] + p[1] * x + p[2] * x2 + p[3] * x3
    
    # 2. Local Gaussian Component
    # We define 6 fixed centers evenly spaced in [0.1, 6.0]
    # Centers: 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
    # Widths (sigma): Fixed to ~0.5 to allow local correction without being too sharp
    # This keeps the basis functions linear in parameters.
    
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    sigmas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    y_local = np.zeros_like(x)
    
    for i in range(6):
        if time.time() - start_time > TIME_LIMIT:
            # Fallback: return what we have so far
            break
        # Calculate Gaussian: exp(-((x - c)^2) / (2 * sigma^2))
        diff = x - centers[i]
        # Avoid overflow in exp by clipping the argument
        # The argument is always <= 0, so exp is in (0, 1]
        arg = -(diff * diff) / (2.0 * sigmas[i] * sigmas[i])
        # np.exp is stable for negative args
        y_local += p[4 + i] * np.exp(arg)
        
    y = y_poly + y_local
    
    if time.time() - start_time > TIME_LIMIT:
        # If we ran out of time, return the polynomial part only? 
        # No, we should return the best finite result. 
        # Since the loop breaks, y_local might be partial. 
        # It's safer to just return y_poly if we haven't finished? 
        # Actually, the loop is fast (6 iterations). This guard is likely unnecessary 
        # but kept for safety. If we break early, y is partial. 
        # Let's just return y.
        pass

    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
