"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_residual(x, y, params):
    """Compute residual for cubic polynomial."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    pred = (p0 + p1 * x + p2 * y + p3 * x**2 + p4 * x * y + p5 * y**2 +
            p6 * x**3 + p7 * x**2 * y + p8 * x * y**2 + p9 * y**3)
    return pred - y


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Complete cubic polynomial in x and y with 10 parameters.
    Uses multi-start BFGS optimization within a time budget to find the best fit.
    """
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    def objective(p):
        pred = (p[0] + p[1] * x + p[2] * y + p[3] * x**2 + p[4] * x * y + p[5] * y**2 +
                p[6] * x**3 + p[7] * x**2 * y + p[8] * x * y**2 + p[9] * y**3)
        residual = pred - y
        return np.sum(residual ** 2)
    
    # Start with the provided params
    best_params = params.copy()
    best_score = objective(best_params)
    
    # Generate multiple starting points
    rng = np.random.default_rng(42)
    starts = [params.copy()]
    
    # Add various random starts
    for _ in range(20):
        starts.append(rng.uniform(-5, 5, size=10))
    
    # Add some structured starts
    starts.append(np.ones(10) * 0.1)
    starts.append(np.ones(10) * 10.0)
    starts.append(np.zeros(10))
    
    # Try each start point within the time budget
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, start, method='BFGS', 
                            options={'maxiter': 500, 'gtol': 1e-6})
            
            if result.fun < best_score and np.all(np.isfinite(result.x)):
                best_score = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Compute final prediction
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = best_params
    return (p0 + p1 * x + p2 * y + p3 * x**2 + p4 * x * y + p5 * y**2 +
            p6 * x**3 + p7 * x**2 * y + p8 * x * y**2 + p9 * y**3)
# EVOLVE-BLOCK-END
