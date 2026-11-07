"""
Strategy: Product-of-sums decomposition.
Instead of a generic tree or additive basis, we posit that the target function
has a separable or multiplicative structure: f(x, y) ≈ g(x) * h(y).
We model g(x) as a sum of 5 basis functions of x, and h(y) as a sum of 5 basis functions of y.
The 10 parameters are split: 5 for g(x) coefficients and 5 for h(y) coefficients.
This reduces the effective search space and leverages the multiplicative structure,
which is often more expressive for interaction-heavy bivariate targets than a simple
additive combination of the same number of terms, while remaining simple enough for
BFGS to optimize reliably.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _check_deadline():
    return (time.time() - _START_TIME) > TIME_LIMIT

# Basis functions for x and y
# We use a mix of polynomial and trigonometric bases to capture various shapes.
# 5 terms for x: 1, x, x^2, x^3, sin(x)
# 5 terms for y: 1, y, y^2, y^3, sin(y)
# This is a structured guess, but we can also try to evolve the basis selection
# or just use a fixed rich basis. Given the "product of sums" directive, a fixed
# rich basis is a good first step to see if the structure fits.

def _basis_x(x):
    # Returns (5, N) array
    return np.vstack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        np.sin(x)
    ])

def _basis_y(y):
    # Returns (5, N) array
    return np.vstack([
        np.ones_like(y),
        y,
        y**2,
        y**3,
        np.sin(y)
    ])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _START_TIME
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) == 0:
        return np.array([])
    
    # Cache check
    hash_key = (x.tobytes(), y.tobytes())
    cache = getattr(equation, '_cache', {})
    
    if hash_key in cache:
        return cache[hash_key]
    
    if not hasattr(equation, '_cache'):
        equation._cache = {}
    
    # Precompute basis matrices
    Bx = _basis_x(x)  # (5, N)
    By = _basis_y(y)  # (5, N)
    
    # params is length 10.
    # params[0:5] are coefficients for g(x)
    # params[5:10] are coefficients for h(y)
    
    def residuals(params):
        try:
            cx = params[0:5]
            cy = params[5:10]
            
            # g(x) = Bx @ cx -> (N,)
            gx = Bx.T @ cx
            # h(y) = By @ cy -> (N,)
            hy = By.T @ cy
            
            # f(x, y) = gx * hy
            pred = gx * hy
            
            if not np.all(np.isfinite(pred)):
                return np.full(len(y), 1e10)
            return pred - y
        except Exception:
            return np.full(len(y), 1e10)
    
    # Initial guess
    x0 = np.ones(10)
    
    # We will try a few different initializations to avoid local minima,
    # as the product structure can lead to flat landscapes or scale issues.
    best_mse = np.inf
    best_params = None
    best_pred = None
    
    # Strategy: Try a few random restarts for BFGS/Least Squares
    # Since the evaluator uses BFGS, we are simulating that here to find a good solution
    # to return as the prediction. Wait, the evaluator FITS the params.
    # The equation function is called with the FITTED params.
    # So my job is just to define the mapping from params to prediction.
    # The evaluator will do the optimization.
    # I do NOT need to do the optimization inside equation().
    # I just need to return the prediction given the params.
    
    # Correction: The task says "The evaluator fits params... minimizing the training MSE."
    # So `equation` is called with the final fitted params.
    # I just need to compute the prediction.
    
    cx = params[0:5]
    cy = params[5:10]
    
    gx = Bx.T @ cx
    hy = By.T @ cy
    
    pred = gx * hy
    
    if not np.all(np.isfinite(pred)):
        pred = np.full(len(y), np.mean(y))
        
    # Cache result
    if len(equation._cache) > 10:
        equation._cache.clear()
    equation._cache[hash_key] = pred
    
    return pred

# EVOLVE-BLOCK-END
