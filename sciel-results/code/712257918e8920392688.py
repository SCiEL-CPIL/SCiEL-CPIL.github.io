"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < TIME_LIMIT


def _build_features(x, y):
    """
    Build a rich set of features for symbolic regression.
    We use a combination of polynomial, trigonometric, and interaction terms.
    The parameters will be fit via linear regression (or BFGS if needed).
    """
    n = len(x)
    # Polynomial features up to degree 3
    x2 = x ** 2
    y2 = y ** 2
    x3 = x ** 3
    y3 = y ** 3
    
    # Interaction terms
    xy = x * y
    x2y = x2 * y
    y2x = y2 * x
    x3y = x3 * y
    y3x = y3 * x
    x2y2 = x2 * y2
    
    # Higher order interactions
    x4 = x ** 4
    y4 = y ** 4
    x2y4 = x2 * y4
    y2x4 = y2 * x4
    
    # Trigonometric features
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    sin_xy = np.sin(x * y)
    cos_xy = np.cos(x * y)
    sin_x2 = np.sin(x2)
    cos_x2 = np.cos(x2)
    sin_y2 = np.sin(y2)
    cos_y2 = np.cos(y2)
    
    # Sqrt and other functions (with safe handling)
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    sqrt_x = np.sqrt(np.abs(x))
    sqrt_y = np.sqrt(np.abs(y))
    
    # Exponential features (clamped to avoid overflow)
    exp_x = np.exp(np.clip(x, -10, 10))
    exp_y = np.exp(np.clip(y, -10, 10))
    
    # Log features (with safe handling)
    log_abs_x = np.log(1 + np.abs(x))
    log_abs_y = np.log(1 + np.abs(y))
    
    # Build feature matrix
    features = np.column_stack([
        np.ones(n),
        x, y,
        x2, y2, x3, y3,
        xy, x2y, y2x, x3y, y3x, x2y2,
        x4, y4, x2y4, y2x4,
        sin_x, cos_x, sin_y, cos_y,
        sin_xy, cos_xy,
        sin_x2, cos_x2, sin_y2, cos_y2,
        abs_x, abs_y,
        sqrt_x, sqrt_y,
        exp_x, exp_y,
        log_abs_x, log_abs_y
    ])
    
    return features


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit params to the training data using linear least squares on the feature matrix.
    The evaluator calls scipy.optimize.minimize with BFGS, but we can use the
    linear structure to our advantage by pre-computing the feature matrix.
    
    However, since the evaluator fits params starting from all ones using BFGS,
    we need to make sure our feature representation allows good convergence.
    
    Strategy: Use a linear model with a rich feature set. The params array has 10
    elements, so we need to map 10 params to our features. We'll select the most
    important features and use a weighted combination.
    """
    n = len(x)
    
    # We have 10 params. Let's design a feature set that uses exactly 10 meaningful features.
    # This way, BFGS can fit them effectively.
    
    # Key features for symbolic regression with 2 inputs:
    # 1. Intercept
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. x*y
    # 7. sin(x)
    # 8. sin(y)
    # 9. x^3
    # 10. y^3
    
    x2 = x ** 2
    y2 = y ** 2
    x3 = x ** 3
    y3 = y ** 3
    xy = x * y
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    
    # Build 10-feature matrix
    F = np.column_stack([
        np.ones(n),
        x,
        y,
        x2,
        y2,
        xy,
        sin_x,
        sin_y,
        x3,
        y3
    ])
    
    # The params array is length 10, so we can directly use F @ params
    # But we need to handle the case where BFGS is fitting these params.
    # The evaluator will call minimize with our equation function.
    
    # For efficiency, we can also try to use a direct linear solve as a fallback
    # or to initialize better, but the evaluator controls the optimization.
    
    # Return the prediction
    pred = F @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
