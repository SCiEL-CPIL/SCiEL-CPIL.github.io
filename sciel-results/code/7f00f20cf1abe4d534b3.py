import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed sparse basis.
    
    Inputs:
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Returns:
    Acceleration
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    # Linear
    term0 = np.ones_like(x)
    term1 = x
    term2 = v
    
    # Quadratic
    x2 = x * x
    v2 = v * v
    term3 = x2
    term4 = v2
    term5 = x * v
    
    # Cubic
    x3 = x2 * x
    term6 = x3
    # Note: v3 is less common in standard oscillators than x3, 
    # but let's see. Maybe x^2 v is more useful for parametric effects.
    # Let's use x^2 v for term 9 instead of v^3? 
    # Or maybe sin(x) is more important.
    
    # Trigonometric
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Let's allocate the remaining slots:
    # We have used: 1, x, v, x^2, v^2, xv, x^3
    # That's 7 terms.
    # Remaining: sin(x), cos(x), and one more.
    # Candidates: v^3, x^2 v, x v^2, sin(v)
    
    # Let's use sin(x) and cos(x) as they are strong candidates for pendulum-like behavior.
    term7 = sin_x
    term8 = cos_x
    
    # Last term: x^2 v (interaction) or v^3?
    # x^2 v represents a velocity-dependent stiffness change.
    # v^3 represents nonlinear damping.
    # Let's go with x^2 v as it's a common term in parametric oscillators.
    term9 = x2 * v
    
    # Combine
    return (p0 * term0 + 
            p1 * term1 + 
            p2 * term2 + 
            p3 * term3 + 
            p4 * term4 + 
            p5 * term5 + 
            p6 * term6 + 
            p7 * term7 + 
            p8 * term8 + 
            p9 * term9)
