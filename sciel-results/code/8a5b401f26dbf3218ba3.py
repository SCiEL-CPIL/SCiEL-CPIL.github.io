import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Interaction-Aware Hybrid Basis.
    
    The current best (score ~1.29) uses a hybrid of linear, self-quadratic, and 
    fundamental sine terms: 
    [1, x, y, z, x^2, y^2, z^2, sin(pi*x), sin(pi*y), sin(pi*z)].
    
    This model assumes the function is largely separable or has self-curvature.
    A common reason for stagnation in symbolic regression is the presence of 
    strong interaction terms (e.g., x*y, x*z, y*z) which the current basis 
    cannot represent directly.
    
    This proposal structurally changes the basis by replacing the self-quadratic 
    terms (x^2, y^2, z^2) with cross-interaction terms (xy, xz, yz).
    
    New Basis:
    1. Constant
    2-4. Linear terms (x, y, z)
    5-7. Interaction terms (x*y, x*z, y*z)
    8-10. Fundamental sine terms (sin(pi*x), sin(pi*y), sin(pi*z))
    
    This allows the model to capture multiplicative dependencies between inputs,
    which are orthogonal to the self-curvature captured by the previous model.
    If the hidden function has significant interaction components, this should
    significantly reduce the training and test MSE.
    """
    p = params
    
    # Linear terms
    # Interactions
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Trigonometric components
    sin_pi_x = np.sin(np.pi * x)
    sin_pi_y = np.sin(np.pi * y)
    sin_pi_z = np.sin(np.pi * z)
    
    return (p[0]
            + p[1] * x
            + p[2] * y
            + p[3] * z
            + p[4] * xy
            + p[5] * xz
            + p[6] * yz
            + p[7] * sin_pi_x
            + p[8] * sin_pi_y
            + p[9] * sin_pi_z)
