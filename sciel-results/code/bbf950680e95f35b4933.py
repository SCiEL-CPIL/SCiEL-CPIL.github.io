import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Duffing-van der Pol oscillator with drive.
    a*x + b*x^3 + c*sin(x) - d*v - e*v^2 + f*cos(omega*t) + g*sin(omega*t) + h*x*v + i*t + j
    """
    a, b, c, d, e, f, g, h, i, j = params
    
    # Normalize inputs to O(1) scale for better conditioning
    x_norm = x / 0.35  # x range ~[-0.2, 0.17], so scale ~0.35
    v_norm = v / 0.35  # v range ~[-0.34, 0.36], so scale ~0.35
    
    # Restoring force: linear + cubic + sinusoidal
    restoring = a * x_norm + b * x_norm**3 + c * np.sin(x_norm)
    
    # Damping: linear + quadratic
    damping = -d * v_norm - e * v_norm**2
    
    # Drive: cos and sin components
    drive = f * np.cos(1.0 * t) + g * np.sin(1.0 * t)
    
    # Cross coupling
    cross = h * x_norm * v_norm
    
    # Linear time trend and bias
    trend = i * t + j
    
    result = restoring + damping + drive + cross + trend
    
    return result
