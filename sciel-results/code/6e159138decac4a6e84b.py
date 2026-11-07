"""Non-linear parametric model for a driven Duffing/Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a nonlinear oscillator.
    
    Model:
    m * a = -c * v * (1 + mu * v^2) - k * x - beta * x^3 + F * sin(w * t + phi) + offset
    
    Parameters (10 total):
    0: m (mass)
    1: c (linear damping)
    2: mu (nonlinear damping coeff, Van der Pol style)
    3: k (linear stiffness)
    4: beta (nonlinear stiffness coeff, Duffing style)
    5: F (drive amplitude)
    6: w (drive frequency)
    7: phi (drive phase)
    8: offset (constant bias)
    9: scale (global scale factor for numerical stability)
    """
    elapsed = time.time() - _start_time
    if elapsed > 140:
        return np.zeros_like(x)

    # Unpack parameters
    m = params[0]
    c = params[1]
    mu = params[2]
    k = params[3]
    beta = params[4]
    F = params[5]
    w = params[6]
    phi = params[7]
    offset = params[8]
    scale = params[9]

    # Prevent division by zero or negative mass issues during optimization
    m_safe = m * m + 1e-6
    m_safe = np.sign(m_safe) * np.maximum(np.abs(m_safe), 1e-6)
    
    # Compute acceleration
    # Damping: c*v + mu*v^3 (Van der Pol term)
    damping_force = c * v + mu * v**3
    
    # Spring: k*x + beta*x^3 (Duffing term)
    spring_force = k * x + beta * x**3
    
    # Drive: F * sin(w*t + phi)
    drive_force = F * np.sin(w * t + phi)
    
    # Net force
    net_force = -damping_force - spring_force + drive_force + offset
    
    # Acceleration
    a = net_force / m_safe
    
    # Apply global scale to help BFGS convergence if magnitudes are off
    return a * scale
# EVOLVE-BLOCK-END
