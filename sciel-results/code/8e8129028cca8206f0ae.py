import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation of motion for a damped nonlinear oscillator with periodic driving.
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p[0]: Amplitude of 1st harmonic drive
        p[1]: Frequency of drive
        p[2]: Phase of drive
        p[3]: Linear spring constant
        p[4]: Cubic spring constant
        p[5]: Linear damping constant
        p[6]: Cubic damping constant
        p[7]: Cross-term damping (x*v)
        p[8]: Quadratic spring constant
        p[9]: Amplitude of 2nd harmonic drive
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Drive terms
    drive_1 = p0 * np.cos(p1 * t + p2)
    drive_2 = p9 * np.cos(2.0 * p1 * t + p2)
    
    # Spring terms
    spring = p3 * x + p8 * x**2 + p4 * x**3
    
    # Damping terms
    damping = p5 * v + p7 * x * v + p6 * v**3
    
    return drive_1 + drive_2 + spring + damping
