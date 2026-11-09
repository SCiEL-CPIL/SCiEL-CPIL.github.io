import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    
    Structure:
    a = c1*x + c2*x^3 + c3*x^2 + c4*v + c5*x^2*v + c6*v^3 + c7*sin(omega*t) + c8*cos(omega*t) + c9*omega + c10
    
    Wait, c9*omega? No, c9 is just a constant or another term.
    Let's map params explicitly.
    p0: x
    p1: x^3
    p2: x^2
    p3: v
    p4: x^2*v
    p5: v^3
    p6: sin(omega*t)
    p7: cos(omega*t)
    p8: omega
    p9: constant
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # State-dependent terms (Restoring + Damping)
    # Restoring: Linear, Cubic (Duffing), Quadratic
    restoring = p0 * x + p1 * x**3 + p2 * x**2
    
    # Damping: Linear, Nonlinear (x^2 v), Cubic
    damping = p3 * v + p4 * x**2 * v + p5 * v**3
    
    # Time-dependent drive
    # Using parameterized frequency p8
    # Centering t might help numerical stability for BFGS if p8 is small, 
    # but with sin/cos basis, it's a phase shift.
    # We'll use raw t as per hint "centered" might refer to the data preprocessing 
    # or just the phase. Since we have free sin/cos coeffs, phase is absorbed.
    
    # Safety check for omega to avoid issues if it grows huge (unlikely in 10 params fit)
    omega = p8
    # If omega is very large, sin/cos are noisy. 
    # But BFGS should find a reasonable value if the data supports it.
    
    drive = p6 * np.sin(omega * t) + p7 * np.cos(omega * t)
    
    # Constant term
    bias = p9
    
    result = restoring + damping + drive + bias
    
    return result
