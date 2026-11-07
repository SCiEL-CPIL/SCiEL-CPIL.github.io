import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator: Duffing + parametric forcing with rational saturation.
    
    a = -(k0+k1*cos(w2*tn))*xn - c*xn^3/(1+s*xn^2) - d*vn - d2*vn^2 + F0*sin(w*tn+phi)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize to O(1)
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.19
    v_n = (v - 0.01) / 0.35
    
    # Parametric linear stiffness
    k_t = p0 + p1 * np.cos(p7 * t_n)
    
    # Linear restoring
    lin = -k_t * x_n
    
    # Cubic with rational saturation (denominator >= 0.5 for safety)
    den = 1.0 + p3 * x_n**2
    den = np.where(den > 0.5, den, 0.5)
    cub = -p2 * x_n**3 / den
    
    # Damping
    damp = -p4 * v_n - p8 * v_n**2
    
    # Periodic forcing
    force = p5 * np.sin(p6 * t_n + p9)
    
    return lin + cub + damp + force
