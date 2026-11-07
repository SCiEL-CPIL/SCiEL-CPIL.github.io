import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using state-space with time-varying
    coefficients from a low-order Fourier series in t.
    
    Model: a(t,x,v) = k(t)*x + d(t)*v + g(t)*x^2 + h(t)*v^2 + c(t)*x*v + f(t)
    where k(t), d(t), g(t), h(t), c(t), f(t) are linear combinations of
    Fourier basis functions [1, sin(wt), cos(wt)] with shared frequency w.
    
    Parameters (10 total):
    p0: frequency w
    p1: k0 (stiffness DC)
    p2: k1 (stiffness sin coeff)
    p3: k2 (stiffness cos coeff)
    p4: d0 (damping DC)
    p5: d1 (damping sin coeff)
    p6: d2 (damping cos coeff)
    p7: g0 (x^2 DC)
    p8: h0 (v^2 DC)
    p9: c0 (x*v DC)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to O(1) scale
    t_norm = (t - 40.0) / 10.0
    x_norm = (x + 0.015) / 0.2
    v_norm = (v - 0.01) / 0.35
    
    # Fourier basis in normalized time
    w = p0
    wt = w * t_norm
    s = np.sin(wt)
    c = np.cos(wt)
    
    # Time-varying coefficients
    k_t = p1 + p2 * s + p3 * c       # effective stiffness
    d_t = p4 + p5 * s + p6 * c       # effective damping
    
    # Static nonlinear terms
    g_t = p7                          # x^2 coefficient
    h_t = p8                          # v^2 coefficient
    c_t = p9                          # x*v coefficient
    
    # Acceleration
    result = k_t * x_norm + d_t * v_norm + g_t * x_norm**2 + h_t * v_norm**2 + c_t * x_norm * v_norm
    
    return result
