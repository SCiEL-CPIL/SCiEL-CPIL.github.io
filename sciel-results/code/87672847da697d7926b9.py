"""Symbolic regression for a nonlinear oscillator with RBF-enhanced basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator with RBF-enhanced basis.
    
    Combines physical oscillator terms with Gaussian RBF terms for localized
    nonlinear corrections. The RBF centers are fixed at distinct points within
    the input ranges, and widths are scaled to the domain size.
    
    Form: p0 + p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6*sin(t) + p7*cos(t) + p8*RBF_t + p9*RBF_xv
    where RBF_t and RBF_xv are sums of Gaussian basis functions.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Physical terms
    term0 = p0
    term1 = p1 * x
    term2 = p2 * (x ** 3)
    term3 = p3 * v
    term4 = p4 * (v ** 3)
    term5 = p5 * x * v
    term6 = p6 * np.sin(t)
    term7 = p7 * np.cos(t)
    
    # RBF for time: centers spread over t range [30, 50]
    # Use 3 centers with width scaled to domain
    t_centers = np.array([35.0, 40.0, 45.0])
    t_width = 5.0  # domain size / 4
    
    rbf_t = np.zeros_like(t, dtype=float)
    for tc in t_centers:
        rbf_t += np.exp(-((t - tc) ** 2) / (2 * t_width ** 2))
    term8 = p8 * rbf_t
    
    # RBF for (x, v) plane: centers at distinct points
    # x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    x_centers = np.array([-0.15, 0.0, 0.12])
    v_centers = np.array([-0.25, 0.0, 0.28])
    
    # Create a small grid of centers in (x, v) space
    xv_rbf = np.zeros_like(x, dtype=float)
    x_width = 0.1  # ~40% of x range
    v_width = 0.15  # ~40% of v range
    
    for xc in x_centers:
        for vc in v_centers:
            xv_rbf += np.exp(-((x - xc) ** 2) / (2 * x_width ** 2) - ((v - vc) ** 2) / (2 * v_width ** 2))
    term9 = p9 * xv_rbf
    
    result = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
    return result
# EVOLVE-BLOCK-END
