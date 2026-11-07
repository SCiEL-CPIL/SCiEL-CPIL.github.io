"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with physics-inspired basis for nonlinear oscillator.
    
    Uses 3 RBF centers in (t, x, v) space plus 7 physics-inspired terms.
    All basis terms are normalized to unit scale for BFGS stability.
    """
    # Input ranges from task: t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    t_center1 = 35.0
    t_center2 = 40.0
    t_center3 = 45.0
    
    x_center1 = -0.065
    x_center2 = 0.0
    x_center3 = 0.065
    
    v_center1 = -0.113
    v_center2 = 0.0
    v_center3 = 0.113
    
    # RBF widths scaled to domain size
    t_width = 5.0
    x_width = 0.1
    v_width = 0.15
    
    # Compute RBF terms (Gaussian)
    rbf1 = np.exp(-((t - t_center1)**2 / (2 * t_width**2) + 
                    (x - x_center1)**2 / (2 * x_width**2) + 
                    (v - v_center1)**2 / (2 * v_width**2)))
    
    rbf2 = np.exp(-((t - t_center2)**2 / (2 * t_width**2) + 
                    (x - x_center2)**2 / (2 * x_width**2) + 
                    (v - v_center2)**2 / (2 * v_width**2)))
    
    rbf3 = np.exp(-((t - t_center3)**2 / (2 * t_width**2) + 
                    (x - x_center3)**2 / (2 * x_width**2) + 
                    (v - v_center3)**2 / (2 * v_width**2)))
    
    # Physics-inspired terms, normalized to unit scale
    # x range ~ [-0.2, 0.17], so x/0.2 ~ unit scale
    x_norm = x / 0.2
    x3_norm = (x / 0.2)**3 * 0.2  # keep x^3 but scaled
    
    # v range ~ [-0.34, 0.36], so v/0.35 ~ unit scale
    v_norm = v / 0.35
    
    # sin(t), cos(t) are already in [-1, 1]
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Cross term x*v, scale by max product ~ 0.2*0.35 = 0.07
    xv_norm = x * v / 0.07
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Combine: 3 RBF + 7 physics terms
    result = (p0 * rbf1 + 
              p1 * rbf2 + 
              p2 * rbf3 + 
              p3 * x_norm + 
              p4 * x3_norm + 
              p5 * v_norm + 
              p6 * sin_t + 
              p7 * cos_t + 
              p8 * xv_norm + 
              p9)
    
    return result
# EVOLVE-BLOCK-END
