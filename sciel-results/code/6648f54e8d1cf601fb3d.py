"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-component parametric Gaussian Mixture Model.
    
    Each component i has:
      - amplitude: a_i
      - mean: mu_i = (mu_x_i, mu_y_i)
      - variance: sigma_i^2 (shared for x and y directions)
    
    Total parameters: 2 * (1 + 2 + 1) = 8, plus 2 for a global linear offset.
    
    Model: f(x,y) = c0 + c1*x + c2*y + sum_i a_i * exp(-((x-mu_xi)^2 + (y-mu_yi)^2) / (2*sigma_i^2))
    
    This captures localized bumps and multi-modal structures that a global
    polynomial or Fourier basis cannot represent efficiently with 10 params.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Global linear offset (2 params)
    base = p0 + p1 * x + p2 * y
    
    # Component 1 (3 params: amplitude, mu_x, mu_y) + 1 variance
    a1 = p3
    mu_x1 = p4
    mu_y1 = p5
    sigma1 = p6
    # Ensure sigma is positive; use abs to avoid issues with BFGS
    s1 = np.abs(sigma1) + 1e-6
    r1_sq = (x - mu_x1) ** 2 + (y - mu_y1) ** 2
    comp1 = a1 * np.exp(-r1_sq / (2.0 * s1 * s1))
    
    # Component 2 (3 params: amplitude, mu_x, mu_y) + 1 variance
    a2 = p7
    mu_x2 = p8
    mu_y2 = p9
    # We only have 10 params total, so we need to be careful.
    # Let's restructure: 
    # p0, p1, p2: linear offset
    # p3, p4, p5, p6: comp1 (amp, mux, muy, sigma)
    # p7, p8, p9: comp2 (amp, mux, muy) -- but we need sigma for comp2 too.
    
    # Actually with 10 params:
    # p0, p1, p2: linear (3)
    # p3, p4, p5, p6: comp1 amp, mux, muy, sigma (4)
    # p7, p8, p9: comp2 amp, mux, muy (3) -- missing sigma for comp2
    
    # Let's use a different allocation:
    # p0: global offset
    # p1, p2: linear x, y
    # p3, p4, p5: comp1 (amp, mux, muy)
    # p6: comp1 sigma
    # p7, p8, p9: comp2 (amp, mux, muy)
    # But we still need comp2 sigma. Let's share sigma or use a fixed one.
    
    # Alternative: use 2 components with shared sigma
    # p0, p1, p2: linear offset (3)
    # p3, p4, p5: comp1 (amp, mux, muy) (3)
    # p6, p7, p8: comp2 (amp, mux, muy) (3)
    # p9: shared sigma (1)
    # Total: 10
    
    # Let's redo with this allocation:
    c0 = p0
    cx = p1
    cy = p2
    
    a1 = p3
    mu_x1 = p4
    mu_y1 = p5
    
    a2 = p6
    mu_x2 = p7
    mu_y2 = p8
    
    sigma = np.abs(p9) + 1e-6
    
    r1_sq = (x - mu_x1) ** 2 + (y - mu_y1) ** 2
    r2_sq = (x - mu_x2) ** 2 + (y - mu_y2) ** 2
    
    comp1 = a1 * np.exp(-r1_sq / (2.0 * sigma * sigma))
    comp2 = a2 * np.exp(-r2_sq / (2.0 * sigma * sigma))
    
    return c0 + cx * x + cy * y + comp1 + comp2
# EVOLVE-BLOCK-END
