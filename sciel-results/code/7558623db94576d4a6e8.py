"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a rich polynomial basis
    in scaled position and velocity, plus Gaussian RBF terms for localized dynamics.
    
    The 10 parameters are:
    - params[0:6] : coefficients for polynomial basis terms
    - params[6:8] : Gaussian RBF center (cx, cv) in scaled space
    - params[8:10]: Gaussian RBF width (sigma_x, sigma_v) in scaled space
    """
    # Scale inputs to O(1) for better conditioning
    # x range ~ [-0.57, 0.53], v range ~ [-0.18, 0.28]
    X = x / 0.5
    V = v / 0.3
    
    # Polynomial basis terms
    X2 = X * X
    V2 = V * V
    X3 = X2 * X
    V3 = V2 * V
    XV = X * V
    
    # Six polynomial terms
    poly = (params[0] + 
            params[1] * X + 
            params[2] * V + 
            params[3] * X2 + 
            params[4] * V2 + 
            params[5] * XV)
    
    # Gaussian RBF term: localized bump in (x, v) space
    cx = params[6]
    cv = params[7]
    sx = params[8]
    sv = params[9]
    
    # Ensure widths are positive to avoid division by zero
    sx_safe = np.abs(sx) + 1e-6
    sv_safe = np.abs(sv) + 1e-6
    
    dx = X - cx
    dv = V - cv
    
    # Gaussian: exp(-0.5 * (dx/sx)^2 - 0.5 * (dv/sv)^2)
    # We use a scaled amplitude of 1.0 (the RBF coefficient is implicit in the width/center fitting)
    # Actually, we need a coefficient for the RBF too. Let's restructure.
    
    # Let's use a different parameter allocation for better flexibility:
    # params[0:7] : 7 polynomial coefficients
    # params[7:9] : RBF center
    # params[9]   : RBF width (isotropic)
    
    # Re-do with 7 polynomial terms:
    # 1, X, V, X2, V2, XV, X3
    
    poly = (params[0] + 
            params[1] * X + 
            params[2] * V + 
            params[3] * X2 + 
            params[4] * V2 + 
            params[5] * XV + 
            params[6] * X3)
    
    # RBF term
    cx = params[7]
    cv = params[8]
    sigma = np.abs(params[9]) + 0.1  # ensure minimum width
    
    dx = X - cx
    dv = V - cv
    r2 = (dx / sigma) ** 2 + (dv / sigma) ** 2
    
    # Gaussian RBF with amplitude 1 (the optimizer can adjust center/width)
    rbf = np.exp(-0.5 * r2)
    
    return poly + rbf
# EVOLVE-BLOCK-END
