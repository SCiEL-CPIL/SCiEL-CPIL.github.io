# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Smooth-hinge Ramberg-Osgood
    sigma = E(T)*strain + H(T)*softplus(strain - eps_y(T), delta)^p + C0 + C1*T + C2*T^2
    
    params:
    [0]: E_base, [1]: E_temp
    [2]: H_base, [3]: H_temp
    [4]: eps_y_base, [5]: eps_y_temp
    [6]: delta (smoothness), [7]: p (hardening exponent)
    [8]: C0 (offset), [9]: C1 (linear temp)
    """
    E0, E1 = params[0], params[1]
    H0, H1 = params[2], params[3]
    ey_base, ey_temp = params[4], params[5]
    delta, p = params[6], params[7]
    C0, C1 = params[8], params[9]
    
    # Temperature-dependent parameters
    E = E0 + E1 * temp
    H = H0 + H1 * temp
    eps_y = ey_base + ey_temp * temp
    
    # Smooth hinge: softplus(x, delta) = delta * log1p(exp(x/delta))
    # For x >> delta: ~ x; for x << -delta: ~ 0
    # This gives: 0 below yield, linear above yield, smooth transition
    x = strain - eps_y
    # Clip for numerical stability
    x_clipped = np.clip(x, -50.0 * delta, 50.0 * delta)
    smooth_hinge = delta * np.log1p(np.exp(x_clipped / delta))
    
    # Hardening term: power of the smooth hinge
    # Use abs for safety with negative values, but physically strain > eps_y
    hardening = np.power(np.maximum(smooth_hinge, 1e-12), p)
    
    # Combine
    stress = E * strain + H * hardening + C0 + C1 * temp
    
    return stress
# EVOLVE-BLOCK-END
