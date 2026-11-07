"""Symbolic regression: stress-strain of an aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a smooth elastic-plastic transition.
    
    The model uses a soft-maximum of elastic and plastic responses to capture
    the yield point smoothly:
    
    sigma = smooth_max(E(T)*eps, sigma_y(T) + H(T)*(eps - eps_y(T)))
    
    Where:
    - E(T) = p0 + p1*T + p2*T^2  (elastic modulus)
    - sigma_y(T) = p3 + p4*T     (yield stress)
    - eps_y(T) = p5 + p6*T       (yield strain)
    - H(T) = p7 + p8*T           (plastic hardening modulus)
    - p9 is a smoothing parameter scale
    
    The soft-max uses: smooth_max(a, b) = 0.5*(a + b + (a-b)/beta * tanh(beta*(a-b)/2))
    with beta = p9, making it smooth and differentiable.
    
    For negative strains, we mirror the behavior (compression symmetry).
    """
    s = strain
    t = temp
    p = params
    
    # Handle sign: model tensile, mirror for compression
    sign = np.sign(s)
    s_abs = np.abs(s)
    
    # Elastic response
    E_T = p[0] + p[1] * t + p[2] * t * t
    elastic = E_T * s_abs
    
    # Plastic response: yield stress + hardening * (strain - yield strain)
    sigma_y = p[3] + p[4] * t
    eps_y = np.abs(p[5] + p[6] * t)  # ensure positive yield strain
    H_T = p[7] + p[8] * t
    
    # Plastic strain beyond yield
    eps_plastic = np.maximum(s_abs - eps_y, 0.0)
    plastic = sigma_y + H_T * eps_plastic
    
    # Soft maximum to smoothly transition between elastic and plastic
    # beta controls sharpness of transition
    beta = np.abs(p[9]) + 1e-6  # ensure positive and avoid division by zero
    
    # Compute soft max: 0.5*(a + b + (a-b)/beta * tanh(beta*(a-b)/2))
    diff = elastic - plastic
    # Scale diff for numerical stability
    scaled_diff = diff / beta
    # Clip to avoid overflow in tanh
    scaled_diff = np.clip(scaled_diff, -50, 50)
    
    soft_max = 0.5 * (elastic + plastic + diff * np.tanh(scaled_diff) / beta * beta)
    # Simplify: 0.5*(a+b) + 0.5*diff*tanh(diff/(2*beta)) * 2/beta * beta... 
    # Let me redo: smooth_max(a,b) = 0.5*(a+b) + 0.5*(a-b)/beta * tanh(beta*(a-b)/2)
    # = 0.5*(a+b) + 0.5*diff/beta * tanh(diff/(2*beta))
    
    smooth_beta = np.abs(p[9]) + 1.0  # scale for numerical stability
    arg = diff / (2.0 * smooth_beta)
    arg = np.clip(arg, -50, 50)
    soft_max = 0.5 * (elastic + plastic) + 0.5 * diff * np.tanh(arg) / smooth_beta * smooth_beta
    # Actually simpler: 
    soft_max = 0.5 * (elastic + plastic) + 0.5 * np.abs(diff) * np.tanh(arg)
    
    # Apply sign
    stress = sign * soft_max
    
    return stress
# EVOLVE-BLOCK-END
