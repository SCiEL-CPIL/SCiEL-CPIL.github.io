"""Combined model: linear-in-parameters with physics-inspired fixed basis functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with physics-inspired fixed basis functions.
    
    The model is linear in the 10 parameters, making the optimization a convex
    problem that BFGS can solve globally and reliably.
    
    Basis functions are designed to capture the physics of aluminium stress-strain:
      - Elastic response (linear strain)
      - Temperature-modulated elasticity
      - Yield transition (tanh with fixed sharpness)
      - Strain hardening (strain * tanh)
      - Temperature effects on yield and hardening
      - Higher-order corrections
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Fixed sharpness for yield transition
    # Aluminium yield strain is typically ~0.001-0.01, so k ~ 100-1000
    k = 200.0
    
    # Precompute common terms
    eps2 = eps * eps
    eps3 = eps2 * eps
    T2 = T * T
    
    # Tanh-based yield transition (fixed non-linearity)
    tanh_eps = np.tanh(k * eps)
    
    # Build 10 physics-inspired basis functions
    # b0: linear elastic response
    b0 = eps
    
    # b1: temperature-modulated elastic response
    b1 = T * eps
    
    # b2: yield transition (saturates)
    b2 = tanh_eps
    
    # b3: temperature-modulated yield transition
    b3 = T * tanh_eps
    
    # b4: strain hardening (strain times yield transition)
    b4 = eps * tanh_eps
    
    # b5: temperature-modulated strain hardening
    b5 = T * eps * tanh_eps
    
    # b6: quadratic elastic correction
    b6 = eps2
    
    # b7: temperature-modulated quadratic correction
    b7 = T * eps2
    
    # b8: cubic elastic correction
    b8 = eps3
    
    # b9: higher-order temperature-strain cross term
    b9 = T2 * eps
    
    # Linear combination
    prediction = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
                  p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return prediction
# EVOLVE-BLOCK-END
