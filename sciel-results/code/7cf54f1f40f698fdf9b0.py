import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a smooth elastic-plastic transition model with temperature dependence
    and polynomial correction terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    # Elastic modulus: E(T) = p0 * (1 + p3 * temp)
    # Yield stress: Y(T) = p1 * (1 + p4 * temp)
    # Transition sharpness: k(T) = p2 * (1 + p5 * temp)
    
    E = p0 * (1.0 + p3 * temp)
    Y = p1 * (1.0 + p4 * temp)
    k = p2 * (1.0 + p5 * temp)
    
    # Ensure k is positive for numerical stability
    k_safe = np.where(np.abs(k) > 0.01, k, np.sign(k) * 0.01)
    k_safe = np.where(k_safe == 0, 0.01, k_safe)
    
    # Core model: smooth transition from elastic to plastic
    # stress = E * strain * exp(-k*strain) + Y * (1 - exp(-k*strain))
    # For small strain: ~ (E + Y*k) * strain (linear)
    # For large strain: ~ Y (plateau)
    
    exp_term = np.exp(-k_safe * strain)
    
    # Handle potential overflow/underflow
    exp_term = np.clip(exp_term, 1e-10, 1e10)
    
    core = E * strain * exp_term + Y * (1.0 - exp_term)
    
    # Polynomial correction terms for fine-tuning
    # These capture residual patterns not explained by the core model
    poly_term = (
        p6 * strain +
        p7 * temp +
        p8 * strain * temp +
        p9 * strain**2
    )
    
    return core + poly_term
