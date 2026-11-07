"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Combines a smooth elastic-plastic structural model with rational basis functions
    for flexible shape fitting. Uses 10 fitted constants.
    """
    start_time = time.time()
    
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Safety guard for non-finite inputs
    if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(T)):
        return np.zeros_like(eps)
        
    # Pre-compute basis functions
    e2 = eps * eps
    e3 = e2 * eps
    
    # Rational basis for smooth saturation/hardening behavior
    denom = 1.0 + e2
    r1 = eps / denom       # ~eps for small eps, ~1/eps for large eps
    r2 = e3 / denom        # ~eps^3 for small eps, ~eps for large eps
    
    # Structural elastic-plastic model components
    # p0: Young's modulus (E)
    E = params[0]
    
    # p1: Yield strain scale
    # p2: Temperature dependence of yield strain
    eps_y = params[1] * (1.0 - params[2] * T)
    
    # Smooth yield transition using tanh
    # p3: Transition width (inverse steepness)
    transition_width = params[3]
    # Avoid division by zero in tanh
    if abs(transition_width) < 1e-12:
        transition_width = 1e-12
    
    yield_factor = 0.5 * (1.0 + np.tanh((eps - eps_y) / transition_width))
    
    # p4: Hardening modulus scale
    # p5: Temperature dependence of hardening
    H = params[4] * (1.0 - params[5] * T)
    
    # Plastic strain
    plastic_strain = np.maximum(eps - eps_y, 0.0)
    
    # Elastic stress component (approximated smoothly)
    elastic_stress = E * eps * (1.0 - yield_factor)
    
    # Plastic stress component
    yield_stress = E * eps_y
    plastic_stress = yield_stress + H * plastic_strain
    
    # Blend elastic and plastic
    structural_stress = elastic_stress * (1.0 - yield_factor) + plastic_stress * yield_factor
    
    # Additional flexible terms to capture residual patterns
    # p6: Offset
    # p7: Rational linear term coefficient
    # p8: Rational cubic term coefficient
    # p9: Temperature interaction term coefficient
    
    stress = (structural_stress +
              params[6] +
              params[7] * r1 +
              params[8] * r2 +
              params[9] * (T * r1))
    
    # Safety guard for non-finite outputs
    if not np.all(np.isfinite(stress)):
        stress = np.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        # In a search context, this would break. Here, it's just a safety check.
        pass
        
    return stress
# EVOLVE-BLOCK-END
