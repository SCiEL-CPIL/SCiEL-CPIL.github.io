"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth piecewise model with a soft-plus (smooth hinge) function
    to separate elastic and plastic regimes.
    
    Model:
        E(T) = p0 + p1 * T
        yield_strain(T) = p2 + p3 * T
        hardening_rate = p4 + p5 * T
        
        soft_plus(x, k) = log(1 + exp(k*x)) / k  (smooth approximation of max(0, x))
        
        stress = E(T) * strain 
                 + hardening_rate * soft_plus(strain - yield_strain(T), k)
                 + p6 * strain^2 + p7 * strain^3
                 + p8 * T * strain + p9 * T
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * temp
    
    # Temperature-dependent yield strain
    yield_strain = p2 + p3 * temp
    
    # Temperature-dependent hardening rate
    hardening_rate = p4 + p5 * temp
    
    # Smooth hinge: soft_plus(x, k) = log(1 + exp(k*x)) / k
    # Use a fixed smoothness parameter k=50 for numerical stability
    k = 50.0
    x = strain - yield_strain
    # Numerically stable soft plus
    soft_plus = np.logaddexp(0, k * x) / k
    
    # Elastic contribution
    elastic = E_T * strain
    
    # Plastic hardening contribution (only active beyond yield)
    plastic = hardening_rate * soft_plus
    
    # Additional polynomial corrections
    corrections = p6 * strain**2 + p7 * strain**3 + p8 * temp * strain + p9 * temp
    
    stress = elastic + plastic + corrections
    
    return stress
# EVOLVE-BLOCK-END
