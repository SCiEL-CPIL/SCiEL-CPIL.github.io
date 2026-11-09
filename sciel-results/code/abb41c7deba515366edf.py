"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # New design:
    # 1. Base linear strain term: a0 + a1 * strain
    # 2. Two strain ramps for yield and hardening: s1 * R(strain - c1) + s2 * R(strain - c2)
    # 3. Two independent temperature ramps: t1 * R(temp - d1) + t2 * R(temp - d2)
    # Total: 2 (base) + 4 (strain) + 4 (temp) = 10 params
    
    a0 = params[0]
    a1 = params[1]
    
    c1 = params[2]
    s1 = params[3]
    c2 = params[4]
    s2 = params[5]
    
    d1 = params[6]
    t1 = params[7]
    d2 = params[8]
    t2 = params[9]
    
    eps = 1e-6
    
    def smooth_ramp(x):
        return (x + np.sqrt(x**2 + eps)) / 2.0
    
    # Strain ramps
    r1 = smooth_ramp(strain - c1)
    r2 = smooth_ramp(strain - c2)
    
    # Temp ramps
    t_r1 = smooth_ramp(temp - d1)
    t_r2 = smooth_ramp(temp - d2)
    
    # Prediction
    pred = a0 + a1 * strain + s1 * r1 + s2 * r2 + t1 * t_r1 + t2 * t_r2
    
    return pred
# EVOLVE-BLOCK-END
