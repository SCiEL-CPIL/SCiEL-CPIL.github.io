"""Piecewise-linear ramp model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear stress-strain model with multiple yield points.
    
    Model: stress = p0*strain + p1*max(0, strain-t1) + p2*max(0, strain-t2) 
                   + p3*max(0, strain-t3) + p4*max(0, strain-t4)
    where t1..t4 are yield strain thresholds (p5..p8),
    and p9 is a temperature-dependent scaling factor.
    
    All 10 params are fitted by BFGS.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, t1, t2, t3, t4, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent scaling: reduce effective stress at higher temps
    # p9 is a temperature sensitivity parameter
    temp_scale = 1.0 - p9 * temp
    
    # Elastic region (below all yield points)
    elastic = p0 * strain
    
    # Plastic hardening regions: each adds a ramp starting at its yield point
    ramp1 = p1 * np.maximum(0.0, strain - t1)
    ramp2 = p2 * np.maximum(0.0, strain - t2)
    ramp3 = p3 * np.maximum(0.0, strain - t3)
    ramp4 = p4 * np.maximum(0.0, strain - t4)
    
    # Combine all terms with temperature scaling
    prediction = (elastic + ramp1 + ramp2 + ramp3 + ramp4) * temp_scale
    
    return prediction
# EVOLVE-BLOCK-END
