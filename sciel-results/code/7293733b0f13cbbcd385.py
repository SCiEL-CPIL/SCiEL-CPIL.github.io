"""Structural change: piecewise-linear kinked basis for aluminum yield/hardening."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.

    Uses a kinked piecewise-linear basis to model:
      - Elastic region (linear in strain)
      - Yield point (kink at strain = p1, modulated by temp)
      - Work-hardening region (kink at strain = p2)
    
    Model:
      sigma = p0 + p1*strain 
            + p3 * relu(strain - p1*(1 + p4*temp))   # post-yield hardening
            + p5 * relu(strain - p2*(1 + p6*temp))   # secondary kink
            + p7 * strain * temp                       # bilinear correction
            + p8 * temp                                # temp offset
            + p9 * relu(p10 * (strain - 0.01))        # small-strain detail
    
    All 10 parameters are fitted by BFGS.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent yield strain and secondary kink strain
    # Use softplus-like: ensure breakpoints stay in reasonable range
    yield_strain = p1 * (1.0 + p4 * temp)
    kink2_strain = p2 * (1.0 + p6 * temp)
    
    # ReLU basis functions create kinks at the breakpoints
    relu1 = np.maximum(strain - yield_strain, 0.0)
    relu2 = np.maximum(strain - kink2_strain, 0.0)
    
    # Elastic + post-yield + secondary hardening
    elastic = p0 + p1 * strain
    post_yield = p3 * relu1
    secondary = p5 * relu2
    
    # Bilinear and temperature corrections
    bilinear = p7 * strain * temp
    temp_offset = p8 * temp
    
    # Small-strain detail term (captures initial nonlinearity)
    detail = p9 * np.maximum(strain - 0.005, 0.0)
    
    stress = elastic + post_yield + secondary + bilinear + temp_offset + detail
    
    return stress
# EVOLVE-BLOCK-END
