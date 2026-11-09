"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # Model: Piecewise linear approximation using smooth ramp functions.
    # Parameters:
    # 0: a0 (intercept)
    # 1: a1 (linear strain slope)
    # 2: a2 (linear temp slope)
    # 3-5: c1, c2, c3 (strain breakpoints)
    # 6-8: s1, s2, s3 (strain slope changes)
    # 9: d1 (temp breakpoint)
    # Note: We use 3 strain ramps to better capture yield/hardening.
    # We use 1 temp ramp to capture temp dependence.
    # Total: 3 + 3 + 3 + 1 = 10 params.
    
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    
    c1 = params[3]
    c2 = params[4]
    c3 = params[5]
    
    s1 = params[6]
    s2 = params[7]
    s3 = params[8]
    
    d1 = params[9]
    
    eps = 1e-6
    
    def smooth_ramp(x):
        # Smooth approximation of max(0, x)
        # Using softplus-like form for differentiability
        return (x + np.sqrt(x**2 + eps)) / 2.0
    
    # Strain ramps
    r1 = smooth_ramp(strain - c1)
    r2 = smooth_ramp(strain - c2)
    r3 = smooth_ramp(strain - c3)
    
    # Temp ramp
    t_r1 = smooth_ramp(temp - d1)
    
    # Prediction
    pred = a0 + a1 * strain + a2 * temp + s1 * r1 + s2 * r2 + s3 * r3 + a2 * t_r1
    
    # Wait, I need to use a separate parameter for the temp ramp slope?
    # I have 10 params.
    # 0: a0
    # 1: a1
    # 2: a2
    # 3: c1
    # 4: c2
    # 5: c3
    # 6: s1
    # 7: s2
    # 8: s3
    # 9: d1
    # I am missing a slope for the temp ramp.
    # I can reuse a2 as the base temp slope, but the ramp needs its own slope.
    # Let's adjust:
    # 0: a0
    # 1: a1 (strain slope)
    # 2: a2 (temp slope)
    # 3: c1 (strain bp 1)
    # 4: c2 (strain bp 2)
    # 5: c3 (strain bp 3)
    # 6: s1 (strain slope change 1)
    # 7: s2 (strain slope change 2)
    # 8: s3 (strain slope change 3)
    # 9: t_slope (temp ramp slope)
    # But I need a temp breakpoint too.
    
    # Let's swap:
    # Use 2 strain breakpoints and 1 temp breakpoint + 1 temp slope?
    # Or 3 strain breakpoints and assume temp is linear?
    # The current model had 2 strain, 2 temp.
    # Stress-strain is more complex in strain.
    # Let's try:
    # 0: a0
    # 1: a1 (strain)
    # 2: a2 (temp)
    # 3: c1 (strain bp 1)
    # 4: c2 (strain bp 2)
    # 5: s1 (strain slope 1)
    # 6: s2 (strain slope 2)
    # 7: d1 (temp bp 1)
    # 8: d2 (temp bp 2)
    # 9: t1 (temp slope)
    # This is the current model.
    
    # Let's try to improve the strain part by having 3 breakpoints but dropping one temp breakpoint.
    # 0: a0
    # 1: a1 (strain)
    # 2: a2 (temp)
    # 3: c1 (strain bp 1)
    # 4: c2 (strain bp 2)
    # 5: c3 (strain bp 3)
    # 6: s1 (strain slope 1)
    # 7: s2 (strain slope 2)
    # 8: s3 (strain slope 3)
    # 9: t1 (temp ramp slope, with fixed bp? No, need bp)
    
    # If I fix the temp breakpoint to 0.5 (mid-range), I can use the 10th param for the slope.
    # Temp range is [0.06, 1]. Mid is ~0.5.
    # This is a reasonable heuristic.
    
    pred = a0 + a1 * strain + a2 * temp + s1 * r1 + s2 * r2 + s3 * r3 + params[9] * smooth_ramp(temp - 0.5)
    
    return pred
# EVOLVE-BLOCK-END
