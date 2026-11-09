# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Structural change: piecewise-linear "sawtooth" basis with parameterized
    breakpoints via abs-based rectification, combined with a smooth tanh yield
    transition and temperature-dependent coefficients. This allows kinked
    work-hardening segments that BFGS can fit via parameterized breakpoints.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [0, 1] range for better conditioning
    s = (strain + 0.002) / 0.966
    t = (temp - 0.06667) / 0.93333
    
    # Temperature-dependent elastic modulus and yield point
    E = p0 * (1.0 + p1 * t)
    sigma_y = p2 * (1.0 - p3 * t)
    
    # Elastic strain limit
    eps_y = np.maximum(sigma_y / np.maximum(np.abs(E), 1e-8), 1e-6)
    
    # Plastic strain
    eps_p = np.maximum(s - eps_y, 0.0)
    
    # Smooth transition using tanh
    w = np.maximum(np.abs(p4), 0.005)
    x = (s - eps_y) / w
    x_clamped = np.clip(x, -15, 15)
    trans = 0.5 * (1.0 + np.tanh(x_clamped))
    
    # Elastic branch: linear with slight nonlinearity
    stress_elastic = E * s + p5 * s**2
    
    # Piecewise-linear "sawtooth" basis for work-hardening segments
    # Two breakpoints in plastic strain space, parameterized by p6 and p7
    bp1 = np.maximum(np.abs(p6), 0.01)  # first breakpoint
    bp2 = np.maximum(np.abs(p7), 0.05)  # second breakpoint, ensure bp2 > bp1
    
    # Ensure bp2 > bp1
    bp2 = np.maximum(bp2, bp1 + 0.01)
    
    # Rectified linear units (kinked basis)
    # Segment 1: active from 0 to bp1
    seg1 = np.maximum(eps_p, 0.0) - np.maximum(eps_p - bp1, 0.0)
    # Segment 2: active from bp1 to bp2
    seg2 = np.maximum(eps_p - bp1, 0.0) - np.maximum(eps_p - bp2, 0.0)
    # Segment 3: active beyond bp2
    seg3 = np.maximum(eps_p - bp2, 0.0)
    
    # Slopes for each segment (temperature-dependent)
    slope1 = p8 * (1.0 - 0.3 * t)  # initial hardening slope
    slope2 = p9 * (1.0 - 0.2 * t)  # intermediate hardening slope
    slope3 = 0.1 * p8 * (1.0 - 0.1 * t)  # final low hardening slope
    
    # Piecewise-linear hardening contribution
    hardening_pw = slope1 * seg1 + slope2 * seg2 + slope3 * seg3
    
    # Saturation component to prevent unbounded growth
    # Use a soft saturation based on total plastic strain
    sat_scale = np.maximum(p8, 0.1)
    saturation = sat_scale * (1.0 - np.exp(-np.maximum(eps_p, 0.0) / np.maximum(bp2, 0.1)))
    
    # Combine piecewise-linear hardening with saturation
    stress_plastic = sigma_y + hardening_pw + saturation
    
    # Blend elastic and plastic
    stress = trans * stress_plastic + (1.0 - trans) * stress_elastic
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
