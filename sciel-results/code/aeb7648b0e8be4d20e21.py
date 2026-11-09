"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a "hat" function basis (piecewise linear triangles) constructed from
    abs operations to model yield drops, peaks, and work-hardening segments
    (zigzag behavior).
    
    params layout (10 constants):
      p0: Elastic modulus (slope)
      p1: Intercept (offset)
      p2: Temperature coefficient for modulus
      p3: Temperature coefficient for intercept
      p4: Amplitude of Hat 1
      p5: Center of Hat 1
      p6: Amplitude of Hat 2
      p7: Center of Hat 2
      p8: Amplitude of Hat 3
      p9: Center of Hat 3
    """
    s = strain
    t = temp
    p = params
    
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]
    p5 = p[5]
    p6 = p[6]
    p7 = p[7]
    p8 = p[8]
    p9 = p[9]
    
    # Normalize temperature to center around 0 for linear effects
    # t is in [0.066, 1], center approx 0.5
    t_centered = t - 0.5
    
    # Temperature-dependent linear terms
    # Modulus decreases with temperature (typically)
    modulus = p0 * (1.0 + p2 * t_centered)
    offset = p1 + p3 * t_centered
    
    # Base elastic response
    stress = modulus * s + offset
    
    # Hat function: max(0, W - |s - c|)
    # Constructed as: 0.5 * (W - |s - c| + |W - |s - c||)
    # We use fixed widths W1, W2, W3 for the three features.
    # These widths are chosen to be small enough to be localized but large enough
    # to be resolvable by the optimizer.
    
    W1 = 0.05
    W2 = 0.10
    W3 = 0.20
    
    # Hat 1: Likely the yield drop or initial plastic feature
    # |s - c|
    abs_s_c1 = np.abs(s - p5)
    hat1 = 0.5 * (W1 - abs_s_c1 + np.abs(W1 - abs_s_c1))
    
    # Hat 2: Likely work hardening peak
    abs_s_c2 = np.abs(s - p7)
    hat2 = 0.5 * (W2 - abs_s_c2 + np.abs(W2 - abs_s_c2))
    
    # Hat 3: Later stage feature
    abs_s_c3 = np.abs(s - p9)
    hat3 = 0.5 * (W3 - abs_s_c3 + np.abs(W3 - abs_s_c3))
    
    # Add hat contributions
    # Amplitudes p4, p6, p8 can be positive or negative to create peaks or valleys
    stress += p4 * hat1
    stress += p6 * hat2
    stress += p8 * hat3
    
    return stress
# EVOLVE-BLOCK-END
